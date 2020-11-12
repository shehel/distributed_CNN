import os
import threading
import time
from functools import wraps
import json
import socket

import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

from torchvision.models.resnet import Bottleneck


#########################################################
#           Define Model Parallel ResNet50              #
#########################################################

# In order to split the ResNet50 and place it on two different workers, we
# implement it in two model shards. The ResNetBase class defines common
# attributes and methods shared by two shards. ResNetShard1 and ResNetShard2
# contain two partitions of the model layers respectively.


num_classes = 1000


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetBase(nn.Module):
    def __init__(self, block, inplanes, num_classes=1000,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNetBase, self).__init__()

        self._lock = threading.Lock()
        self._block = block
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

    def _make_layer(self, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * self._block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * self._block.expansion, stride),
                norm_layer(planes * self._block.expansion),
            )

        layers = []
        layers.append(self._block(self.inplanes, planes, stride, downsample, self.groups,
                                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * self._block.expansion
        for _ in range(1, blocks):
            layers.append(self._block(self.inplanes, planes, groups=self.groups,
                                      base_width=self.base_width, dilation=self.dilation,
                                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class ResNetShard1(ResNetBase):
    """
    The first part of ResNet.
    """
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard1, self).__init__(
            Bottleneck, 64, num_classes=num_classes, *args, **kwargs)

        self.device = device
        self.seq = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 3),
            self._make_layer(128, 4, stride=2)
        ).to(self.device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out =  self.seq(x)
        return out.cpu()


class ResNetShard2(ResNetBase):
    """
    The second part of ResNet.
    """
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard2, self).__init__(
            Bottleneck, 512, num_classes=num_classes, *args, **kwargs)

        self.device = device
        self.seq = nn.Sequential(
            self._make_layer(256, 6, stride=2),
            self._make_layer(512, 3, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).to(self.device)

        self.fc =  nn.Linear(512 * self._block.expansion, num_classes).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.fc(torch.flatten(self.seq(x), 1))
        return out.cpu()


class DistResNet50(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """
    def __init__(self, split_size, workers, *args, **kwargs):
        super(DistResNet50, self).__init__()

        self.split_size = split_size

        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            ResNetShard1,
            args = ("cpu:0",) + args,
            # args = args,
            kwargs = kwargs
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            ResNetShard2,
            args = ("cpu:0",) + args,
            # args = args,
            kwargs = kwargs
        )

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params


#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


def run_master(split_size):

    # put the two model parts on worker1 and worker2 respectively
    model = DistResNet50(split_size, ["worker1", "worker2"])
    loss_fn = nn.MSELoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    for i in range(num_batches):
        print(f"Processing batch {i}")
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                      .scatter_(1, one_hot_indices, 1)

        # The distributed autograd context is the dedicated scope for the
        # distributed backward pass to store gradients, which can later be
        # retrieved using the context_id by the distributed optimizer.
        with dist_autograd.context() as context_id:
            outputs = model(inputs)
            dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
            opt.step(context_id)


def run_worker(rank, world_size, num_split):
    # os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_ADDR'] = '172.10.0.2'
    # os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_PORT'] = '12345'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    """
    # source: https://pytorch.org/tutorials/intermediate/dist_tuto.html
    dist.init_process_group(
    init_method='tcp://10.1.1.20:23456',
    rank=args.rank,
    world_size=4)
    """

    # block until all rpcs finish
    rpc.shutdown()

"""
def main():
  per_worker_batch_size = 64

  # read tf_config from json file
  with open('/root/TF_CONFIG.json') as f:
    tf_config = json.load(f)


  # modify worker index based on assigned ip_address
  tf_config['task']['index'] = getIndex(tf_config)

  os.environ['TF_CONFIG'] = json.dumps(tf_config)

  # specify number of workers as length of worker list in tf_config
  num_workers = len(tf_config['cluster']['worker'])


  # create a MirroredStartegy
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.RING)

  global_batch_size = per_worker_batch_size * num_workers
  multi_worker_dataset = mnist.mnist_dataset(global_batch_size)

  # open a strategy scope
  with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = mnist.build_and_compile_cnn_model()

  # train the model on all available workers
  multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)

  print('Worker_' +str(tf_config['task']['index'])+ ': Finished training my batch')


def getIPAddress():
    # extract ip address of container
    try:
        node_name = socket.gethostname()
        ip = socket.gethostbyname(node_name)
    except:
        print("Unable to get Hostname and IP")
    return ip


# returns the index of the worker based on extracted ip address
def getIndex(tf_config):
    # extract the ip address of the container
    ip = getIPAddress()
    # if ip address is static one assigned to chief-worker, then index=0
    if ip == '172.10.0.2':
        index = 0
    # otherwise, get the index of the ip address in the tf_config's list
    else:
        # concatenate ip address with port number
        ipStr = ip + ':23456'
        # get the index of the matching ip:port string in the tf_config's list
        index = tf_config['cluster']['worker'].index(ipStr)

    print('Extracted index=' + str(index) + ' from IPAddr=' + ip + '\n')
    return index
"""


if __name__=="__main__":
    world_size = 3
    for num_split in [1, 2, 4, 8]:
        tik = time.time()
        mp.spawn(run_worker, args=(world_size, num_split), nprocs=world_size, join=True)
        tok = time.time()
        print(f"number of splits = {num_split}, execution time = {tok - tik}")