#FROM ubuntu:20.04
FROM python:3.8
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
#&& apt-get install -y python3-pip

#RUN mkdir /fireDetection
#WORKDIR /fireDetection
#COPY . /fireDetection/

#RUN pip3 install tensorflow && \
#    pip3 install Keras && \
#    pip3 install numpy jupyter && \
#    pip3 install torch && \
#    pip3 install torchvision

RUN pip3 install torch && \
    pip3 install torchvision && \
    pip3 install torch torchvision torchaudio


ADD run_main.sh /root/run_main.sh
RUN chmod +x /root/run_main.sh

CMD ["/root/run_main.sh"]

