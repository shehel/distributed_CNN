FROM python:3.8

#RUN pip install numpy Keras tensorflow
RUN    pip3 install torch && \
    pip3 install torchvision && \
    pip3 install torch torchvision torchaudio


ADD run_main.sh /root/run_main.sh
RUN chmod +x /root/run_main.sh


CMD ["/root/run_main.sh"]
