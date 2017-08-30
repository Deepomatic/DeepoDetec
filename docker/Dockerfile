FROM tensorflow/tensorflow:1.1.0-gpu

LABEL maintainer "Arthur Wilcke <arthur@deepomatic.com>"

RUN apt update && \
    apt install -y git protobuf-compiler python-pil python-lxml

RUN pip install progressbar

VOLUME /home
VOLUME /runs
VOLUME /pretrained

WORKDIR /home/models

#RUN protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH $PYTHONPATH:/home/models:/home/models/slim

ENTRYPOINT ["tensorboard","--logdir","/runs"]
