FROM tensorflow/tensorflow:1.1.0-gpu

LABEL maintainer "Arthur Wilcke <arthur@deepomatic.com>"

RUN apt update && \
    apt install -y git protobuf-compiler python-pil python-lxml

VOLUME /home/arthur/tf
WORKDIR /home/arthur/tf

ENV PYTHONPATH $PYTHONPATH:/home/arthur/tf/models:/home/arthur/tf/models/slim
