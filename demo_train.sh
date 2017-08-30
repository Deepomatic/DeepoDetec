#!/bin/bash

python models/object_detection/train.py \
    --logtostderr \
    --gpu 1 \
    --train_dir /runs/demo_train \
    --pipeline_config_path configs/faster_rcnn_resnet101_fashion.config
