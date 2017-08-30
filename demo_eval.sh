#!/bin/bash

python models/object_detection/eval.py \
    --lodtostderr \
    --checkpoint_dir /runs/demo_train \
    --eval_dir /runs/demo_eval \
    --pipeline_config_path configs/faster_rcnn_resnet101_fashion.config \
    --gpu 2
