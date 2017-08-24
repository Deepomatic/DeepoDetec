#!/bin/bash

nvidia-docker build -t "tf_detection" .
nvidia-docker run -v /home/arthur/tf:/home/arthur/tf -v /mnt/datasets:/data -ti -p 6006:6006 tf_detection bash
