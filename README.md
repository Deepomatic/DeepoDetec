# DeepoDetec

Repository for tensorflow detection for Deepomatic. 

It is a slightly modified version of the [Tensorflow Detection API](https://github.com/tensorflow/models/blob/master/object_detection). The tutorials there apply for this, except _make_tf_record.py_ can be used to create datasets from Deepomatic json format.
Pretrained checkpoints for fine-tuning can be found [here](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)

## Getting started

After cloning the repository, run the following commands, replacing the mounted volumes with the location of your data:
```bash
cd docker
./docker.sh \
  -p /path/to/pretrained/models \
  -d /path/to/datasets \
  -r /path/to/save/runs
```

This will build the docker with everything you need.
One more configuration step is needed:
```bash
# enter the docker running tensorflow-detection
./enter_docker.sh
protoc object_detection/protos/*.proto --python_out=.
```

This will compile all the protos that the API depends on.

## Usage

First you will need to create your datasets, using the provided _make_tf_record.py_. Once it has been created, and all your data is mounted as indicated above, you are ready to start training. From the docker (which can be entered through _docker/enter_docker.sh_), simply use _models/object_detection/train.py_ and _eval.py_ as indicated in the documentation.
For your convenience, a demo script is provided in _demo_train.sh_ and _demo_eval.py_.

