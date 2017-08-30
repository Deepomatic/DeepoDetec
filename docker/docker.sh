#!/bin/bash
# get flags
data_dir=''
runs_dir=''
pretrained_dir=''

while getopts 'p:d:r:' flag; do
  case "${flag}" in
    d) data_dir="${OPTARG}" ;;
    r) runs_dir="${OPTARG}" ;;
    p) pretrained_dir="${OPTARG}" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

# get wd of main repo
PWD=`pwd | rev | cut -d / -f2- | rev`

nvidia-docker build -t "tf_detection" .
nvidia-docker run \
    -v $PWD:/home \
    -v $data_dir:/data \
    -v $runs_dir:/runs \
    -v $pretrained_dir:/pretrained \
    -p 6006:6006 \
    -ti tf_detection
