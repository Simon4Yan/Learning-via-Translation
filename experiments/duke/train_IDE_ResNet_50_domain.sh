#!/usr/bin/env bash

CAFFE=caffe
DATASET=duke
NET=ResNet_50
SNAPSHOTS_DIR=output/${DATASET}_train

LOG="experiments/logs/${DATASET}_market2duke_re-id_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

cd $(dirname ${BASH_SOURCE[0]})/../../

mkdir -p ${SNAPSHOTS_DIR}
mkdir -p experiments/logs/

GLOG_logtostderr=1 ${CAFFE}/build/tools/caffe train \
  -solver models/${DATASET}/${NET}_domain/${NET}_solver.prototxt \
  -gpu 0\
  -weights data/imagenet_models//${NET}.caffemodel  2>&1 | tee ${LOG}
