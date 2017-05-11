#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_RGB_googlenet.prototxt -weights /panfs/probescratch/BigLearning/hengganc/results/16-0112-1118-vclass-googlenet-fine-tune/snapshots_googlenet_singleFrame_RGB_iter_5000.caffemodel  
echo "Done."
