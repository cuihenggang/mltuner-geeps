#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_RGB.prototxt -weights RGB_lstm_model_iter_30000.caffemodel
echo "Done."
