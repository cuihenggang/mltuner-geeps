#!/usr/bin/env sh

pdsh -R ssh -w ^examples/cifar10/8parts/machinefile "pkill caffe_geeps"
