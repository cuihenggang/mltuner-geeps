#!/usr/bin/env sh

mkdir $1
cp examples/imagenet/1part/caffenet_train_val.prototxt $1/.
cp examples/imagenet/1part/caffenet_solver.prototxt $1/.
./build/tools/caffe train --solver=examples/imagenet/1part/caffenet_solver.prototxt 2>&1 | tee $1/output.txt

