#!/usr/bin/env sh

python scripts/duplicate.py examples/imagenet/8parts/adam_train_val.prototxt 8
python scripts/duplicate.py examples/imagenet/8parts/adam_solver.prototxt 8
mkdir $1
cp examples/imagenet/8parts/adam_train_val.prototxt.template $1/.
cp examples/imagenet/8parts/adam_solver.prototxt.template $1/.
cp examples/imagenet/8parts/machinefile $1/.
cp examples/imagenet/8parts/ps_config $1/.
mpirun -machinefile examples/imagenet/8parts/machinefile ./build/tools/caffe_mpi train --solver=examples/imagenet/8parts/adam_solver.prototxt --ps_config=examples/imagenet/8parts/ps_config 2>&1 | tee $1/output.txt

