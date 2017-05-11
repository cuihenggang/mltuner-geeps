#!/usr/bin/env sh

python scripts/duplicate.py examples/fullimagenet/32parts/adam_train_val.prototxt 32
python scripts/duplicate.py examples/fullimagenet/32parts/adam_solver.prototxt 32
mkdir $1
cp examples/fullimagenet/32parts/adam_train_val.prototxt.template $1/.
cp examples/fullimagenet/32parts/adam_solver.prototxt.template $1/.
cp examples/fullimagenet/32parts/machinefile $1/.
cp examples/fullimagenet/32parts/ps_config $1/.
mpirun -machinefile examples/fullimagenet/32parts/machinefile ./build/tools/caffe_mpi train --solver=examples/fullimagenet/32parts/adam_solver.prototxt --ps_config=examples/fullimagenet/32parts/ps_config 2>&1 | tee $1/output.txt

