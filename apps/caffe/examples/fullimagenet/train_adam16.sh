#!/usr/bin/env sh

python scripts/duplicate.py examples/fullimagenet/16parts/adam_train_val.prototxt 16
python scripts/duplicate.py examples/fullimagenet/16parts/adam_solver.prototxt 16
mkdir $1
cp examples/fullimagenet/16parts/adam_train_val.prototxt.template $1/.
cp examples/fullimagenet/16parts/adam_solver.prototxt.template $1/.
cp examples/fullimagenet/16parts/machinefile $1/.
cp examples/fullimagenet/16parts/ps_config $1/.
mpirun -machinefile examples/fullimagenet/16parts/machinefile ./build/tools/caffe_mpi train --solver=examples/fullimagenet/16parts/adam_solver.prototxt --ps_config=examples/fullimagenet/16parts/ps_config 2>&1 | tee $1/output.txt

