#!/usr/bin/env sh

python scripts/duplicate.py examples/fullimagenet/8parts/mynet_train_val.prototxt 8
python scripts/duplicate.py examples/fullimagenet/8parts/mynet_solver.prototxt 8
mkdir $1
git status > $1/git-status
git show > $1/git-show
git diff > $1/git-diff
cp examples/fullimagenet/8parts/mynet_train_val.prototxt.template $1/.
cp examples/fullimagenet/8parts/mynet_solver.prototxt.template $1/.
cp examples/fullimagenet/8parts/machinefile $1/.
cp examples/fullimagenet/8parts/ps_config $1/.
mpirun -machinefile examples/fullimagenet/8parts/machinefile ./build/tools/caffe_mpi train --solver=examples/fullimagenet/8parts/mynet_solver.prototxt --ps_config=examples/fullimagenet/8parts/ps_config_mynet 2>&1 | tee $1/output.txt

