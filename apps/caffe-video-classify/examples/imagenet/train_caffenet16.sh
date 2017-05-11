#!/usr/bin/env sh

python scripts/duplicate.py examples/imagenet/16parts/caffenet_train_val.prototxt 16
python scripts/duplicate.py examples/imagenet/16parts/caffenet_solver.prototxt 16
mkdir $1
git status > $1/git-status
git show > $1/git-show
git diff > $1/git-diff
cp examples/imagenet/16parts/caffenet_train_val.prototxt.template $1/.
cp examples/imagenet/16parts/caffenet_solver.prototxt.template $1/.
cp examples/imagenet/16parts/machinefile $1/.
cp examples/imagenet/16parts/ps_config $1/.
mpirun -machinefile examples/imagenet/16parts/machinefile ./build/tools/caffe_mpi train --solver=examples/imagenet/16parts/caffenet_solver.prototxt --ps_config=examples/imagenet/16parts/ps_config 2>&1 | tee $1/output.txt

