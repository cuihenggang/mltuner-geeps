#!/usr/bin/env sh

python scripts/duplicate.py examples/imagenet/8parts/googlenet_train_val.prototxt 8
python scripts/duplicate.py examples/imagenet/8parts/googlenet_solver.prototxt 8
mkdir $1
pwd > $1/pwd
git status > $1/git-status
git show > $1/git-show
git diff > $1/git-diff
cp examples/imagenet/train_googlenet.sh $1/.
cp examples/imagenet/8parts/googlenet_train_val.prototxt.template $1/.
cp examples/imagenet/8parts/googlenet_solver.prototxt.template $1/.
cp examples/imagenet/8parts/machinefile $1/.
cp examples/imagenet/8parts/ps_config $1/.
mpirun -machinefile examples/imagenet/8parts/machinefile ./build/tools/caffe_mpi train --solver=examples/imagenet/8parts/googlenet_solver.prototxt --ps_config=examples/imagenet/8parts/ps_config --snapshot=/panfs/probescratch/BigLearning/hengganc/results/16-0216-1048-googlenet-from-16-0211-1142-lr0004/googlenet_snapshot_iter_160000 2>&1 | tee $1/output.txt
