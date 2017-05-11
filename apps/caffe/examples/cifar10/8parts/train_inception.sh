#!/usr/bin/env sh

pdsh -R ssh -w ^examples/cifar10/8parts/machinefile "pkill -9 caffe_geeps"
pdsh -R ssh -w ^examples/cifar10/8parts/machinefile "pkill -9 python"

python scripts/duplicate.py examples/cifar10/8parts/inception_train_val.prototxt 8
python scripts/duplicate.py examples/cifar10/8parts/inception_solver.prototxt 8

LOG=output.txt

if [ "$#" -eq 1 ]; then
  mkdir $1
  pwd > $1/pwd
  git status > $1/git-status
  git diff > $1/git-diff
  git show | head -100 > $1/git-show
  cp examples/cifar10/8parts/train_inception.sh $1/.
  cp examples/cifar10/8parts/inception_train_val.prototxt.template $1/.
  cp examples/cifar10/8parts/inception_solver.prototxt.template $1/.
  cp examples/cifar10/8parts/machinefile $1/.
  cp examples/cifar10/8parts/ps_config_inception $1/.
  LOG=$1/output.txt
fi

pdsh -R ssh -w ^examples/cifar10/8parts/machinefile "cd $(pwd) && ./build/tools/caffe_geeps train --solver=examples/cifar10/8parts/inception_solver.prototxt --ps_config=examples/cifar10/8parts/ps_config_inception --machinefile=examples/cifar10/8parts/machinefile --output_dir=$1 --worker_id=%n" 2>&1 | tee ${LOG}
