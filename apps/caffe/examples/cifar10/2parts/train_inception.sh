#!/usr/bin/env sh

pdsh -R ssh -w ^examples/cifar10/2parts/machinefile "pkill caffe_geeps"

python scripts/duplicate.py examples/cifar10/2parts/inception_train_val.prototxt 2
python scripts/duplicate.py examples/cifar10/2parts/inception_solver.prototxt 2

LOG=output.txt

if [ "$#" -eq 1 ]; then
  mkdir $1
  pwd > $1/pwd
  git status > $1/git-status
  git diff > $1/git-diff
  cp examples/cifar10/2parts/train_inception.sh $1/.
  cp examples/cifar10/2parts/inception_train_val.prototxt.template $1/.
  cp examples/cifar10/2parts/inception_solver.prototxt.template $1/.
  cp examples/cifar10/2parts/machinefile $1/.
  cp examples/cifar10/2parts/ps_config_inception $1/.
  LOG=$1/output.txt
fi

pdsh -R ssh -w ^examples/cifar10/2parts/machinefile "cd $(pwd) && ./build/tools/caffe_geeps train --solver=examples/cifar10/2parts/inception_solver.prototxt --ps_config=examples/cifar10/2parts/ps_config_inception --machinefile=examples/cifar10/2parts/machinefile --worker_id=%n" 2>&1 | tee ${LOG}
