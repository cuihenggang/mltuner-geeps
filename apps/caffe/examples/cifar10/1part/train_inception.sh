#!/usr/bin/env sh

pdsh -R ssh -w ^examples/cifar10/1part/machinefile "pkill caffe_geeps"

python scripts/duplicate.py examples/cifar10/1part/inception_train_val.prototxt 1
python scripts/duplicate.py examples/cifar10/1part/inception_solver.prototxt 1

LOG=output.txt

if [ "$#" -eq 1 ]; then
  mkdir $1
  pwd > $1/pwd
  git status > $1/git-status
  git diff > $1/git-diff
  cp examples/cifar10/1part/train_inception.sh $1/.
  cp examples/cifar10/1part/inception_train_val.prototxt.template $1/.
  cp examples/cifar10/1part/inception_solver.prototxt.template $1/.
  cp examples/cifar10/1part/machinefile $1/.
  cp examples/cifar10/1part/ps_config_inception $1/.
  LOG=$1/output.txt
fi

pdsh -R ssh -w ^examples/cifar10/1part/machinefile "cd $(pwd) && ./build/tools/caffe_geeps train --solver=examples/cifar10/1part/inception_solver.prototxt --ps_config=examples/cifar10/1part/ps_config_inception --machinefile=examples/cifar10/1part/machinefile --worker_id=%n" 2>&1 | tee ${LOG}
