#!/usr/bin/env sh

pdsh -R ssh -w ^examples/imagenet/16parts/machinefile "pkill -9 caffe_geeps"

python scripts/duplicate.py examples/imagenet/16parts/inception_train_val.prototxt 16
python scripts/duplicate.py examples/imagenet/16parts/inception_solver.prototxt 16

LOG=output.txt

if [ "$#" -eq 1 ]; then
  mkdir $1
  pwd > $1/pwd
  git status > $1/git-status
  git diff > $1/git-diff
  cp examples/imagenet/16parts/train_inception.sh $1/.
  cp examples/imagenet/16parts/inception_train_val.prototxt.template $1/.
  cp examples/imagenet/16parts/inception_solver.prototxt.template $1/.
  cp examples/imagenet/16parts/machinefile $1/.
  cp examples/imagenet/16parts/ps_config_inception $1/.
  LOG=$1/output.txt
fi

pdsh -R ssh -w ^examples/imagenet/16parts/machinefile "cd $(pwd) && ./build/tools/caffe_geeps train --solver=examples/imagenet/16parts/inception_solver.prototxt --ps_config=examples/imagenet/16parts/ps_config_inception --machinefile=examples/imagenet/16parts/machinefile --worker_id=%n" 2>&1 | tee ${LOG}
