#!/usr/bin/env sh

python scripts/duplicate.py examples/fullimagenet/8parts/adam_train_val.prototxt 8
python scripts/duplicate.py examples/fullimagenet/8parts/adam_solver.prototxt 8
mkdir $1
cp examples/fullimagenet/8parts/adam_train_val.prototxt.template $1/.
cp examples/fullimagenet/8parts/adam_solver.prototxt.template $1/.
cp examples/fullimagenet/8parts/machinefile $1/.
cp examples/fullimagenet/8parts/ps_config $1/.
CURRENT_DIR=`pwd`
pdsh -R ssh -w h[0-7] "cd $CURRENT_DIR && ./build/tools/caffe_pdsh train --solver=examples/fullimagenet/8parts/adam_solver.prototxt --ps_config=examples/fullimagenet/8parts/ps_config --num_workers=8 --worker_id=%n" 2>&1 | tee $1/output.txt
