#!/bin/bash

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

python ../../scripts/duplicate.py 8parts/lstm_solver_RGB.prototxt 8
python ../../scripts/duplicate.py 8parts/train_test_lstm_RGB.prototxt 8
mkdir $1
git status > $1/git-status
git show > $1/git-show
git diff > $1/git-diff
cp 8parts/lstm_solver_RGB.prototxt.template $1/.
cp 8parts/train_test_lstm_RGB.prototxt.template $1/.
cp 8parts/machinefile $1/.
cp 8parts/ps_config $1/.
mpirun -machinefile 8parts/machinefile ../../build/tools/caffe_mpi train --solver=8parts/lstm_solver_RGB.prototxt --ps_config=8parts/ps_config --weights=single_frame_all_layers_hyb_RGB_iter_5000.caffemodel
# 2>&1 | tee $1/output.txt
