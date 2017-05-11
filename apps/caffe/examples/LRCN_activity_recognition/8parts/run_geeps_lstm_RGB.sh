#!/bin/bash

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

python ../../../scripts/duplicate.py lstm_solver_RGB.prototxt 8
python ../../../scripts/duplicate.py train_test_lstm_RGB.prototxt 8
python ../../../scripts/duplicate.py sequence_input_layer.py 8 $ sequence_input_layer%i.py
mkdir $1
pwd > $1/pwd
git status > $1/git-status
git show > $1/git-show
git diff > $1/git-diff
cp lstm_solver_RGB.prototxt.template $1/.
cp train_test_lstm_RGB.prototxt.template $1/.
cp sequence_input_layer.py.template $1/.
cp machinefile $1/.
cp ps_config $1/.
mpirun -machinefile machinefile ../../../build/tools/caffe_mpi train --solver=lstm_solver_RGB.prototxt --ps_config=ps_config --weights=../single_frame_all_layers_hyb_RGB_iter_5000.caffemodel 2>&1 | tee $1/output.txt
