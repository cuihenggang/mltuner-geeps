#!/bin/bash

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

python ../../../scripts/duplicate.py lstm_solver_RGB_googlenet.prototxt 16
python ../../../scripts/duplicate.py train_test_lstm_RGB_googlenet.prototxt 16
python ../../../scripts/duplicate.py sequence_input_layer.py 16 $ sequence_input_layer%i.py
mkdir $1
pwd > $1/pwd
git status > $1/git-status
git show > $1/git-show
git diff > $1/git-diff
cp run_geeps_lstm_RGB_googlenet.sh $1/.
cp lstm_solver_RGB_googlenet.prototxt.template $1/.
cp train_test_lstm_RGB_googlenet.prototxt.template $1/.
cp sequence_input_layer.py.template $1/.
cp machinefile $1/.
cp ps_config_googlenet $1/.
mpirun -machinefile machinefile ../../../build/tools/caffe_mpi train --solver=lstm_solver_RGB_googlenet.prototxt --ps_config=ps_config_googlenet --weights=/panfs/probescratch/BigLearning/hengganc/results/16-0112-1118-vclass-googlenet-fine-tune/snapshots_googlenet_singleFrame_RGB_iter_5000.caffemodel 2>&1 | tee $1/output.txt
