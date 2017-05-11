#!/bin/bash

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

python ../../../scripts/duplicate.py lstm_solver_RGB_googlenet.prototxt 8
python ../../../scripts/duplicate.py train_test_lstm_RGB_googlenet.prototxt 8
python ../../../scripts/duplicate.py sequence_input_layer.py 8 $ sequence_input_layer%i.py
mkdir $1
pwd > $1/pwd
git status > $1/git-status
git show | head -n 100 > $1/git-show
git diff > $1/git-diff
cp run_geeps_lstm_RGB_googlenet.sh $1/.
cp lstm_solver_RGB_googlenet.prototxt.template $1/.
cp train_test_lstm_RGB_googlenet.prototxt.template $1/.
cp sequence_input_layer.py.template $1/.
cp machinefile $1/.
cp ps_config_googlenet $1/.
pdsh -R ssh -w ^machinefile "cd $(pwd) && ../../../build/tools/caffe_geeps train --solver=lstm_solver_RGB_googlenet.prototxt --ps_config=ps_config_googlenet --weights=/panfs/probescratch/BigLearning/hengganc/results/rnn/16-0112-1118-vclass-googlenet-fine-tune/snapshots_googlenet_singleFrame_RGB_iter_5000.caffemodel --machinefile=machinefile --worker_id=%n" 2>&1 | tee $1/output.txt
