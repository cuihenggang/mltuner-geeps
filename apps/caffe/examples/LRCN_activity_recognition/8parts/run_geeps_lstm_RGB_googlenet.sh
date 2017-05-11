#!/bin/bash

EXP_PATH=examples/LRCN_activity_recognition/8parts

python ./scripts/duplicate.py $EXP_PATH/lstm_solver_RGB_googlenet.prototxt 8
python ./scripts/duplicate.py $EXP_PATH/train_test_lstm_RGB_googlenet.prototxt 8
python ./scripts/duplicate.py $EXP_PATH/sequence_input_layer.py 8 $ $EXP_PATH/sequence_input_layer%i.py
mkdir $1
pwd > $1/pwd
git status > $1/git-status
git show | head -n 100 > $1/git-show
git diff > $1/git-diff
cp $EXP_PATH/run_geeps_lstm_RGB_googlenet.sh $1/.
cp $EXP_PATH/lstm_solver_RGB_googlenet.prototxt.template $1/.
cp $EXP_PATH/train_test_lstm_RGB_googlenet.prototxt.template $1/.
cp $EXP_PATH/sequence_input_layer.py.template $1/.
cp $EXP_PATH/machinefile $1/.
cp $EXP_PATH/ps_config_googlenet $1/.
pdsh -R ssh -w ^$EXP_PATH/machinefile "cd $(pwd) && HDF5_DISABLE_VERSION_CHECK=1 PYTHONPATH=$EXP_PATH ./build/tools/caffe_geeps train --solver=$EXP_PATH/lstm_solver_RGB_googlenet.prototxt --ps_config=$EXP_PATH/ps_config_googlenet --weights=/panfs/probescratch/BigLearning/hengganc/results/rnn/16-0112-1118-vclass-googlenet-fine-tune/snapshots_googlenet_singleFrame_RGB_iter_5000.caffemodel --machinefile=$EXP_PATH/machinefile --worker_id=%n" 2>&1 | tee $1/output.txt
