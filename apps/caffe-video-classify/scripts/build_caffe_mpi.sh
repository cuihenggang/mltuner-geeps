#!/bin/bash

CURRENT_DIR=`pwd`

mpic++ mpi/caffe_mpi.cpp -o build/tools/caffe_mpi -pthread -fPIC -DNDEBUG -O2 -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/include -I.build_release/src -I./src -I./include -I/usr/local/cuda/include -I../../lazy-table-module/include -Wall -Wno-sign-compare  -lcaffe -Wl,-rpath,$CURRENT_DIR/build/lib -Wl,-rpath,$CURRENT_DIR/../../flat-map -L/usr/lib -L/usr/local/lib -L/usr/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -L.build_release/lib -L../../flat-map -lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lboost_program_options -lhdf5_hl -lhdf5 -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_thread -lstdc++ -lcudnn -lcblas -latlas -llazytablemodule_shared
