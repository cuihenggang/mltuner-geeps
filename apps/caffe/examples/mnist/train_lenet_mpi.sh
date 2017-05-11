#!/usr/bin/env sh

mpirun -machinefile examples/mnist/8parts/machinefile ./build/tools/caffe_mpi train --solver=examples/mnist/8parts/lenet_solver.prototxt --ps_config=examples/mnist/8parts/ps_config

