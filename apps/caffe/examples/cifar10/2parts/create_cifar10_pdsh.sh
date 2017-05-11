#!/usr/bin/env sh

python scripts/duplicate.py examples/cifar10/2parts/create_cifar10.sh 2
pdsh -R ssh -w ^examples/cifar10/2parts/machinefile "cd $(pwd) && ./examples/cifar10/2parts/create_cifar10.sh.%n"
