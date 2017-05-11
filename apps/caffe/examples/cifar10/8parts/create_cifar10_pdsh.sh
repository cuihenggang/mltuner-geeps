#!/usr/bin/env sh

python scripts/duplicate.py examples/cifar10/8parts/create_cifar10.sh 8
pdsh -R ssh -w ^examples/cifar10/8parts/machinefile "cd $(pwd) && ./examples/cifar10/8parts/create_cifar10.sh.%n"
