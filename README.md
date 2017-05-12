# MLtuner-GeePS

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

MLtuner-GeePS is a parameter server library that scales single-machine GPU machine learning applications (such as Caffe) to a cluster of machines and automatically tunes the tunables for the training.


## Download and build MLtuner-GeePS and Caffe application

Run the following command to download MLtuner-GeePS and (our slightly modified) Caffe:

```
git clone https://github.com/cuihenggang/mltuner-geeps.git
```

If you use the Ubuntu 16.04 system, you can run the following commands (from geeps root directory) to install the dependencies:

```
./scripts/install-geeps-deps-ubuntu16.sh
./scripts/install-caffe-deps-ubuntu16.sh
```

Also, please make sure your CUDA library is installed in `/usr/local/cuda`.

After installing the dependencies, you can build MLtuner-GeePS by simply running this command from mltuner-geeps root directory:

```
scons -j8
```

You can then build (our slightly modified) Caffe by first entering the `apps/caffe` directory and then running `make -j8`:

```
cd apps/caffe
make -j8
```


## CIFAR-10 example on two machines

You can run Caffe distributedly across a cluster of machines with MLtuner-GeePS. In this section, we will demonstrate the steps to run Caffe's CIFAR-10 example on two machines.

All commands in this section are executed from the `apps/caffe` directory:

```
cd apps/caffe
```

You will first need to prepare a machine file as `examples/cifar10/2parts/machinefile`, with each line being the host name of one machine. Since we use two machines in this example, this machine file should have two lines, such as:

```
host0
host1
```

We will use `pdsh` to launch commands on those machines with the `ssh` protocol, so please make sure that you can `ssh` to those machines without password.

When you have your machine file in ready, you can run the following command to download and prepare the CIFAR-10 dataset:

```
./data/cifar10/get_cifar10.sh
./examples/cifar10/2parts/create_cifar10_pdsh.sh
```

Our script will partition the datasets into two parts, one for each machine. You can then train an Inception network on it with this command:

```
./examples/cifar10/2parts/train_inception.sh YOUR_OUTPUT_DIRECTORY
```


## ImageNet with Inception-BN


Please look at our [wiki](https://github.com/cuihenggang/geeps/wiki) for more details. Happy training!


## Reference Paper

Henggang Cui, Hao Zhang, Gregory R. Ganger, Phillip B. Gibbons, and Eric P. Xing.
[GeePS: Scalable Deep Learning on Distributed GPUs with a GPU-Specialized Parameter Server](https://users.ece.cmu.edu/~hengganc/archive/paper/[eurosys16]geeps.pdf).
In ACM European Conference on Computer Systems, 2016 (EuroSys'16)
