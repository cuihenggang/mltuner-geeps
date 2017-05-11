#!/usr/bin/env sh

echo "Begin"

# Figure out the paths.
script_path=`readlink -f $0`
echo ${script_path}
script_dir=`dirname $script_path`
echo ${script_dir}
example_dir=`dirname $script_dir`
echo ${example_dir}
app_dir=`dirname $example_dir`
echo ${app_dir}
progname=caffe_multinode
prog_path=${app_dir}/build/tools/${progname}
echo ${prog_path}

host_filename="/tank/projects/biglearning/hengganc/tmp/machine_file"
host_file=$(readlink -f $host_filename)
echo ${host_file}

##=====================================
## Parameters
##=====================================

# Input files:
solver_filename="${app_dir}/examples/mnist/lenet_solver.prototxt"
 # Uncomment if (re-)start training from a snapshot
#snapshot_filename="${app_dir}/examples/mnist/lenet_iter_100.solverstate"

config_path="/tank/projects/biglearning/hengganc/output"
##=====================================

ssh_options="-oStrictHostKeyChecking=no \
-oUserKnownHostsFile=/dev/null \
-oLogLevel=quiet"

# Parse hostfile
host_list=`cat $host_file | awk '{ print $2 }'`
unique_host_list=`cat $host_file | awk '{ print $2 }' | uniq`
num_unique_hosts=`cat $host_file | awk '{ print $2 }' | uniq | wc -l`

echo "cat $host_file | awk '{ print $2 }' | uniq"
echo ${num_unique_hosts}

# output_dir=$app_dir/output
# output_dir="${output_dir}/caffe.${dataset}.S${staleness}"
# output_dir="${output_dir}.M${num_unique_hosts}"
# output_dir="${output_dir}.T${num_app_threads}"
# log_dir=$output_dir/logs
# net_outputs_prefix="${output_dir}/${dataset}"

# Kill previous instances of this program
echo "Killing previous instances of '$progname' on servers, please wait..."
for ip in $unique_host_list; do
  ssh $ssh_options $ip \
    killall -q $progname
done
echo "All done!"

# Spawn program instances
client_id=0
for ip in $unique_host_list; do
  echo Running client $client_id on $ip
  log_path=${log_dir}.${client_id}

  cmd="mkdir -p ${output_dir}; \
      mkdir -p ${log_path}; \
      GLOG_logtostderr=false \
      GLOG_stderrthreshold=0 \
      GLOG_log_dir=$log_path \
      GLOG_v=-1 \
      GLOG_minloglevel=0 \
      GLOG_vmodule="" \
      $prog_path train \
      --solver=examples/mnist/lenet_solver.prototxt \
      --worker_id ${client_id} \
      --num_workers ${num_unique_hosts} \
      --ps_configs ${config_path}"

  echo $cmd

  ssh $ssh_options $ip $cmd &
  #eval $cmd  # Use this to run locally (on one machine).

  # Wait a few seconds for the name node (client 0) to set up
  if [ $client_id -eq 0 ]; then
    echo $cmd   # echo the cmd for just the first machine.
    echo "Waiting for name node to set up..."
    sleep 3
  fi
  client_id=$(( client_id+1 ))
done
