import os

host_file_name = "/tank/projects/biglearning/hengganc/tmp/machine_file"
host_file_fd = open(host_file_name)
workers = []
num_workers = 0
for line in host_file_fd:
  workers.append(line)
  num_workers = num_workers + 1
host_file_fd.close()

ssh_options = "-oStrictHostKeyChecking=no \
              -oUserKnownHostsFile=/dev/null \
              -oLogLevel=quiet"
program_path = "/tank/projects/biglearning/hengganc/LazyTable/applications/caffe/build/tools/caffe_multinode"
log_path = "/tank/projects/biglearning/hengganc/tmp/log"
ps_configs = "/tank/projects/biglearning/hengganc/output"

for worker_id in range(num_workers):
  cmd = "GLOG_logtostderr=false \
        GLOG_stderrthreshold=0 \
        GLOG_log_dir=%s \
        GLOG_v=-1 \
        GLOG_minloglevel=0 \
        GLOG_vmodule="" \
        %s train \
        --solver=examples/mnist/lenet_solver.prototxt \
        --worker_id %i \
        --num_workers %i \
        --ps_configs %s" \
    % (log_path, program_path, worker_id, num_workers, ps_configs)

  ssh_cmd = "ssh %s %s %s &" % (ssh_options, workers[worker_id], cmd)
  print ssh_cmd
  os.system(ssh_cmd)
