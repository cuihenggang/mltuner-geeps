import time
import os
import sys
from hyperopt import fmin, tpe, hp

data_file = sys.argv[1]
# data_file = '/users/hengganc/nfs/output/17-0302-1233-test-hyperopt/results.dat'

def my_func(input):
  print input
  output_fd = open(data_file, 'a')
  for x in input:
    output_fd.write('%f ' % x)
  output_fd.write('P\n')
  output_fd.close()
  while True:
    input_fd = open(data_file, 'r')
    last_line = ''
    for line in input_fd:
      last_line = line
    strs = last_line.split()
    if len(strs) < len(input) + 1 or strs[len(input)] == 'P':
      input_fd.close()
      time.sleep(1)
      continue
    else:
      input_fd.close()
      return float(strs[len(input)])

search_space = hp.choice('tunable_id',
  [
    (0, hp.uniform('lr', -5, 0)),
    (1, hp.quniform('slack', 0, 3, 1)),
    (2, hp.quniform('batchsize', 0, 4, 1)),
    (3, hp.uniform('momentum', 0, 1)),
    (4, hp.uniform('lr2', -5, 0)),
    (5, hp.quniform('slack2', 0, 3, 1)),
    (6, hp.quniform('batchsize2', 0, 4, 1)),
    (7, hp.uniform('momentum2', 0, 1))
  ])
# search_space = (hp.uniform('lr', -5, 0), hp.quniform('slack', 0, 3, 1), hp.quniform('batchsize', 0, 3, 1))

os.system('rm %s' % data_file)
best = fmin(fn=my_func,
    space=search_space,
    algo=tpe.suggest,
    max_evals=10000000)
print best
