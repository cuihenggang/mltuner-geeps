import time
from hyperopt import fmin, tpe, hp

def my_func(input):
  print input
  print len(input)
  output_fd = open('/users/hengganc/hyperopt/file.txt', 'a')
  for x in input:
    output_fd.write('%f ' % x)
  output_fd.write('P\n')
  output_fd.close()
  while True:
    input_fd = open('/users/hengganc/hyperopt/file.txt', 'r')
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

search_space = (hp.uniform('x', -10, 10), hp.uniform('y', -10, 10))

best = fmin(fn=my_func,
    space=search_space,
    algo=tpe.suggest,
    max_evals=100)
print best
