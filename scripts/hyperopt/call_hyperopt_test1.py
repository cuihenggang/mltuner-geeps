import time
from hyperopt import fmin, tpe, hp

def my_func(x):
  print 'sample %f' % x
  output_fd = open('/users/hengganc/hyperopt/file.txt', 'a')
  output_fd.write('%f P\n' % x)
  output_fd.close()
  while True:
    input_fd = open('/users/hengganc/hyperopt/file.txt', 'r')
    last_line = ''
    for line in input_fd:
      last_line = line
    strs = last_line.split()
    if len(strs) < 2 or strs[1] == 'P':
      input_fd.close()
      time.sleep(1)
      continue
    else:
      input_fd.close()
      return float(strs[0])

def my_func2(x):
  print x
  return x * x

best = fmin(fn=my_func2,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=1000)
print best

# from hyperopt import fmin, tpe, hp
# best = fmin(fn=lambda x: x ** 2,
    # space=hp.uniform('x', -10, 10),
    # algo=tpe.suggest,
    # max_evals=10000)
# print best