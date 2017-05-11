import sys
import os
from datetime import datetime, timedelta

input_file = sys.argv[1]
pdsh = 1
if len(sys.argv) > 2:
  pdsh = int(sys.argv[2])

input_fd = open(input_file, 'r')
times = []
branches = []
clocks = []
losses = []
for line in input_fd:
  strs = line.split()
  if len(strs) == 20 + pdsh and strs[0 + pdsh] == 'BRANCH':
    branch = int(strs[3 + pdsh])
    clock = int(strs[7 + pdsh])
    loss = float(strs[16 + pdsh])
    time = float(strs[19 + pdsh])
    branches.append(branch)
    clocks.append(clock)
    losses.append(loss)
    times.append(time)

for i in range(len(clocks)):
  print '%f,%i,%i,%f' % (times[i], branches[i], clocks[i], losses[i])

