import sys
import os
from datetime import datetime, timedelta

input_file = sys.argv[1]
pdsh = 1
if len(sys.argv) > 2:
  pdsh = int(sys.argv[2])

input_fd = open(input_file, 'r')
time = 0.0
times = [time]
accuracy = 0.0
accuracies = [accuracy]
for line in input_fd:
  strs = line.split()
  if len(strs) == 20 + pdsh and strs[0 + pdsh] == 'BRANCH':
    time = float(strs[19 + pdsh])
  if len(strs) == 4 + pdsh and strs[0 + pdsh] == 'Validation':
    times.append(time)
    accuracy = float(strs[3 + pdsh])
    accuracies.append(accuracy)

for i in range(len(times)):
  print '%f,%f' % (times[i], accuracies[i])

