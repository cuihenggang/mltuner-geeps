import sys
import os
from datetime import datetime, timedelta

input_file = sys.argv[1]

input_fd = open(input_file, 'r')
accuracy = 0
time = 0
for line in input_fd:
  strs = line.split()
  if len(strs) == 7 and strs[1] == 'Stopping':
    accuracy = float(strs[6])
  if len(strs) == 5 and strs[1] == 'Training':
    time = float(strs[4])
    print '%f,%f' % (accuracy, time)
