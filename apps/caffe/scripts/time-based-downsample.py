import sys
import os
from datetime import datetime, timedelta

input_file = sys.argv[1]
interval = float(sys.argv[2])
output_file = '%s.tds%f' % (input_file, interval)

input_fd = open(input_file, 'r')
output_fd = open(output_file, 'w')
starting_time = 0.0
time = 0.0
branch = 0
clock = 0
loss = 0.0
count = 0
for line in input_fd:
  strs = line.split(',')
  assert len(strs) == 4
  current_time = float(strs[0])
  if branch != int(strs[1]):
    starting_time = current_time
    time = 0.0
    branch = 0
    clock = 0
    loss = 0.0
    count = 0
  time = time + current_time
  branch = int(strs[1])
  clock = clock + int(strs[2])
  loss = loss + float(strs[3])
  count = count + 1
  if current_time - starting_time > interval:
    output_str = '%f,%i,%i,%f\n' % (time / count, branch, clock / count, loss / count)
    output_fd.write(output_str)
    starting_time = current_time
    time = 0.0
    branch = 0
    clock = 0
    loss = 0.0
    count = 0
