import sys
import os
from datetime import datetime, timedelta

input_file = sys.argv[1]
rate = int(sys.argv[2])
output_file = '%s.ds%i' % (input_file, rate)

input_fd = open(input_file, 'r')
output_fd = open(output_file, 'w')
time = 0.0
branch = 0
clock = 0
loss = 0.0
count = 0
for line in input_fd:
  strs = line.split(',')
  assert len(strs) == 4
  if branch != int(strs[1]):
    time = 0.0
    branch = 0
    clock = 0
    loss = 0.0
    count = 0
  time = time + float(strs[0])
  branch = int(strs[1])
  clock = clock + int(strs[2])
  loss = loss + float(strs[3])
  count = count + 1
  if count == rate:
    output_str = '%f,%i,%i,%f\n' % (time / rate, branch, clock / rate, loss / rate)
    output_fd.write(output_str)
    time = 0.0
    branch = 0
    clock = 0
    loss = 0.0
    count = 0
