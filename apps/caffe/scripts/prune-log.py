import sys
import os
from datetime import datetime, timedelta

input_file = sys.argv[1]
pdsh = 1
if len(sys.argv) > 2:
  pdsh = int(sys.argv[2])

input_fd = open(input_file, 'r')
for line in input_fd:
  strs = line.split()
  if len(strs) == 20 + pdsh and strs[0 + pdsh] == 'BRANCH':
    continue
  print line,
