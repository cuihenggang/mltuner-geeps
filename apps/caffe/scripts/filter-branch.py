import sys
import os
from datetime import datetime, timedelta

input_file = sys.argv[1]
selected_branch = int(sys.argv[2])
output_file = '%s.branch%i' % (input_file, selected_branch)

input_fd = open(input_file, 'r')
output_fd = open(output_file, 'w')
for line in input_fd:
  strs = line.split(',')
  assert len(strs) == 4
  branch = int(strs[1])
  if branch == selected_branch:
    output_fd.write(line)
