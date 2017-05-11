import sys
import os
from datetime import datetime, timedelta

input_file = sys.argv[1]
pdsh = 1
if len(sys.argv) > 2:
  pdsh = int(sys.argv[2])

input_fd = open(input_file, 'r')
this_search_start_time = 0
total_time = 0
total_search_time = 0
total_adjust_time = 0
total_train_time = 0
no_adjust_total_time = 0
no_adjust_accuracy = 0
peak_accuracy = 0
train_time_in_search = 0
for line in input_fd:
  strs = line.split()
  if len(strs) == 20 + pdsh and strs[0 + pdsh] == 'BRANCH':
    branch = int(strs[3 + pdsh])
    clock = int(strs[7 + pdsh])
    loss = float(strs[16 + pdsh])
    time = float(strs[19 + pdsh])
    total_time = time
  if len(strs) == 4 + pdsh and strs[0 + pdsh] == 'Validation':
    accuracy = float(strs[3 + pdsh])
    if accuracy > peak_accuracy:
      peak_accuracy = accuracy
  if len(strs) == 8 + pdsh and strs[0 + pdsh] == 'Run':
    this_search_time = time - this_search_start_time - train_time_in_search
    # print 'search finish with %f' % this_search_time
    if total_search_time == 0:
      total_search_time = this_search_time
      print 'search_time:'
      print this_search_time
      print this_search_time / 3600
      print
    else:
      print 'adjust_time:'
      print this_search_time
      print this_search_time / 3600
      print
      total_adjust_time = total_adjust_time + this_search_time
    search_finish_time = time
  if len(strs) == 4 + pdsh and strs[1 + pdsh] == 'refining':
    this_search_start_time = time
    train_time = time - search_finish_time + train_time_in_search
    print 'train_time:'
    print train_time
    print train_time / 3600
    print
    total_train_time = total_train_time + train_time
    # print 'search start at %f' % this_search_start_time
    if no_adjust_total_time == 0:
      no_adjust_total_time = time
      no_adjust_accuracy = peak_accuracy
  if len(strs) == 4 + pdsh and strs[1 + pdsh] == 'span':
    train_time_in_search = float(strs[3 + pdsh])

train_time = total_time - search_finish_time + train_time_in_search
print 'train_time:'
print train_time
print train_time / 3600
print
total_train_time = total_train_time + train_time

print 'total_search_time:'
print total_search_time
print total_search_time / 3600
print
print 'total_adjust_time:'
print total_adjust_time
print total_adjust_time / 3600
print
print 'total_train_time:'
print total_train_time
print total_train_time / 3600
print
print 'total_time:'
print total_time
print total_time / 3600
print
print 'overhead:'
print (total_search_time + total_adjust_time) / total_train_time
print
print 'peak_accuracy:'
print peak_accuracy
print
print 'no_adjust_total_time:'
print no_adjust_total_time
print
print 'no_adjust_accuracy:'
print no_adjust_accuracy
print

