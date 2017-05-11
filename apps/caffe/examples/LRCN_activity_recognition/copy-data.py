import sys
import os

file_names_file_name = sys.argv[1]
src_dir = sys.argv[2]
dst_dir = sys.argv[3]

os.system('sudo mkdir %s' % dst_dir)
file_names_file = open(file_names_file_name, 'r')
count = 0
for line in file_names_file:
  file_name = line.split()[0].split('/')[1]
  src_file_name = os.path.join(src_dir, file_name)
  dst_file_name = os.path.join(dst_dir, file_name)
  os.system('sudo cp -r %s %s' % (src_file_name, dst_file_name))
  count = count + 1
  if count % 10 == 0:
    print count
print 'done with %s' % file_names_file_name
