import sys
import os

input_file_name = sys.argv[1]
num_parts = int(sys.argv[2])

output_dir_root = '%iparts' %  (num_parts)
# os.system('mkdir %s' % output_dir_root)
output_files = []
for i in range(num_parts):
  output_file_name = '%s.%i' % (input_file_name, i)
  output_files.append(open(os.path.join(output_dir_root, output_file_name), 'w'))

input_file = open(input_file_name, 'r')
count  = 0
for line in input_file:
  part_id = count % num_parts
  output_files[part_id].write(line)
  count = count + 1
  if count % 1000 == 0:
    print count

for i in range(num_parts):
  output_files[i].close()
