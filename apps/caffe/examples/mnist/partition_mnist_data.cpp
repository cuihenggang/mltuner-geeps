// This script partitions the MNIST dataset
// Usage:
//    convert_mnist_data [FLAGS] input_image_file input_label_file
//                        output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <boost/format.hpp>

#include <stdint.h>
#include <sys/stat.h>

#include <iostream>
#include <fstream>  // NOLINT(readability/streams)
#include <vector>
#include <string>

#include "caffe/proto/caffe.pb.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

DEFINE_int32(num_parts, 1, "Number of parts");

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void partition_dataset(const char* image_filename, const char* label_filename,
        const char* output_image_prefix, const char* output_label_prefix,
        int num_parts) {
  // Open input files
  ifstream image_file(image_filename, ios::in | ios::binary);
  ifstream label_file(label_filename, ios::in | ios::binary);
  CHECK(image_file) << "Unable to open file " << image_filename;
  CHECK(label_file) << "Unable to open file " << label_filename;

  // Open output files
  vector<ofstream *> output_image_files(num_parts);
  vector<ofstream *> output_label_files(num_parts);
  for (int i = 0; i < num_parts; i++) {
    string output_image_filename =
        (boost::format("%s.%i") % output_image_prefix % i).str();
    output_image_files[i] = new ofstream(
        output_image_filename.c_str(), ios::out | ios::binary);
    string output_label_filename =
        (boost::format("%s.%i") % output_label_prefix % i).str();
    output_label_files[i] = new ofstream(
        output_label_filename.c_str(), ios::out | ios::binary);
    CHECK(*output_image_files[i])
        << "Unable to open file " << output_image_filename;
    CHECK(*output_label_files[i])
        << "Unable to open file " << output_label_filename;
  }

  // Read the magic and the meta data
  uint32_t image_magic_image_endian;
  uint32_t label_magic_image_endian;
  uint32_t num_items_image_endian;
  uint32_t num_labels_image_endian;
  uint32_t rows_image_endian;
  uint32_t cols_image_endian;
  uint32_t image_magic;
  uint32_t label_magic;
  uint32_t num_items;
  uint32_t num_labels;
  uint32_t rows;
  uint32_t cols;

  image_file.read(reinterpret_cast<char*>(&image_magic_image_endian), 4);
  image_magic = swap_endian(image_magic_image_endian);
  CHECK_EQ(image_magic, 2051) << "Incorrect image file magic.";
  label_file.read(reinterpret_cast<char*>(&label_magic_image_endian), 4);
  label_magic = swap_endian(label_magic_image_endian);
  CHECK_EQ(label_magic, 2049) << "Incorrect label file magic.";
  image_file.read(reinterpret_cast<char*>(&num_items_image_endian), 4);
  num_items = swap_endian(num_items_image_endian);
  label_file.read(reinterpret_cast<char*>(&num_labels_image_endian), 4);
  num_labels = swap_endian(num_labels_image_endian);
  CHECK_EQ(num_items, num_labels);
  image_file.read(reinterpret_cast<char*>(&rows_image_endian), 4);
  rows = swap_endian(rows_image_endian);
  image_file.read(reinterpret_cast<char*>(&cols_image_endian), 4);
  cols = swap_endian(cols_image_endian);

  // Decide number of items for each part
  vector<uint32_t> num_part_items(num_parts);
  vector<uint32_t> part_item_starts(num_parts);
  uint32_t div = num_items / num_parts;
  uint32_t res = num_items % num_parts;
  cout << "num_items = " << num_items << endl;
  for (int i = 0; i < num_parts; i++) {
    num_part_items[i] = div + (res > i ? 1 : 0);
    part_item_starts[i] = div * i + (res > i ? i : res);
    cout << "num_part_items = " << num_part_items[i] << endl;
  }

  // Write metadata
  for (int i = 0; i < num_parts; i++) {
    output_image_files[i]->write(reinterpret_cast<char*>(&image_magic_image_endian), 4);
    output_label_files[i]->write(reinterpret_cast<char*>(&label_magic_image_endian), 4);
    num_items_image_endian = swap_endian(num_part_items[i]);
    output_image_files[i]->write(reinterpret_cast<char*>(&num_items_image_endian), 4);
    output_label_files[i]->write(reinterpret_cast<char*>(&num_items_image_endian), 4);
    output_image_files[i]->write(reinterpret_cast<char*>(&rows_image_endian), 4);
    output_image_files[i]->write(reinterpret_cast<char*>(&cols_image_endian), 4);
  }

  char label;
  char* pixels = new char[rows * cols];

  int i = 0;
  cout << "partition data" << endl;
  for (int item_id = 0; item_id < num_items; ++item_id) {
    image_file.read(pixels, rows * cols);
    label_file.read(&label, 1);
    while (part_item_starts[i] + num_part_items[i] < item_id) {
      i++;
      assert(i < num_parts);
    }
    output_image_files[i]->write(pixels, rows * cols);
    output_label_files[i]->write(&label, 1);
  }

  for (int i = 0; i < num_parts; i++) {
    delete output_image_files[i];
    delete output_label_files[i];
  }
  delete pixels;
}

int main(int argc, char** argv) {
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("This script partitions the MNIST dataset"
        "Usage:\n"
        "    partition_mnist_data [FLAGS] input_image_file input_label_file "
        "    output_image_files_prefix output_label_files_prefix\n"
        "The MNIST dataset could be downloaded at\n"
        "    http://yann.lecun.com/exdb/mnist/\n"
        "You should gunzip them after downloading,"
        "or directly use data/mnist/get_mnist.sh\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int num_parts = FLAGS_num_parts;

  if (argc != 5) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
        "examples/mnist/convert_mnist_data");
  } else {
    google::InitGoogleLogging(argv[0]);
    partition_dataset(argv[1], argv[2], argv[3], argv[4], num_parts);
  }
  return 0;
}
