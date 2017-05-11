#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include "boost/scoped_ptr.hpp"
#include "boost/format.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
using namespace caffe;  // NOLINT(build/namespaces)
using std::max;
using std::pair;
using boost::scoped_ptr;
int main(int argc, char** argv) {
  // ::google::InitGoogleLogging(argv[0]);
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
        " a leveldb/lmdb\n"
        "Usage:\n"
        "    compute_image_mean INPUT_PREFIX NUM_PARTS [OUTPUT_FILE]\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc < 3 || argc > 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
    return 1;
  }
  BlobProto sum_blob;
  int channels = 0;
  int dim = 0;
  int data_size;
  int num_parts = atoi(argv[2]);
  for (int i = 0; i < num_parts; i++) {
    BlobProto blob_proto;
    string mean_file = (boost::format("%s.%d") % argv[1] % i).str();
    // "/tank/projects/biglearning/hengganc/LazyTable/applications/caffe/data/ilsvrc12/8parts/imagenet_mean.binaryproto.0";
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    if (i == 0) {
      sum_blob.set_num(1);
      sum_blob.set_channels(blob_proto.channels());
      sum_blob.set_height(blob_proto.height());
      sum_blob.set_width(blob_proto.width());
      channels = sum_blob.channels();
      dim = sum_blob.height() * sum_blob.width();
      data_size = channels * dim;
      for (int j = 0; j < data_size; j++) {
        sum_blob.add_data(0.);
      }
    }
    CHECK_EQ(blob_proto.data_size(), sum_blob.data_size());
    LOG(INFO) << "blob_proto.data() = " << LOG(INFO) << blob_proto.data(115);
    for (int i = 0; i < sum_blob.data_size(); ++i) {
      sum_blob.set_data(i, sum_blob.data(i) + blob_proto.data(i));
    }
  }
  // Take the average
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / num_parts);
  }
  // Write to disk
  if (argc == 4) {
    LOG(INFO) << "Write to " << argv[3];
    WriteProtoToBinaryFile(sum_blob, argv[3]);
  }
  LOG(INFO) << "Number of channels: " << channels;
  std::vector<float> mean_values(channels, 0.0);
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < dim; ++i) {
      mean_values[c] += sum_blob.data(dim * c + i);
    }
    LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
  }
  return 0;
}

