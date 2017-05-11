//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "boost/format.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
using std::ofstream;
using std::endl;
namespace db = caffe::db;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARTrainBatches = 5;
const int batchSize = 1;

void make_permutation(const string& input_folder, const string& output_folder,
    const string& db_type, int num_parts, int part_id) {
  string train_permutation_name = (boost::format("%s/cifar10_train.keylist.%d") % output_folder % part_id).str();
  ofstream train_fout(train_permutation_name.c_str());
  int count = 0;

  LOG(INFO) << "Writing Training data";
  for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
      if (count++ % num_parts != part_id) {
        continue;
      }
      CHECK_EQ((fileid * kCIFARBatchSize + itemid) % num_parts, part_id);
      train_fout << caffe::format_int(fileid * kCIFARBatchSize + itemid, 5) << endl;
    }
  }

  LOG(INFO) << "Writing Testing data";
  count = 0;
  scoped_ptr<db::DB> test_db(db::GetDB(db_type));
  string test_permutation_name = (boost::format("%s/cifar10_test.keylist.%d") % output_folder % part_id).str();
  ofstream test_fout(test_permutation_name.c_str());
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
    if (count++ % num_parts != part_id) {
      continue;
    }
    test_fout << caffe::format_int(itemid, 5) << endl;
  }
}

int main(int argc, char** argv) {
  if (argc != 6) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder db_type\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    int num_parts = atoi(argv[4]);
    int part_id = atoi(argv[5]);
    make_permutation(string(argv[1]), string(argv[2]), string(argv[3]), num_parts, part_id);
  }
  return 0;
}
