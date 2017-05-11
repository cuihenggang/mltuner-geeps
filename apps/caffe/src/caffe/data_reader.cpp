#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>
#include "caffe/util/rng.hpp"   /* for shuffling */

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
static boost::mutex bodies_mutex_;

DataReader::DataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
          param.data_param().prefetch() * param.data_param().batch_size())) {
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  body_ = weak.lock();
  if (!body_) {
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_);
}

DataReader::~DataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

//

DataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

DataReader::QueuePair::~QueuePair() {
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//

DataReader::Body::Body(
    const LayerParameter& param)
    : param_(param),
      new_queue_pairs_(),
      prefetch_schedule_queue_(new PrefetchScheduleQueue()) {
  StartInternalThread();
}

DataReader::Body::~Body() {
  StopInternalThread();
}

void DataReader::Body::InternalThreadEntry() {
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
  db->Open(param_.data_param().source(), db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  vector<shared_ptr<QueuePair> > qps;
  string keylist = param_.data_param().keylist();
  LOG(INFO) << "keylist = " << keylist;
  if (keylist != "") {
    std::ifstream fin(keylist.c_str());
    CHECK(fin);
    string key;
    while (!fin.eof()) {
      fin >> key;
      keylist_.push_back(new string(key));
    }
    LOG(INFO) << "num_keys = " << keylist_.size();
    /* Sometimes we might want to shuffle the data before starting,
     * in order to get reproduceable results */
    int num_preshuffles = param_.data_param().num_preshuffles();
    for (int i = 0; i < num_preshuffles; i++) {
      shuffle(keylist_.begin(), keylist_.end());
    }
    /* Prefetch the first batch */
    PrefetchSchedule prefetch_schedule = prefetch_schedule_queue_->pop();
    num_items_read_for_batch_ = 0;
    current_prefetch_batch_size_ = prefetch_schedule.batch_size;
    global_idx_ = prefetch_schedule.start_idx;
    CHECK_EQ(global_idx_, 0);
  }
  CHECK(keylist_.size());
  try {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    CHECK_EQ(solver_count, 1);
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(cursor.get(), qp.get());
      qps.push_back(qp);
    }
    // Main loop
    while (!must_stop()) {
      // LOG(INFO) << "DataReader::Body::InternalThreadEntry()";
      for (int i = 0; i < solver_count; ++i) {
        seek_item(cursor.get());
        read_one(cursor.get(), qps[i].get());
        next_item(cursor.get());
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

void DataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
  Datum* datum = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
  datum->ParseFromString(cursor->value());
  qp->full_.push(datum);
}

void DataReader::Body::seek_item(db::Cursor* cursor) {
  // if (permutation_.size()) {
    // if (key_batch_internal_count_ == 0) {
      // /* A new key batch */
      // const string& key_to_seek = permutation_[key_batch_idx_]->key;
      // LOG(INFO) << "Seek to key batch #" << key_batch_idx_ << " with key " << key_to_seek;
      // cursor->SeekToKey(key_to_seek);
      // CHECK(cursor->valid());
    // }
  // }
}
void DataReader::Body::next_item(db::Cursor* cursor) {
  num_items_read_for_batch_++;
  global_idx_++;
  // LOG(INFO) << "current_prefetch_batch_size_ = " << current_prefetch_batch_size_;
  if (num_items_read_for_batch_ >= current_prefetch_batch_size_) {
    /* Prefetch the next batch */
    PrefetchSchedule prefetch_schedule = prefetch_schedule_queue_->pop();
    num_items_read_for_batch_ = 0;
    current_prefetch_batch_size_ = prefetch_schedule.batch_size;
    global_idx_ = prefetch_schedule.start_idx;
  }
  // LOG(INFO) << "Prefetch the next batch, current_prefetch_batch_size_ = " << current_prefetch_batch_size_;
  if (global_idx_ >= keys_to_seek_.size()) {
    shuffle(keylist_.begin(), keylist_.end());
    for (size_t i = 0; i < keylist_.size(); i++) {
      keys_to_seek_.push_back(keylist_[i]);
    }
  }
  const string& key_to_seek = *keys_to_seek_[global_idx_];
  // LOG(INFO) << "Seek to key idx " << idx_to_seek << " with key " << key_to_seek;
  cursor->SeekToKey(key_to_seek);
  CHECK(cursor->valid());
}
}  // namespace caffe
