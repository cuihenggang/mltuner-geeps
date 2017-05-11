#ifndef CAFFE_DATA_READER_HPP_
#define CAFFE_DATA_READER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

struct PrefetchSchedule {
  size_t start_idx;
  size_t batch_size;
  PrefetchSchedule() {}
  PrefetchSchedule(size_t start_idx, size_t batch_size)
      : start_idx(start_idx), batch_size(batch_size) {}
  PrefetchSchedule(const PrefetchSchedule& other)
      : start_idx(other.start_idx), batch_size(other.batch_size) {}
};
typedef BlockingQueue<PrefetchSchedule> PrefetchScheduleQueue;

/**
 * @brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */
class DataReader {
 public:
  explicit DataReader(const LayerParameter& param);
  ~DataReader();

  inline BlockingQueue<Datum*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue<Datum*>& full() const {
    return queue_pair_->full_;
  }

  inline void SchedulePrefetch(const PrefetchSchedule& prefetch_schedule) {
    body_->prefetch_schedule_queue_->push(prefetch_schedule);
  }

 protected:
  // Queue pairs are shared between a body and its readers
  class QueuePair {
   public:
    explicit QueuePair(int size);
    ~QueuePair();

    BlockingQueue<Datum*> free_;
    BlockingQueue<Datum*> full_;

  DISABLE_COPY_AND_ASSIGN(QueuePair);
  };

  // A single body is created per source
  class Body : public InternalThread {
   public:
    explicit Body(const LayerParameter& param);
    virtual ~Body();

   protected:
    void InternalThreadEntry();
    void seek_item(db::Cursor* cursor);
    void read_one(db::Cursor* cursor, QueuePair* qp);
    void next_item(db::Cursor* cursor);

    const LayerParameter param_;
    BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;
    shared_ptr<PrefetchScheduleQueue> prefetch_schedule_queue_;

    size_t global_idx_;
    std::vector<string *> keylist_;
    int current_prefetch_batch_size_;
    int num_items_read_for_batch_;
    std::vector<string *> keys_to_seek_;

    friend class DataReader;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().source();
  }

  const shared_ptr<QueuePair> queue_pair_;
  shared_ptr<Body> body_;

  static map<const string, boost::weak_ptr<DataReader::Body> > bodies_;

DISABLE_COPY_AND_ASSIGN(DataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
