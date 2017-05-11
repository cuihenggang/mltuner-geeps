#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  /* This batch size is the maximum batch size,
   * and the real batch size can be changed on the fly. */
  int max_batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  reader_.SchedulePrefetch(PrefetchSchedule(0, 1));
  Datum& datum = *(reader_.full().pop());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = max_batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, max_batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(int batch_size, Batch<Dtype>* batch) {
  // LOG(INFO) << "DataLayer<Dtype>::load_batch()";
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  vector<int> label_shape(1, batch_size);
  batch->label_.Reshape(label_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void DataLayer<Dtype>::ChangeBatchSize(
    int batch_size, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(batch_size, this->layer_param_.data_param().batch_size());
  int top_size = this->output_labels_ ? 2 : 1;
  for (int i = 0; i < top_size; i++) {
    vector<int> blob_shape = top[i]->shape();
    blob_shape[0] = batch_size;
    top[i]->Reshape(blob_shape);
  }
}

template<typename Dtype>
void DataLayer<Dtype>::SchedulePrefetch(
    const PrefetchSchedule& prefetch_schedule) {
  this->prefetch_schedule_queue_->push(prefetch_schedule);
  reader_.SchedulePrefetch(prefetch_schedule);
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
