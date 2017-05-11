#ifdef USE_CUDNN

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BatchNormLayer<Dtype>::LayerSetUp(bottom, top);

  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createTensor4dDesc<Dtype>(&scale_bias_mean_var_desc_);

  // currently only SPATIAL mode is supported (most commonly used mode)
  // If there's enough demand we can implement CUDNN_BATCHNORM_PER_ACTIVATION
  // though it's not currently implemented for the CPU layer
  mode_ = CUDNN_BATCHNORM_SPATIAL;
  int channels = bottom[0]->channels();

  CHECK_EQ(this->blobs_.size(), 0);
  this->blobs_.resize(2);
  this->blobs_[0].reset(new Blob<Dtype>(1, channels, 1, 1));  // scale
  this->blobs_[1].reset(new Blob<Dtype>(1, channels, 1, 1));  // bias

  shared_ptr<Filler<Dtype> > scale_filler(
    GetFiller<Dtype>(this->layer_param_.batch_norm_param().scale_filler()));
  scale_filler->Fill(this->blobs_[0].get());

  shared_ptr<Filler<Dtype> > bias_filler(
    GetFiller<Dtype>(this->layer_param_.batch_norm_param().bias_filler()));
  bias_filler->Fill(this->blobs_[1].get());

  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BatchNormLayer<Dtype>::Reshape(bottom, top);

  // set up main tensors
  cudnn::setTensor4dDesc<Dtype>(
    &bottom_desc_, bottom[0]->num(),
    bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  cudnn::setTensor4dDesc<Dtype>(
    &top_desc_, bottom[0]->num(),
    bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());

  // aux tensors for caching mean & invVar from fwd to bwd pass
  int C = bottom[0]->channels();
  int H = bottom[0]->height();
  int W = bottom[0]->width();
  if (mode_ == CUDNN_BATCHNORM_SPATIAL) {
    // save_mean_.Reshape(1, C, 1, 1);
    // save_inv_var_.Reshape(1, C, 1, 1);
    top[1]->Reshape(1, C, 1, 1);  // save_mean
    top[2]->Reshape(1, C, 1, 1);  // save_inv_var
    top[3]->Reshape(1, C, 1, 1);  // running_mean
    top[4]->Reshape(1, C, 1, 1);  // running_var
  } else if (mode_ == CUDNN_BATCHNORM_PER_ACTIVATION) {
    // save_mean_.Reshape(1, C, H, W);
    // save_inv_var_.Reshape(1, C, H, W);
    top[1]->Reshape(1, C, H, W);  // save_mean
    top[2]->Reshape(1, C, H, W);  // save_inv_var
    top[3]->Reshape(1, C, H, W);  // running_mean
    top[4]->Reshape(1, C, H, W);  // running_var
  } else {
    LOG(FATAL) << "Unknown cudnnBatchNormMode_t";
  }
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc_,
                                            bottom_desc_, mode_));
}

template <typename Dtype>
CuDNNBatchNormLayer<Dtype>::~CuDNNBatchNormLayer() {
  if (!handles_setup_) return;

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc_);
}

INSTANTIATE_CLASS(CuDNNBatchNormLayer);
}  // namespace caffe
#endif
