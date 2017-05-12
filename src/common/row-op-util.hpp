#ifndef __row_op_util_hpp__
#define __row_op_util_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

// Overloaded functions to handle operations on RowData or RowOpVal
// each of which can be of the vector type or unordered map

#include <vector>

#include "geeps-user-defined-types.hpp"
#include "portable-bytes.hpp"
#include "common/gpu-util/math_functions.hpp"

struct DoubleIndex {
  size_t id0;
  size_t id1;
  DoubleIndex(size_t id0_i = 0, size_t id1_i = 0) : id0(id0_i), id1(id1_i) {}
};

// ops for single value
void set_zero(val_t& data);
void resize_maybe(val_t& data, const col_idx_t size);
void pack_data(PortableBytes& bytes, const val_t& data);
bool is_able_to_pack_const_data(const val_t& data);
void pack_const_data(PortableBytes& bytes, const val_t& data);
bool is_empty(const val_t& data);

// ops for array
void set_zero(ArrayData& data);
void resize_maybe(ArrayData& data, const col_idx_t size);
void pack_data(PortableBytes& bytes, const ArrayData& data);
bool is_able_to_pack_const_data(const ArrayData& data);
void pack_const_data(PortableBytes& bytes, const ArrayData& data);
bool is_empty(const ArrayData& data);
void operator += (ArrayData& left, const ArrayData& right);
void add_row_batch(
    ArrayData *rows_y, const ArrayData *rows_x, size_t batch_size);
void add_row_batch(
    cublasHandle_t cublas_handle,
    ArrayData *rows_y, const ArrayData *rows_x, size_t batch_size);
void assign_rows_to_double_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    size_t num_vals_limit,
    cudaStream_t cuda_stream);
void assign_rows_from_double_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    size_t num_vals_limit,
    cudaStream_t cuda_stream);
void add_rows_from_double_index_gpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    size_t num_vals_limit,
    cudaStream_t cuda_stream);

//////////// helper functions for single value ////////////

inline void set_zero(val_t& data) {
  data = 0;
}

inline void resize_maybe(val_t& data, const col_idx_t size) {
  // do nothing
}

inline void pack_data(PortableBytes& bytes, const val_t& data) {
  bytes.pack<val_t>(data);
}

inline void unpack_data(val_t& data, PortableBytes& bytes) {
  bytes.unpack<val_t>(data);
}

inline bool is_able_to_pack_const_data(const val_t& data) {
  return false;
}

inline void pack_const_data(PortableBytes& bytes, val_t& data) {
  // not allowed
  assert(0);
}

inline bool is_empty(const val_t& data) {
  return false;
}

//////////// helper functions for array data ////////////

inline void set_zero(ArrayData& data) {
  data.init();
}

inline void resize_maybe(ArrayData& data, const col_idx_t size) {
  // do nothing
}

inline void pack_data(PortableBytes& bytes, const ArrayData& data) {
  bytes.pack<ArrayData>(data);
}

inline void unpack_data(ArrayData& data, PortableBytes& bytes) {
  bytes.unpack<ArrayData>(data);
}

inline bool is_able_to_pack_const_data(const ArrayData& data) {
  return false;
}

inline void pack_const_data(PortableBytes& bytes, ArrayData& data) {
  // not allowed
  assert(0);
}

inline bool is_empty(const ArrayData& data) {
  return false;
}

inline void operator += (ArrayData& left, const ArrayData& right) {
  for (uint i = 0; i < ROW_DATA_SIZE; i++) {
    left.data[i] += right.data[i];
  }
}

inline void add_row_batch(
    ArrayData *rows_y, const ArrayData *rows_x, size_t batch_size) {
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  size_t n = batch_size * ROW_DATA_SIZE;
  cpu_axpy<val_t>(n, 1, x, y);
}

inline void add_row_batch_gpu(
    cublasHandle_t cublas_handle,
    ArrayData *rows_y, const ArrayData *rows_x, size_t batch_size) {
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  size_t n = batch_size * ROW_DATA_SIZE;
  gpu_axpy<val_t>(cublas_handle, n, 1, x, y);
}

inline void assign_rows_to_double_index_cpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    size_t num_vals_limit) {
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  for (size_t row_index_id = 0; row_index_id < num_rows; row_index_id++) {
    /* Assign rows from "id1" to "id0" */
    size_t row_from = index[row_index_id].id1 + index_offset.id1;
    size_t row_to = index[row_index_id].id0 + index_offset.id0;
    for (size_t val_id = 0; val_id < row_size; val_id++) {
      size_t x_idx = row_from * row_size + val_id;
      size_t y_idx = row_to * row_size + val_id;
      if (y_idx < num_vals_limit) {
        y[y_idx] = x[x_idx];
      }
    }
  }
}

inline void assign_rows_from_double_index_cpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    size_t num_vals_limit) {
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  for (size_t row_index_id = 0; row_index_id < num_rows; row_index_id++) {
    /* Add rows from "id0" to "id1" */
    size_t row_from = index[row_index_id].id0 + index_offset.id0;
    size_t row_to = index[row_index_id].id1 + index_offset.id1;
    for (size_t val_id = 0; val_id < row_size; val_id++) {
      size_t x_idx = row_from * row_size + val_id;
      size_t y_idx = row_to * row_size + val_id;
      if (x_idx < num_vals_limit) {
        y[y_idx] = x[x_idx];
      }
    }
  }      
}

inline void add_rows_from_double_index_cpu(
    ArrayData *rows_y, const ArrayData *rows_x, const DoubleIndex *index,
    size_t num_rows, DoubleIndex index_offset, size_t row_size,
    size_t num_vals_limit) {
  val_t *y = reinterpret_cast<val_t *>(rows_y);
  const val_t *x = reinterpret_cast<const val_t *>(rows_x);
  for (size_t row_index_id = 0; row_index_id < num_rows; row_index_id++) {
    /* Add rows from "id0" to "id1" */
    size_t row_from = index[row_index_id].id0 + index_offset.id0;
    size_t row_to = index[row_index_id].id1 + index_offset.id1;
    for (size_t val_id = 0; val_id < row_size; val_id++) {
      size_t x_idx = row_from * row_size + val_id;
      size_t y_idx = row_to * row_size + val_id;
      if (x_idx < num_vals_limit) {
        y[y_idx] += x[x_idx];
      }
    }
  }      
}

#endif  // defined __row_op_util_hpp__
