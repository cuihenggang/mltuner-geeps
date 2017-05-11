#ifndef __extended_utils_hpp__
#define __extended_utils_hpp__

#include "common/common-utils.hpp"
#include "common/row-op-util.hpp"
#include "common/wire-protocol.hpp"
#include "common/gpu-util/math_functions.hpp"

#if defined(SUSTINA)
#include "affinity.hpp"
#endif

inline void mallocHost(void **ptr, size_t size) {
#if defined(SUSTINA)
  /* We find, on our Susitna cluster, the CPU/GPU data transfer bandwidth
   * is much higher, when the CPU memory is allocated in NUMA zone 0. */
  int numa_node_id = 0;
  void *opaque = set_mem_affinity(numa_node_id);
#endif
  /* On CUDA 7.5 and CUDA 7.0, the cudaMallocHost() will sometimes fail
  * even though there is still available memory to allocate.
  * I don't know why it's happening, but as a workaround,
  * I added this while loop to retry cudaMallocHost(). */
  while (cudaMallocHost(ptr, size) != cudaSuccess) {
    cout << "*** WARNING: cudaMallocHost failed, will retry" << endl;
  }
#if defined(SUSTINA)
  restore_mem_affinity(opaque);
#endif
}

inline void mallocHost(RowData **ptr, size_t size) {
  mallocHost(reinterpret_cast<void **>(ptr), size);
}

/* Features:
 * - Single threaded
 * - The entries are freed in the same order as they are allocated. */
struct SimpleCacheHelper {
  size_t size_;
  size_t free_start_;
  size_t free_end_;
  size_t free_count_;
  size_t unused_tail_;
  SimpleCacheHelper() {
    init(0);
  }
  ~SimpleCacheHelper() {
    clear();
  }
  void init(size_t size) {
    size_ = size;
    free_count_ = size_;
    unused_tail_ = size_;
    if (size_) {
      free_start_ = 0;
      free_end_ = size_ - 1;
    }
  }
  void clear() {
    init(0);
  }
  size_t size() {
    return size_;
  }
  size_t get(size_t count, bool wait) {
    CHECK_GE(free_count_, count);
    CHECK_LT(free_start_, size_);
    CHECK_LT(free_end_, size_);
    CHECK_NE(free_start_, free_end_);
    if (free_start_ < free_end_) {
      /* All free space is after free_start_ */
      if (free_start_ + count < free_end_) {
        /* There are enough contiguous free space after free_start_ */
        size_t index = free_start_;
        free_start_ += count;
        free_count_ -= count;
        return index;
      } else {
        cerr << "Insufficient space\n";
        assert(0);
      }
    } else {
      /* There are some free space at the beginning */
      if (free_start_ + count < size_) {
        /* There are enough contiguous free space after free_start_ */
        size_t index = free_start_;
        free_start_ += count;
        free_count_ -= count;
        return index;
      } else {
        /* There are NOT enough contiguous free space after free_start_.
         * We mark the space after free_start_ as unused_tail_,
         * and we go back to the front. */
        unused_tail_ = free_start_;
        free_start_ = 0;
        free_count_ -= (size_ - unused_tail_);
        CHECK_LT(free_start_, free_end_);
        if (free_start_ + count < free_end_) {
          size_t index = free_start_;
          free_start_ += count;
          free_count_ -= count;
          return index;
        } else {
          cerr << "Insufficient space\n";
          assert(0);
        }
      }
    }
  }
  void put(size_t index, size_t count) {
    CHECK_LT(index, size_);
    if (index == (free_end_ + 1) % size_) {
      free_end_ = (free_end_ + count) % size_;
    } else {
      /* There should be an unused tail */
      CHECK_EQ(index, 0);
      CHECK_EQ(unused_tail_, (free_end_ + 1) % size_);
      free_count_ += (size_ - unused_tail_);
      unused_tail_ = size_;
      CHECK_LE(count, size_);
      free_end_ = count - 1;
    }
    free_count_ += count;
    // if (free_count_ == size_) {
      // free_start_ = 0;
      // free_end_ = size_ - 1;
    // }
  }
};

struct MultithreadedCacheHelper {
  struct AllocMapEntry {
    size_t size;
    int tag;
    AllocMapEntry(size_t size = 0, int tag = 0) : size(size), tag(tag) {}
  };
  typedef std::map<size_t, AllocMapEntry> AllocMap;
  typedef AllocMap::iterator AllocMapIter;
  AllocMap alloc_map_;
  size_t size_;
  size_t allocated_;
  size_t last_alloc_start_;
  boost::mutex mutex_;
  boost::condition_variable cvar_;
  MultithreadedCacheHelper() {
    init(0);
  }
  ~MultithreadedCacheHelper() {
    clear();
  }
  void init(size_t size) {
    size_ = size;
    allocated_ = 0;
    alloc_map_.clear();
    last_alloc_start_ = size_;
      /* Initialize last_alloc_start_ to size_, so that this statement is true:
       *   alloc_map_.find(last_alloc_start_) == alloc_map_.end() */
  }
  void clear() {
    init(0);
  }
  size_t size() {
    return size_;
  }
  size_t get(size_t count, bool wait, int tag = 0) {
    boost::unique_lock<boost::mutex> lock(mutex_);
    while (true) {
      size_t search_start = 0;
      AllocMapIter last_alloc_pos_ = alloc_map_.find(last_alloc_start_);
      if (last_alloc_pos_ != alloc_map_.end()) {
        size_t last_alloc_start = last_alloc_pos_->first;
        size_t last_alloc_count = last_alloc_pos_->second.size;
        /* Search after the last allocated position */
        search_start = last_alloc_start + last_alloc_count;
      }
      size_t start;
      if (search_start < size_) {
        start = search_start;
        for (AllocMapIter map_it = alloc_map_.begin();
             map_it != alloc_map_.end(); map_it++) {
          CHECK_LT(start, size_);
          size_t allocated_start = map_it->first;
          size_t allocated_count = map_it->second.size;
          if (allocated_start < search_start) {
            /* Only search after the last allocated position */
            continue;
          }
          CHECK_LE(start, allocated_start);
          if (start + count <= allocated_start) {
            /* Allocated it before this entry */
            alloc_map_[start] = AllocMapEntry(count, tag);
            last_alloc_start_ = start;
            allocated_ += count;
            // cout << "MultithreadedCacheHelper got " << start
                 // << " for " << count
                 // << " allocated = " << allocated_ << endl;
            return start;
          } else {
            start = allocated_start + allocated_count;
          }
        }
        /* Check the space after the last entry */
        if (start + count <= size_) {
          /* Allocated it at the end */
          alloc_map_[start] = AllocMapEntry(count, tag);
          last_alloc_start_ = start;
          allocated_ += count;
          // cout << "MultithreadedCacheHelper got " << start
               // << " for " << count
               // << " allocated = " << allocated_ << endl;
          return start;
        }
      }
      /* Search the space before the last allocated position */
      start = 0;
      for (AllocMapIter map_it = alloc_map_.begin();
           map_it != alloc_map_.end(); map_it++) {
        if (start >= search_start) {
          /* Only search before the last allocated position */
          break;
        }
        CHECK_LT(start, size_);
        size_t allocated_start = map_it->first;
        size_t allocated_count = map_it->second.size;
        CHECK_LE(start, allocated_start);
        if (start + count <= allocated_start) {
          /* Allocated it before this entry */
          alloc_map_[start] = AllocMapEntry(count, tag);
          last_alloc_start_ = start;
          allocated_ += count;
          // cout << "MultithreadedCacheHelper got " << start
               // << " for " << count
               // << " allocated = " << allocated_ << endl;
          return start;
        } else {
          start = allocated_start + allocated_count;
        }
      }
      /* If no wait, return size_, indicating there's no more space */
      if (!wait) {
        cerr << "MultithreadedCacheHelper has no more space\n";
        cout << "need " << count << endl;
        cout << "allocated " << allocated_ << endl;
        cout << "size " << size_ << endl;
        print_space();
        return size_;
      }
      /* No more space, wait to be notified */
      // cout << "MultithreadedCacheHelper waits for more space\n";
      // cout << " allocated = " << allocated_ << endl;
      // cvar_.wait(lock);
      if (!cvar_.timed_wait(lock,
          boost::posix_time::milliseconds(12000))) {
        cerr << "MultithreadedCacheHelper waits for more space timed out\n";
        cout << "need " << count << endl;
        cout << "allocated " << allocated_ << endl;
        cout << "size " << size_ << endl;
        print_space();
        return size_;
      }
    }
  }
  void put(size_t start, size_t count) {
    boost::unique_lock<boost::mutex> lock(mutex_);
    alloc_map_.erase(start);
    allocated_ -= count;
    // cout << "MultithreadedCacheHelper put " << start
         // << " for " << count
         // << " allocated = " << allocated_ << endl;
    cvar_.notify_all();
  }
  void print_space() {
    for (AllocMap::iterator map_it = alloc_map_.begin();
           map_it != alloc_map_.end(); map_it++) {
      size_t allocated_start = map_it->first;
      size_t allocated_count = map_it->second.size;
      int tag = map_it->second.tag;
      cerr << "allocated_start = " << allocated_start << endl;
      cerr << "allocated_count = " << allocated_count << endl;
      cerr << "tag = " << tag << endl;
    }
  }
};

template <typename CacheHelper>
struct GpuCache {
  RowData *data_;
  size_t size_;
  size_t memsize_;
  CacheHelper helper_;
  GpuCache() : helper_() {
    init(0);
  }
  ~GpuCache() {
    clear();
  }
  void init(size_t size) {
    size_ = size;
    memsize_ = size_ * sizeof(RowData);
    data_ = NULL;
    if (memsize_) {
      CUDA_CHECK(cudaMalloc(&data_, memsize_));
    }
    helper_.init(size);
  }
  void clear() {
    if (data_) {
      CUDA_CHECK(cudaFree(data_));
    }
    init(0);
    helper_.clear();
  }
  size_t size() {
    return size_;
  }
  RowData *get(size_t count, bool wait, int tag = 0) {
    size_t index = helper_.get(count, wait, tag);
    if (index >= size_) {
      /* No more space */
      return NULL;
    }
    return &data_[index];
  }
  void put(RowData *buffer, size_t count) {
    size_t index = static_cast<size_t>(buffer - data_);
    helper_.put(index, count);
  }
  void print_space() {
    helper_.print_space();
  }
};

struct DataStorage {
  enum MemoryType {
    UNINITIALIZED,
    GPU,
    CPU,
    PINNED_CPU
  } type_;
  size_t size_;
  size_t memsize_;
  RowData *ptr_;
  void init(size_t size, MemoryType type) {
    if (type_ != UNINITIALIZED) {
      CHECK_EQ(type, type_);
      CHECK_EQ(size, size_);
      CHECK(ptr_);
      return;
    }
    CHECK(!size_);
    CHECK(!memsize_);
    CHECK(!ptr_);
    type_ = type;
    size_ = size;
    memsize_ = size_ * sizeof(RowData);
    switch (type_) {
      case GPU:
        init_gpu();
        break;
      case CPU:
        init_cpu();
        break;
      case PINNED_CPU:
        init_pinned_cpu();
        break;
      default:
        CHECK_EQ(type_, UNINITIALIZED);
    }
  }
  void init_gpu() {
    CHECK_EQ(type_, GPU);
    if (!memsize_) {
      return;
    }
    CHECK(!ptr_);
    CUDA_CHECK(cudaMalloc(&ptr_, memsize_));
  }
  void init_cpu() {
    CHECK_EQ(type_, CPU);
    if (!memsize_) {
      return;
    }
    CHECK(!ptr_);
    ptr_ = reinterpret_cast<RowData *>(malloc(memsize_));
    CHECK(ptr_);
  }
  void init_pinned_cpu() {
    CHECK_EQ(type_, PINNED_CPU);
    if (!memsize_) {
      return;
    }
    CHECK(!ptr_);
    mallocHost(&ptr_, memsize_);
  }
  void zerofy_data_cpu() {
    CHECK_EQ(type_, CPU);
    if (!memsize_) {
      return;
    }
    CHECK(ptr_);
    memset(ptr_, 0, memsize_);
  }
  void zerofy_data_gpu(cudaStream_t cuda_stream) {
    CHECK_EQ(type_, GPU);
    if (!memsize_) {
      return;
    }
    CHECK(ptr_);
    CHECK(cuda_stream);
    /* We zerofy the data using cudaMemsetAsync() and
     * call cudaStreamSynchronize() after it. */
    CUDA_CHECK(cudaMemsetAsync(ptr_, 0, memsize_, cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  }
  size_t size() {
    return size_;
  }
  size_t memsize() {
    return memsize_;
  }
  MemoryType type() {
    return type_;
  }
  RowData *data() {
    if (!memsize_) {
      return NULL;
    }
    CHECK(ptr_);
    return ptr_;
  }
  void init_empty() {
    type_ = UNINITIALIZED;
    size_ = 0;
    memsize_ = 0,
    ptr_ = NULL;
  }
  void init_from(const DataStorage& other) {
    init(other.size_, other.type_);
  }
  void copy(const DataStorage& other) {
    clear();
    type_ = other.type_;
    size_ = other.size_;
    memsize_ = other.memsize_;
  }
  void copy_data_gpu(const DataStorage& other, cudaStream_t cuda_stream) {
    CHECK_EQ(type_, GPU);
    CHECK_EQ(other.type_, GPU);
    CHECK_EQ(memsize_, other.memsize_);
    CHECK(cuda_stream);
    CHECK(ptr_);
    CHECK(other.ptr_);
    CUDA_CHECK(cudaMemcpyAsync(
        ptr_, other.ptr_, memsize_, cudaMemcpyDefault, cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  }
  void copy_data_cpu(const DataStorage& other) {
    CHECK_NE(type_, GPU);
    CHECK_NE(other.type_, GPU);
    CHECK_EQ(memsize_, other.memsize_);
    CHECK(ptr_);
    CHECK(other.ptr_);
    memcpy(ptr_, other.ptr_, memsize_);
  }
  void clear() {
    switch (type_) {
      case GPU:
        clear_gpu();
        break;
      case CPU:
        clear_cpu();
        break;
      case PINNED_CPU:
        clear_pinned_cpu();
        break;
      default:
        CHECK_EQ(type_, UNINITIALIZED);
    }
    size_ = 0;
    memsize_ = 0;
    ptr_ = NULL;
    type_ = UNINITIALIZED;
  }
  void clear_gpu() {
    CHECK_EQ(type_, GPU);
    if (ptr_) {
      CUDA_CHECK(cudaFree(ptr_));
    }
  }
  void clear_cpu() {
    CHECK_EQ(type_, CPU);
    if (ptr_) {
      free(ptr_);
    }
  }
  void clear_pinned_cpu() {
    CHECK_EQ(type_, PINNED_CPU);
    if (ptr_) {
      CUDA_CHECK(cudaFreeHost(ptr_));
    }
  }
  DataStorage() {
    init_empty();
  }
  DataStorage(const DataStorage& other) {
    init_empty();
  }
  ~DataStorage() {
    clear();
  }
  DataStorage& operator=(const DataStorage& other) {
    init_empty();
    return *this;
  }
};

#endif  // defined __extended_utils_hpp__
