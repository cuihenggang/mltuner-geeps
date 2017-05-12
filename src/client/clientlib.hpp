#ifndef __lazytable_impl_hpp__
#define __lazytable_impl_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <zmq.hpp>
#include <tbb/tick_count.h>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <set>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <queue>

#include "stats-tracker.hpp"
#include "common/extended-utils.hpp"
#include "common/router-handler.hpp"
#include "common/work-pusher.hpp"
#include "server/server-entry.hpp"
#include "mltuner/mltuner-entry.hpp"
#include "mltuner/mltuner-utils.hpp"

using std::string;
using std::vector;
using std::pair;
using std::cerr;
using std::cout;
using std::endl;
using std::make_pair;
using boost::ref;
using boost::format;
using boost::bind;
using boost::make_shared;
using boost::shared_ptr;
using boost::unique_lock;
using boost::mutex;
using boost::condition_variable;


class ServerClientDecode;
class ClientServerEncode;

struct FlatOps : DataStorage {
  enum {
    NONE,
    INC,
  } flag;
  int branch_id;
  int clock;
  void reset() {
    flag = NONE;
    branch_id = -1;
    clock = UNINITIALIZED_CLOCK;
  }
  FlatOps() : DataStorage() {
    reset();
  }
};

struct OpMemBufferPool {
  bool gpu_;
  size_t size_;
  size_t free_start_;
  size_t free_end_;
  size_t num_free_;
  std::vector<FlatOps *> op_mems;
  boost::mutex mutex_;
  boost::condition_variable cvar_;
  OpMemBufferPool() :
      gpu_(false), size_(0), free_start_(0), free_end_(0), num_free_(0) {}
  void init(bool gpu, size_t size, size_t flatops_size) {
    boost::unique_lock<boost::mutex> lock(mutex_);
    gpu_ = gpu;
    size_ = size;
    free_start_ = 0;
    free_end_ = size - 1;
    num_free_ = size;
    if (op_mems.size()) {
      for (size_t i = 0; i < op_mems.size(); i++) {
        delete op_mems[i];
      }
    }
    op_mems.resize(size);
    for (size_t i = 0; i < size; i++) {
      op_mems[i] = new FlatOps();
      if (gpu_) {
        op_mems[i]->init(flatops_size, DataStorage::GPU);
        /* We didn't zerofy the data here,
         * because we assume the flat_op data will be zerofied when used */
      } else {
        op_mems[i]->init(flatops_size, DataStorage::CPU);
      }
    }
  }
  FlatOps *get() {
    boost::unique_lock<boost::mutex> lock(mutex_);
    // CHECK_GT(num_free_, 0);
    while (num_free_ == 0) {
      // cvar_.wait(lock);
      if (!cvar_.timed_wait(lock,
          boost::posix_time::milliseconds(12000))) {
        cerr << "OpMemBufferPool waits for more space timed out\n";
      }
    }
    CHECK_LT(free_start_, size_);
    FlatOps *free_entry = op_mems[free_start_];
    free_start_ = (free_start_ + 1) % size_;
    num_free_--;
    return free_entry;
  }
  void put(FlatOps *flat_ops) {
    boost::unique_lock<boost::mutex> lock(mutex_);
    CHECK(size_);
    free_end_ = (free_end_ + 1) % size_;
    num_free_++;
    CHECK_EQ(op_mems[free_end_], flat_ops);
    cvar_.notify_all();
  }
  ~OpMemBufferPool() {
    if (op_mems.size()) {
      for (size_t i = 0; i < op_mems.size(); i++) {
        delete op_mems[i];
      }
    }
  }
};

typedef std::vector<FlatOps *> StaticOpLog; /* Indexed by clock */

struct StaticCacheMetadata {
  size_t cache_idx;
  int tablet_server_id;
  size_t row_idx;   /* oplog index */
};
typedef boost::unordered_map<TableRow, StaticCacheMetadata> StaticCacheIndex;

struct PerChannelIndexInfo {
  size_t index_start;
  size_t index_size;
};

struct SharedBuffer {
  RowData *buffer;
  int branch_id;
  size_t num_rows;
  int ref_count;
  size_t storage_offset;
  bool need_keep;
  bool has_local_storage;
  bool pinned_entry;
  boost::mutex mutex;
  boost::condition_variable cvar;
  SharedBuffer() :
      buffer(NULL), branch_id(-1), num_rows(0),
      ref_count(0), storage_offset(std::numeric_limits<size_t>::max()),
      need_keep(false), has_local_storage(false), pinned_entry(false) {}
};

struct OpDataBuffer {
  RowData *buffer;
  int branch_id;
  bool buffer_being_used;
  bool updates_ready;
  boost::mutex mutex;
  boost::condition_variable cvar;
  OpDataBuffer() {
    init();
  }
  OpDataBuffer& operator=(const OpDataBuffer& cc) {
    /* Uncopiable */
    return *this;
  }
  OpDataBuffer(const OpDataBuffer& copy) {
    /* Uncopiable */
    init();
  }
  void init() {
    buffer = NULL;
    branch_id = -1;
    buffer_being_used = false;
    updates_ready = false;
  }
};

struct OpInfo {
  enum OpType {
    READ,
    POST_READ,
    PRE_UPDATE,
    UPDATE,
    LOCAL_ACCESS,
    POST_LOCAL_ACCESS,
    CLOCK,
  } type;
  table_id_t table;
  std::vector<row_idx_t> rows;
  uint table_id;
  size_t num_vals_limit;
  int prestep_handle;
  bool fetch_local;   /* For local access only */
  bool keep_local;    /* For local access only */
  enum IndexType {
    SINGLE_INDEX,
    DOUBLE_INDEX,
  } index_type;
  size_t row_index_size;
  DoubleIndex *row_index_gpu;
  DoubleIndex *row_index_cpu;
  std::vector<PerChannelIndexInfo> index_infos;
  bool batch_last_update;
  bool table_last_update;
  std::vector<OpDataBuffer> op_data_buffers;   /* One buffer for every clock */
  SharedBuffer *shared_buffer;  /* Shared buffer is only used by local ops */
  iter_t last_finished_clock;
      /* No lock protecting this, because it's only used
       * by the background worker threads */
  int branch_id;
  double read_time;
  double fetch_time;
  double alloc_wait_time;
  double update_time;
  double keep_time;
  double reclaim_wait_time;
  size_t rows_fetched;
  size_t rows_kept;
  OpInfo(OpType optype,
      int prestep_handle, bool keep_local = false)
      : type(optype), prestep_handle(prestep_handle), keep_local(keep_local) {
    /* Only for POST_READ and UPDATE */
    CHECK(type == POST_READ || type == UPDATE || type == POST_LOCAL_ACCESS);
    init();
  }
  OpInfo(OpType optype)
    : type(optype) {
    /* Only for CLOCK */
    CHECK_EQ(type, CLOCK);
    init();
  }
  OpInfo(OpType optype,
      table_id_t table, const std::vector<row_idx_t> rows,
      size_t num_vals_limit = std::numeric_limits<size_t>::max(),
      bool fetch_local = false)
      : type(optype), table(table), rows(rows),
        num_vals_limit(num_vals_limit), fetch_local(fetch_local) {
    init();
  }
  void init() {
    table_id = table;
        /* Assuming table keys are consecutive numbers from zero */
    row_index_size = 0;
    row_index_gpu = NULL;
    row_index_cpu = NULL;
    last_finished_clock = UNINITIALIZED_CLOCK;
    read_time = 0;
    fetch_time = 0;
    alloc_wait_time = 0;
    update_time = 0;
    keep_time = 0;
    reclaim_wait_time = 0;
    rows_fetched = 0;
    rows_kept = 0;
  }
};

typedef std::vector<OpInfo> OpSeq;
// struct OpSeq {
  // std::vector<OpInfo> op_infos;
// }

// typedef GpuCache<SimpleCacheHelper> ThreadCache;
typedef GpuCache<MultithreadedCacheHelper> ThreadCache;

// typedef graphlab::hopscotch_map<TableRow, uint> LocalStorageIndex;
// struct LocalStorage {
  // size_t num_rows;
  // LocalStorageIndex index;
  // DataStorage data;
// };
typedef DataStorage LocalStorage;

struct LocalStorageOfModelBranch {
  int branch_id;
  LocalStorage local_storage;
};
typedef std::vector<LocalStorageOfModelBranch> LocalStorageOfModelBranches;

struct KeyBatchInfo {
  table_id_t table;
  vector<row_idx_t> rows;
  size_t num_rows;
  size_t num_fetches;
  size_t num_keeps;
  bool fetchkeep;
  bool pinned;
  size_t key_index;
  SharedBuffer *shared_buffer;
  KeyBatchInfo() :
      num_fetches(0), num_keeps(0), fetchkeep(false), pinned(false),
      shared_buffer(NULL) {}
};
typedef boost::unordered_map<TableRow, KeyBatchInfo> KeyBatchInfoMap;

struct ThreadData {
  uint thread_id;
  iter_t current_clock;
  Stats thread_stats;

  /* For background opseq worker */
  bool bg_worker_started;
  boost::shared_ptr<boost::thread> alloc_worker_thread;
  boost::shared_ptr<boost::thread> reclaim_worker_thread;

  /* For memory affinity */
  uint numa_node_id;

  /* Local storage */
  LocalStorageOfModelBranches local_storage_of_model_branches;

  size_t thread_cache_size;
  size_t fetchkeep_local_data_size;
  RowKeys param_row_keys_gpu;
  RowKeys param_row_keys_cpu;

  /* Operation sequence, collected by the virtual iteration */
  OpSeq opseq;
  ThreadCache thread_cache;
  int last_handle;

  cudaStream_t cuda_stream;
  cublasHandle_t cublas_handle;
  RowData *staging_mem;
  size_t staging_mem_size;

  /* Branch schedule of the current clock */
  int branch_id;
  iter_t internal_clock;
  uint branch_idx;
  Tunable tunable;
};

typedef DataStorage DataCache;
struct DataCacheOfModelBranch {
  int branch_id;
  iter_t data_age;
  VecClock per_server_data_age;
  iter_t internal_data_age;
  VecClock per_server_internal_data_age;
  DataStorage data_cache;
  void reset() {
    branch_id = -1;
    data_age = UNINITIALIZED_CLOCK;
    internal_data_age = UNINITIALIZED_CLOCK;
    for (uint i = 0; i < per_server_data_age.size(); i++) {
      per_server_data_age[i] = UNINITIALIZED_CLOCK;
    }
    for (uint i = 0; i < per_server_internal_data_age.size(); i++) {
      per_server_internal_data_age[i] = UNINITIALIZED_CLOCK;
    }
  }
};
typedef std::vector<DataCacheOfModelBranch> DataCacheOfModelBranches;

struct StaticCache {
  size_t num_rows;
  std::vector<RowKey> row_keys;
  StaticCacheIndex static_cache_index;
  DataCacheOfModelBranches data_cache_of_model_branches;
  StaticOpLog oplog;
  OpMemBufferPool opmem_buffer_pool;
  std::vector<size_t> per_server_row_start;
  std::vector<size_t> per_server_num_rows;
  StaticCache() {}
  StaticCache(const StaticCache& other) {}
  StaticCache& operator=(const StaticCache& other) {
    /* Just to make std::vector happy */
    return *this;
  }
};

struct CachedTable {
  StaticCache static_cache_cpu;
  StaticCache static_cache_gpu;
  int branch_id;
  VecClock server_clock;
  iter_t server_clock_min;
  CachedTable() {}
  CachedTable(const CachedTable& other) {}
  CachedTable& operator=(const CachedTable& other) {
    /* Just to make std::vector happy */
    return *this;
  }
};
typedef std::vector<CachedTable> CachedTables;

struct CommunicationChannel {
  /* Communication stack */
  boost::shared_ptr<zmq::context_t> zmq_ctx;
  boost::shared_ptr<RouterHandler> router;
  boost::shared_ptr<ClientServerEncode> encoder;
  boost::shared_ptr<ServerClientDecode> decoder;
  boost::shared_ptr<WorkPusher> work_pusher;
  boost::shared_ptr<boost::thread> bg_worker_thread;
  boost::shared_ptr<boost::thread> server_thread;
  /* Shared Cache */
  CachedTables cached_tables;
  // StaticCache static_cache_cpu;
  // StaticCache static_cache_gpu;
  std::vector<iter_t> server_started;
  iter_t all_server_started;
  uint numa_node_id;
  BgthreadStats bgthread_stats;
  /* For the virtual iteration */
  size_t num_oplog_entries;
  size_t comm_buffer_size;
  RowData *send_buffer;
  RowData *recv_buffer;
  cudaStream_t cuda_stream_recv;
  cublasHandle_t cublas_handle_recv;
  cudaStream_t cuda_stream_send;
  cublasHandle_t cublas_handle_send;
  boost::shared_ptr<boost::mutex> mutex;
  boost::shared_ptr<boost::condition_variable> cvar;
  CommunicationChannel& operator=(const CommunicationChannel& cc) {
    /* STL vector needs that to be defined, but it shouldn't
     * actually be called.
     */
    assert(0);
    return *this;
  }
  CommunicationChannel(const CommunicationChannel& copy) {
    /* Uncopiable */
    /* STL vector needs that, and will call this function even though
     * we are not copying anything. So we just leave it blank.
     */
    // assert(0);
  }
  CommunicationChannel() {
    /* Should be constructed mannually */
  }
};

class ClientTunerEncoder;
class TunerClientDecoder;
struct TunerCommunicationChannel {
  /* Communication stack */
  boost::shared_ptr<zmq::context_t> zmq_ctx;
  boost::shared_ptr<RouterHandler> router;
  boost::shared_ptr<ClientTunerEncoder> encoder;
  boost::shared_ptr<TunerClientDecoder> decoder;
  boost::shared_ptr<boost::thread> tuner_thread;
  /* The scheduler communication channel shares the same
   * background pusher thread as comm_channel[0] */
};


class ClientLib;
extern ClientLib *client_lib;  // singleton

class ClientLib {
  struct bgcomm_clock_msg_t {
    iter_t clock;
    uint table_id;
    int branch_id;
    iter_t internal_clock;
  };

  struct bgcomm_forward_make_branch_msg_t {
    int branch_id;
    iter_t clock_to_happen;
    int parent_branch_id;
  };

  struct bgcomm_forward_inactivate_branch_msg_t {
    int branch_id;
    iter_t clock_to_happen;
  };

  Stats proc_stats;

  std::vector<size_t> rows_per_channel;   /* Indexed by table_id */
  std::vector<CommunicationChannel> comm_channels;
  uint num_channels;

  uint process_id;

  /* Data storage */
  std::map<std::string, table_id_t> table_directory;

  /* Thread_management */
  boost::mutex global_clock_mutex;
  boost::condition_variable global_clock_cvar;

  // boost::thread_specific_ptr<ThreadData> thread_data;
  /* NOTE: this shared_ptr only works when you have
   * only one worker thread per client */
  boost::shared_ptr<ThreadData> thread_data;
  uint nr_threads;

  /* Clock for this client process */
  VecClock all_table_global_clocks;
  iter_t local_data_clock;
  // TODO(hengganc): these should be atomic types

  /* Server states */
  vector<string> host_list;
  vector<uint> port_list;
  uint tcp_base_port;
  uint num_servers;

  /* Memory usage */
  size_t ngr_capacity;

  /* Virtual iteration states */
  bool virtual_iteration_all_finished;
  boost::mutex virtual_iteration_mutex;
  boost::condition_variable virtual_iteration_cvar;

  /* For ModelSitter */
  boost::shared_ptr<BranchManager> branch_manager;
  TunerCommunicationChannel tuner_comm_channel;

  /* Log states */
  tbb::tick_count start_time;
  tbb::tick_count first_come_time;

  /* Config */
  const GeePsConfig config;

 private:
  // This class allows a singleton object.  Hence the private constructor.
  ClientLib(uint process_id, const GeePsConfig& config);

  void init_static_cache(
      StaticCache& static_cache, uint num_servers);
  void init_comm_channel(
      uint channel_id, const GeePsConfig& config);
  void init_tuner_comm_channel(const GeePsConfig& config);
  col_idx_t get_col_count(table_id_t table);
  uint64_t get_hash(table_id_t table, row_idx_t row_id);
  uint get_machine_id(table_id_t table, row_idx_t row_id);
  uint get_machine_id(uint64_t hash);
  uint get_channel_id(table_id_t table, row_idx_t row_id);
  uint get_channel_id(const TableRow& table_row);
  uint get_channel_id(uint64_t hash);
  void log_iter(const ThreadData& thread_data_ref);
  void reset_perf_counters();
  void create_oplog_entry(
      uint channel_id, size_t oplog_idx, uint table_id, bool gpu);
  void create_fetchlog_entries(iter_t clock, bool gpu);
  void reclaim_oplog(
      uint channel_id, iter_t start_clock, iter_t end_clock, uint table_id,
      bool gpu);
  int read_row_batch_static_cache(
      RowData *buffer, OpInfo& op_info, int branch_id, uint branch_idx,
      iter_t required_internal_data_age,
      cudaStream_t& cuda_stream, RowData *staging_mem);
  void read_batch_gpu(
      RowData *buffer, OpInfo& op_info, uint channel_id,
      int branch_id, uint branch_idx, iter_t required_internal_data_age,
      cudaStream_t& cuda_stream);
  void read_batch_cpu(
      RowData *buffer, OpInfo& op_info, uint channel_id,
      int branch_id, uint branch_idx, iter_t required_internal_data_age);
  bool update_batch_static_cache(
      OpInfo& op_info, int branch_id, const RowOpVal *updates, iter_t clock,
      cudaStream_t& cuda_stream, RowData *staging_mem);
  void update_batch_gpu(
      OpInfo& op_info, const RowOpVal *updates, uint channel_id, iter_t clock,
      int branch_id, cudaStream_t& cuda_stream);
  void update_batch_cpu(
      OpInfo& op_info, const RowOpVal *updates, uint channel_id, iter_t clock,
      int branch_id);
  void clock_all(iter_t clock);
  void clock_table(iter_t clock, uint table_id);
  void push_clock_work(
      uint channel_id, iter_t clock, uint table_id,
      int branch_id, iter_t internal_clock);
  bool recv_row_batch_static_cache(
      uint channel_id, uint server_id, iter_t data_age, iter_t self_clock,
      int branch_id, uint branch_idx,
      iter_t internal_data_age, iter_t internal_self_clock,
      uint table_id, RowKey *row_keys, RowData *row_data, uint batch_size);
  void recv_row_batch_gpu(
      uint channel_id, uint server_id, iter_t data_age, iter_t self_clock,
      int branch_id, uint branch_idx,
      iter_t internal_data_age, iter_t internal_self_clock,
      uint table_id, RowKey *row_keys, RowData *row_data, uint batch_size);
  void recv_row_batch_cpu(
      uint channel_id, uint server_id, iter_t data_age, iter_t self_clock,
      int branch_id, uint branch_idx,
      iter_t internal_data_age, iter_t internal_self_clock,
      uint table_id, RowKey *row_keys, RowData *row_data, uint batch_size);
  void push_updates_static_cache(
      uint channel_id, iter_t clock, uint table_id,
      int update_branch_id, int read_branch_id);
  bool find_row_static_cache(
      table_id_t table, row_idx_t row_id, uint channel_id,
      bool unique_request = true, bool non_blocking = false,
      bool force_refresh = false);
  bool find_row_cbk_static_cache(
      table_id_t table, row_idx_t row_id, uint32_t tablet_server_id,
      uint channel_id);
  void read_batch_local(RowData *buffer, OpInfo& op_info,
      LocalStorage& local_storage, size_t local_storage_offset,
      cudaStream_t& cuda_stream, RowData *staging_mem);
  void update_batch_local(OpInfo& op_info, const RowOpVal *updates,
      LocalStorage& local_storage, size_t local_storage_offset,
      cudaStream_t& cuda_stream, RowData *staging_mem);

  void vi_thread_summarize();
  void vi_decide_pinned_data(KeyBatchInfoMap& local_key_batches);
  void vi_create_thread_cache();
  void vi_create_local_storage(KeyBatchInfoMap& local_key_batches);
  void vi_decide_param_cache();
  void vi_process_channel_finalize(
      ThreadData& thread_data_ref, uint channel_id, bool gpu);
  void vi_process_channel_table_finalize(
      ThreadData& thread_data_ref, uint channel_id, uint table_id, bool gpu);
  void vi_process_finalize();
  void vi_thread_finalize();
  void vi_create_double_index(OpInfo& opinfo);

 public:
  static void CreateInstance(uint process_id, const GeePsConfig& config) {
    if (client_lib == NULL) {
      client_lib = new ClientLib(process_id, config);
    }
  }

  /* Routines called by application code */
  std::string json_stats();
  void shutdown();
  void find_row(
      table_id_t table, row_idx_t row_id,
      bool unique_request = true, bool non_blocking = false,
      bool force_refresh = false);

  /* Routines only executed by the background thread */
  void thread_start();
  void thread_stop();
  void find_row_cbk(
      table_id_t table, row_idx_t row_id, uint32_t tablet_server_id);
  void get_stats_cbk(const string& server_stats);
  void recv_row_batch(
      uint channel_id, uint server_id, iter_t data_age, iter_t self_clock,
      int branch_id, iter_t internal_data_age, iter_t internal_self_clock,
      uint table_id, RowKey *row_keys, RowData *row_data, uint batch_size);
  void server_clock(
      uint channel_id, uint server_id, iter_t iter, uint table_id);
  void server_started(uint channel_id, uint server_id);
  void cbk_iterate(uint channel_id, vector<ZmqPortableBytes>& msgs);
  void push_updates(
      uint channel_id, iter_t clock, uint table_id,
      int branch_id, iter_t internal_clock);
  void push_worker_started_work(uint channel_id);
  void cbk_worker_started(uint channel_id, vector<ZmqPortableBytes>& args);
  void worker_started(uint channel_id);
  void alloc_worker_entry();
  void reclaim_worker_entry();
  void localaccess_impl(
      OpInfo& op_info, iter_t clock, OpSeq& opseq, ThreadCache& thread_cache,
      LocalStorageOfModelBranches& local_storage_of_model_branches,
      RowData *staging_mem, cudaStream_t cuda_stream, bool is_from_bg_worker);
  void read_impl(
      OpInfo& op_info, iter_t clock, OpSeq& opseq, ThreadCache& thread_cache,
      RowData *staging_mem, cudaStream_t cuda_stream, bool is_from_bg_worker);
  void preupdate_impl(
      OpInfo& op_info, iter_t clock, OpSeq& opseq, ThreadCache& thread_cache,
      RowData *staging_mem, cudaStream_t cuda_stream, bool is_from_bg_worker);
  void postlocalaccess_impl(
      OpInfo& op_info, iter_t clock, OpSeq& opseq, ThreadCache& thread_cache,
      LocalStorageOfModelBranches& local_storage_of_model_branches,
      RowData *staging_mem, cudaStream_t cuda_stream, bool is_from_bg_worker);
  void postread_impl(
      OpInfo& op_info, iter_t clock, OpSeq& opseq, ThreadCache& thread_cache,
      RowData *staging_mem, cudaStream_t cuda_stream, bool is_from_bg_worker);
  void update_impl(
      OpInfo& op_info, iter_t clock, OpSeq& opseq, ThreadCache& thread_cache,
      RowData *staging_mem, cudaStream_t cuda_stream, bool is_from_bg_worker);
  void reclaim_worker_clock(
      OpInfo& op_info, RowData *staging_mem,
      cudaStream_t cuda_stream, cublasHandle_t cublas_handle);

  /* Interfaces for virtual iteration */
  int virtual_read_batch(
      table_id_t table, const vector<row_idx_t>& row_ids,
      size_t num_vals_limit);
  int virtual_postread_batch(int prestep_handle);
  int virtual_preupdate_batch(
      table_id_t table, const vector<row_idx_t>& row_ids,
      size_t num_vals_limit);
  int virtual_update_batch(int prestep_handle);
  int virtual_localaccess_batch(
      table_id_t table, const vector<row_idx_t>& row_ids,
      size_t num_vals_limit, bool fetch);
  int virtual_postlocalaccess_batch(int prestep_handle, bool keep);
  int virtual_clock();
  int virtual_op(const OpInfo& opinfo);
  void copy_row_keys_from_preop(OpInfo& op_info);
  void finish_virtual_iteration();

  /* Interfaces for GeePS */
  void start_opseq();
  void app_acquire_routine(RowData **buffer_mem_ret, int handle);
  void app_release_routine(int handle);

  /* Routines for ModelSitter */
  void iterate();
  void report(double progress);
  int get_branch_id(iter_t clock);
  int get_branch_id_no_wait(iter_t clock);
  Tunable get_tunable();
  void get_current_branch_info(Tunable *tunable, int *flag);
  shared_ptr<BranchManager> get_branch_manager();
  uint get_branch_idx(int branch_id);
  void recv_branch_schedules(uint batch_size, iter_t *clocks, int *branch_ids);
  void wait_for_all_clocks(ScopedLock *global_clock_lock_ptr, iter_t clock);
  void signal_bgworker_forward_msg(vector<ZmqPortableBytes>& args);
  void cbk_forward_msg(uint channel_id, vector<ZmqPortableBytes>& args);
  void recv_make_branch(
      int branch_id, const Tunable& tunable, int flag,
      int parent_branch_id, iter_t clock_to_happen,
      vector<ZmqPortableBytes>& args);
  void make_branch_for_table(int branch_id, uint table_id);
  void make_branch_for_local_data(int branch_id);
  void recv_inactivate_branch(
      int branch_id, iter_t clock_to_happen,
      vector<ZmqPortableBytes>& args);
  void inactivate_branch_for_table(int branch_id, uint table_id);
  void inactivate_branch_for_local_data(int branch_id);
  void recv_save_branch(
      int branch_id, iter_t clock_to_happen,
      vector<ZmqPortableBytes>& args);
  void save_branch_for_local_data(int branch_id);
};

#endif  // defined __lazytable_impl_hpp__
