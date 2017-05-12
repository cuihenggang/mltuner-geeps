#ifndef __tablet_server_hpp__
#define __tablet_server_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

// Lazy table shards server

#include <tbb/tick_count.h>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <set>
#include <map>
#include <queue>

#include "common/extended-utils.hpp"
#include "common/row-op-util.hpp"

#include "mltuner/mltuner-utils.hpp"

using std::string;
using std::vector;
using boost::shared_ptr;

class ServerClientEncode;
class MetadataServer;   /* Used in get_stats() */

class TabletStorage {
 public:
  struct Stats {
    int64_t nr_request;
    int64_t nr_request_prior;
    int64_t nr_local_request;
    int64_t nr_send;
    int64_t nr_update;
    int64_t nr_local_update;
    int64_t nr_rows;
    double send_data_time;
    double clock_ad_apply_op_time;
    double clock_ad_send_pending_time;
    double clock_ad_time_tot;
    double iter_var_time;
    double inc_time;

    void reset() {
      nr_request = 0;
      nr_request_prior = 0;
      nr_local_request = 0;
      nr_send = 0;
      nr_update = 0;
      nr_local_update = 0;
      send_data_time = 0.0;
      clock_ad_time_tot = 0.0;
      clock_ad_apply_op_time = 0.0;
      clock_ad_send_pending_time = 0.0;
      inc_time = 0.0;
      iter_var_time = 0.0;
    }

    Stats() {
      reset();
    }

    Stats& operator += (const Stats& rhs) {
      return *this;
    }
    std::string to_json() {
      std::stringstream ss;
      ss << "{"
         << "\"nr_rows\": " << nr_rows << ", "
         << "\"nr_request\": " << nr_request << ", "
         << "\"nr_request_prior\": " << nr_request_prior << ", "
         << "\"nr_local_request\": " << nr_local_request << ", "
         << "\"nr_send\": " << nr_send << ", "
         << "\"nr_update\": " << nr_update << ", "
         << "\"nr_local_update\": " << nr_local_update << ", "
         << "\"send_data_time\": " << send_data_time << ", "
         << "\"clock_ad_apply_op_time\": " << clock_ad_apply_op_time << ", "
         << "\"clock_ad_send_pending_time\": "
         << clock_ad_send_pending_time << ", "
         << "\"clock_ad_time_tot\": " << clock_ad_time_tot << ", "
         << "\"iter_var_time\": " << iter_var_time << ", "
         << "\"inc_time\": " << inc_time
         << " } ";
      return ss.str();
    }
  };
  Stats server_stats;

  typedef boost::unordered_map<TableRow, uint> Row2Index;

  // typedef boost::unordered_map<iter_t, RowKeys> PendingReadsLog;
  // typedef std::vector<PendingReadsLog> MulticlientPendingReadsLog;

  struct ModelBranch {
    int branch_id;
    DataStorage store;
    DataStorage history_store;
    DataStorage temp_store;
  };
  typedef std::vector<ModelBranch> ModelBranches;

  // struct PendingRead {
    // int branch_id;
  // };
  // typedef std::vector<PendingRead> PendingReads;
      // /* Indexed by client_id */

  struct DataTable {
    VecClock vec_clock;
    iter_t global_clock;
    Row2Index row2idx;
    size_t num_rows;
    std::vector<RowKey> row_keys;
    ModelBranches model_branches;
    // PendingReads pending_reads;
  };
  typedef std::vector<DataTable> DataTables;

 private:
  uint channel_id;
  uint num_channels;
  uint process_id;
  uint num_processes;
  uint num_clients;

  boost::unordered_map<std::string, table_id_t> table_directory;

  shared_ptr<ServerClientEncode> communicator;

  cudaStream_t cuda_stream;
  cublasHandle_t cublas_handle;

  vector<iter_t> worker_started_states;
  iter_t all_workers_started;

  DataTables data_tables;
  shared_ptr<BranchManager> branch_manager;

  Config config;

  tbb::tick_count start_time;
  tbb::tick_count first_come_time;

 private:
  template<class T>
  void resize_storage(vector<T>& storage, uint size) {
    if (storage.capacity() <= size) {
      uint capacity = get_nearest_power2(size);
      storage.reserve(capacity);
      // cerr << "capacity is " << capacity << endl;
    }
    if (storage.size() <= size) {
      storage.resize(size);
      // cerr << "size is " << size << endl;
    }
  }
  uint get_row_idx(table_id_t table, row_idx_t row);
  void process_all_client_subscribed_reads(
      iter_t clock, uint table_id,
      int branch_id, iter_t internal_clock);
  void process_subscribed_reads(
      uint client_id, iter_t clock, uint table_id,
      int branch_id, iter_t internal_data_age);
  void apply_updates_blank(
      int branch_id, uint branch_idx, uint table_id,
      RowOpVal *update_rows,size_t batch_size);
  void make_branch_for_table(
      int branch_id, Tunable& tunable,
      int parent_branch_id, iter_t clock_to_happen, uint table_id);
  void inactivate_branch_for_table(
      int branch_id, iter_t clock_to_happen, uint table_id);
  void save_branch_for_table(
      int branch_id, iter_t clock_to_happen, uint table_id);

  void init_user_defined_update_states(
      int branch_id, uint branch_idx, uint table_id, size_t batch_size);
  void apply_user_defined_updates(
      int branch_id, uint branch_idx, uint table_id,
      RowOpVal *update_rows, size_t batch_size,
      const Tunable& tunable);
  void apply_updates_momentum(
      int branch_id, uint branch_idx, uint table_id,
      RowOpVal *update_rows, size_t batch_size,
      const Tunable& tunable);
  void apply_updates_adadelta(
      int branch_id, uint branch_idx, uint table_id,
      RowOpVal *update_rows, size_t batch_size,
      const Tunable& tunable);
  void apply_updates_adagrad(
      int branch_id, uint branch_idx, uint table_id,
      RowOpVal *update_rows, size_t batch_size,
      const Tunable& tunable);
  void apply_updates_adam(
      int branch_id, uint branch_idx, uint table_id,
      RowOpVal *update_rows, size_t batch_size,
      const Tunable& tunable);
  void apply_updates_nesterov(
      int branch_id, uint branch_idx, uint table_id,
      RowOpVal *update_rows, size_t batch_size,
      const Tunable& tunable);
  void apply_updates_rmsprop(
      int branch_id, uint branch_idx, uint table_id,
      RowOpVal *update_rows, size_t batch_size,
      const Tunable& tunable);
  void apply_updates_adarevision_with_momentum(
      int branch_id, uint branch_idx, uint table_id,
      RowOpVal *update_rows, size_t batch_size,
      const Tunable& tunable);
  void apply_updates_adadelta_with_momentum(
      int branch_id, uint branch_idx, uint table_id,
      RowOpVal *update_rows, size_t batch_size,
      const Tunable& tunable);
  void apply_updates_adagrad_with_momentum(
      int branch_id, uint branch_idx, uint table_id,
      RowOpVal *update_rows, size_t batch_size,
      const Tunable& tunable);
  void apply_updates_rmsprop_with_momentum(
      int branch_id, uint branch_idx, uint table_id,
      RowOpVal *update_rows, size_t batch_size,
      const Tunable& tunable);

 public:
  TabletStorage(
      uint channel_id, uint num_channels, uint process_id, uint num_processes,
      shared_ptr<ServerClientEncode> communicator,
      cudaStream_t cuda_stream, cublasHandle_t cublas_handle,
      const Config& config);
  void update_row_batch(
      uint client_id, iter_t clock, uint table_id, int branch_id,
      RowKey *row_keys, RowOpVal *updates, uint batch_size);
  // void read_row_batch(
      // uint client_id, iter_t clock, uint table_id, int branch_id);
  void clock(
      uint client_id, iter_t clock, uint table_id,
      int branch_id, iter_t internal_clock);
  void get_stats(
      uint client_id, shared_ptr<MetadataServer> metadata_server);
    /* Now it also needs the stats from the MetadataServer.
     * This is just a work-around, and we need to fix it in the future.
     */
  void reset_perf_counters();
  void worker_started(uint client_id);
  void make_branch(
      int branch_id, Tunable& tunable, int flag,
      int parent_branch_id, iter_t clock_to_happen);
  void inactivate_branch(int branch_id, iter_t clock_to_happen);
  void save_branch(int branch_id, iter_t clock_to_happen);
};


#endif  // defined __tablet_server_hpp__
