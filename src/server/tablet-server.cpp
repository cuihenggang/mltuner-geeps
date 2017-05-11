/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

/* LazyTable tablet server */

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <utility>
#include <string>
#include <vector>

#include "server-encoder-decoder.hpp"
#include "tablet-server.hpp"
#include "metadata-server.hpp"

using std::string;
using std::cerr;
using std::cout;
using std::endl;
using std::vector;
using std::pair;
using std::make_pair;
using boost::format;
using boost::lexical_cast;
using boost::shared_ptr;
using boost::make_shared;

TabletStorage::TabletStorage(
    uint channel_id, uint num_channels, uint process_id, uint num_processes,
    shared_ptr<ServerClientEncode> communicator,
    cudaStream_t cuda_stream, cublasHandle_t cublas_handle,
    const Config& config) :
      channel_id(channel_id), num_channels(num_channels),
      process_id(process_id), num_processes(num_processes),
      num_clients(num_processes),
      communicator(communicator),
      cuda_stream(cuda_stream), cublas_handle(cublas_handle),
      // prio_multi_client_pending_reads_log(num_processes),
      // multi_client_pending_reads_log(num_processes),
      config(config) {
  /* Initialize server fields */
  worker_started_states.resize(num_processes);
  for (uint i = 0; i < num_processes; i++) {
    worker_started_states[i] = 0;
  }
  all_workers_started = 0;

  /* Initialize data tables */
  data_tables.resize(config.num_tables);
  for (uint table_id = 0; table_id < data_tables.size(); table_id++) {
    DataTable& data_table = data_tables[table_id];
    data_table.vec_clock.resize(num_processes);
    for (uint client_id = 0; client_id < num_processes; client_id++) {
      data_table.vec_clock[client_id] = UNINITIALIZED_CLOCK;
    }
    data_table.global_clock = UNINITIALIZED_CLOCK;
    data_table.num_rows = 0;
    ModelBranches& model_branches = data_table.model_branches;
    model_branches.resize(config.num_branches);
    for (uint branch_idx = 0; branch_idx < model_branches.size();
         branch_idx++) {
      model_branches[branch_idx].branch_id = -1;
    }
    // PendingReads& pending_reads = data_table.pending_reads;
    // pending_reads.resize(num_processes);
  }

  /* Initialize branch manager */
  uint num_queues = 0;
  branch_manager = make_shared<BranchManager>(
      config.num_branches, num_queues);
}

uint TabletStorage::get_row_idx(table_id_t table, row_idx_t row) {
  /* !!!FIXME: we are not using index here */
  return row;
}

void TabletStorage::update_row_batch(
      uint client_id, iter_t clock, uint table_id, int branch_id,
      RowKey *row_keys, RowOpVal *updates, uint batch_size) {
  // cout << "update_row_batch(): "
       // << " table_id = " << table_id
       // << " batch_size = " << batch_size << endl;
  server_stats.nr_update += batch_size;
  if (client_id == process_id) {
    server_stats.nr_local_update += batch_size;
  }

  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  BranchInfo *branch_info = branch_manager->branch_info_map[branch_id];
  CHECK(branch_info);
  uint branch_idx = branch_info->branch_idx;
  const Tunable& tunable = branch_info->tunable;
  int branch_flag = branch_info->flag;
  CHECK_LT(branch_idx, model_branches.size()) << " branch_id = " << branch_id;
  ModelBranch& model_branch = model_branches[branch_idx];
  DataStorage& data_store = model_branch.store;
  
  CHECK_LT(client_id, data_table.vec_clock.size());
  iter_t cur_clock = data_table.vec_clock[client_id];
  if (cur_clock != UNINITIALIZED_CLOCK) {
    CHECK_EQ(clock, cur_clock + 1) << " table = " << table_id;
  }

  if (batch_size == 0) {
    return;
  }

  if (data_table.row_keys.size() == 0) {
    data_table.num_rows = batch_size;
    data_table.row_keys.resize(data_table.num_rows);
    memcpy(data_table.row_keys.data(), row_keys,
        batch_size * sizeof(RowKey));
  }
  CHECK_EQ(data_table.row_keys.size(), batch_size);

  if (data_store.size() == 0) {
    data_store.init(batch_size, DataStorage::CPU);
    data_store.zerofy_data_cpu();
    if (config.update_func != "blank") {
      init_user_defined_update_states(
          branch_id, branch_idx, table_id, batch_size);
    }
    /* HACK: We assume the first (and only the first) update is
     * used to initialize the data */
    apply_updates_blank(branch_id, branch_idx, table_id, updates, batch_size);
  } else if (branch_flag == TRAINING_BRANCH) {
    CHECK_EQ(data_store.size(), batch_size);
    if (config.update_func == "blank") {
      apply_updates_blank(branch_id, branch_idx, table_id, updates, batch_size);
    } else {
      apply_user_defined_updates(
          branch_id, branch_idx, table_id, updates, batch_size, tunable);
    }
  }
}

void TabletStorage::apply_updates_blank(
    int branch_id, uint branch_idx,
    uint table_id, RowOpVal *update_rows, size_t batch_size) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];
  DataStorage& data_store = model_branch.store;
  
  val_t *update = reinterpret_cast<val_t *>(update_rows);
  CHECK_EQ(data_store.size(), batch_size);
  val_t *master_data = reinterpret_cast<val_t *>(data_store.data());
  size_t num_vals = batch_size * ROW_DATA_SIZE;

  cpu_add(num_vals,
      master_data,
      update,
      master_data);
}

// void TabletStorage::read_row_batch(
    // uint client_id, iter_t clock, uint table_id, int branch_id) {
  // CHECK_LT(table_id, data_tables.size());
  // DataTable& data_table = data_tables[table_id];
  // CHECK_LT(client_id, data_table.pending_reads.size());
  // data_table.pending_reads[client_id].branch_id = branch_id;
// }

void TabletStorage::process_all_client_subscribed_reads(
    iter_t clock, uint table_id,
    int branch_id, iter_t internal_clock) {
  /* Not starting from the same client */
  uint client_to_start = clock % num_clients;
  /* Process pending reads */
  for (uint i = 0; i < num_clients; i++) {
    uint client_id = (client_to_start + i) % num_clients;
    process_subscribed_reads(
        client_id, clock, table_id, branch_id, internal_clock);
  }
}

void TabletStorage::process_subscribed_reads(
    uint client_id, iter_t clock, uint table_id,
    int branch_id, iter_t internal_data_age) {
  /* !!!FIXME: should use a proper way to find the rows to send */
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  uint branch_idx = branch_manager->get_branch_idx(branch_id);
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];
  DataStorage& data_store = model_branch.store;
  RowKeys& row_keys = data_table.row_keys;
  // cout << "process_subscribed_reads(): "
       // << " table_id = " << table_id
       // << " data_store.size() = " << data_store.size() << endl;
  // CHECK(data_store.size())
      // << "branch_id = " << branch_id << ", clock = " << clock;
  CHECK_EQ(data_store.size(), row_keys.size());
  RowData *row_data = data_store.data();
  CHECK_EQ(data_table.global_clock, clock);
  iter_t data_age = data_table.global_clock;
  iter_t self_clock = data_table.vec_clock[client_id];
  /* Each tablet shard is single-threaded, so we don't need the lock */
  ScopedLock *branch_manager_lock_ptr = NULL;
  ClockSchedule clock_schedule = branch_manager->get_clock_schedule(
      branch_manager_lock_ptr, self_clock);
  iter_t self_internal_clock = clock_schedule.internal_clock;
  if (internal_data_age == -1) {
    self_internal_clock = -1;
  }
  communicator->read_row_batch_reply(
      client_id, process_id, data_age, self_clock,
      branch_id, internal_data_age, self_internal_clock,
      table_id, row_keys.data(), row_data, row_keys.size());
}

void TabletStorage::reset_perf_counters() {
  server_stats.reset();
}

void TabletStorage::worker_started(uint client_id) {
  worker_started_states[client_id] = 1;
  all_workers_started = clock_min(worker_started_states);
  if (all_workers_started) {
    communicator->server_started(process_id);
  }
}

void TabletStorage::clock(
    uint client_id, iter_t clock, uint table_id,
    int branch_id, iter_t internal_clock) {
  int timing = true;
  tbb::tick_count clock_ad_start;
  tbb::tick_count clock_ad_apply_op_end;
  tbb::tick_count clock_ad_end;

  if (timing) {
    clock_ad_start = tbb::tick_count::now();
  }

  /* Record the branch schedule */
  branch_manager->add_clock_schedule(clock, branch_id);

  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  CHECK_LT(client_id, data_table.vec_clock.size());
  if (data_table.vec_clock[client_id] != UNINITIALIZED_CLOCK) {
    CHECK_EQ(clock, data_table.vec_clock[client_id] + 1) << "table = " << table_id;
  }
  data_table.vec_clock[client_id] = clock;
  // if (process_id == 0 && channel_id == 0) {
    // cout << "Client " << client_id << " reaches " << clock << endl;
  // }

  iter_t new_global_clock = clock_min(data_table.vec_clock);
  if (new_global_clock != data_table.global_clock) {
    /* New clock */
    if (data_table.global_clock != UNINITIALIZED_CLOCK) {
      CHECK_EQ(new_global_clock, data_table.global_clock + 1);
    }
    data_table.global_clock = new_global_clock;

    if (config.update_func == "adarevision" ||
        config.update_func == "adarevision+momentum") {
      /* AdaRevision */
      /* Save the old accumulated gradient */
      uint branch_idx = branch_manager->get_branch_idx(branch_id);
      CHECK_LT(table_id, data_tables.size());
      DataTable& data_table = data_tables[table_id];
      ModelBranches& model_branches = data_table.model_branches;
      CHECK_LT(branch_idx, model_branches.size());
      ModelBranch& model_branch = model_branches[branch_idx];
      size_t batch_size = model_branch.store.size();
      CHECK_EQ(model_branch.history_store.size(), batch_size * 5);
      val_t *accum_gradients = reinterpret_cast<val_t *>(
          model_branch.history_store.data());
      val_t *old_accum_gradients = reinterpret_cast<val_t *>(
          &model_branch.history_store.data()[batch_size]);
      size_t memsize = batch_size * sizeof(RowData);
      memcpy(old_accum_gradients, accum_gradients, memsize);
    }

    /* Send pending read requests */
    process_all_client_subscribed_reads(
        data_table.global_clock, table_id, branch_id, internal_clock);
  }

  if (timing) {
    clock_ad_end = tbb::tick_count::now();
    server_stats.clock_ad_send_pending_time +=
      (clock_ad_end - clock_ad_apply_op_end).seconds();
    server_stats.clock_ad_time_tot +=
      (clock_ad_end - clock_ad_start).seconds();
  }
}

void TabletStorage::get_stats(
      uint client_id, shared_ptr<MetadataServer> metadata_server) {
  // server_stats.nr_rows = 0;
  // for (uint table_id = 0; table_id < data_tables.size(); table_id++) {
    // server_stats.nr_rows += data_tables[table_id].num_rows;
  // }

  std::stringstream combined_server_stats;
  combined_server_stats << "{"
         << "\"storage\": " << server_stats.to_json() << ", "
         << "\"metadata\": " << metadata_server->get_stats() << ", "
         << "\"router\": " << communicator->get_router_stats()
         << " } ";
  communicator->get_stats(client_id, combined_server_stats.str());
}

void TabletStorage::make_branch(
    int branch_id, Tunable& tunable, int flag,
    int parent_branch_id, iter_t clock_to_happen) {
  int optype = 0;
  BranchOpRefCountKey ref_count_key(optype, clock_to_happen);
  BranchInfo *branch_info = branch_manager->branch_info_map[branch_id];
  if (!branch_info) {
    BranchInfo *new_branch_info =
        new BranchInfo(branch_id, tunable, flag, parent_branch_id);
    branch_manager->branch_info_map[branch_id] = new_branch_info;
    branch_info = new_branch_info;
  } else {
    CHECK_EQ(branch_info->parent_branch_id, parent_branch_id);
  }
  branch_info->branch_op_ref_count_map[ref_count_key]++;

  if (branch_info->branch_op_ref_count_map[ref_count_key] < num_processes) {
    /* Each client will send a make_branch request to the server,
     * and the server only processes the request,
     * when it's received from each of the clients. */
    return;
  }

  uint ref_count = config.num_tables;
  branch_info->branch_idx = branch_manager->alloc_branch_idx(ref_count);
  CHECK_LT(branch_info->branch_idx, config.num_branches);

  /* Create a pending make branch request for every table */
  for (uint table_id = 0; table_id < config.num_tables; table_id++) {
    DataTable& data_table = data_tables[table_id];
    if (data_table.global_clock == UNINITIALIZED_CLOCK) {
      CHECK_EQ(clock_to_happen, INITIAL_CLOCK);
    } else {
      CHECK_EQ(clock_to_happen, data_table.global_clock + 1);
    }
    make_branch_for_table(
        branch_id, tunable, parent_branch_id, clock_to_happen, table_id);
  }
}

void TabletStorage::make_branch_for_table(
    int branch_id, Tunable& tunable,
    int parent_branch_id, iter_t clock_to_happen, uint table_id) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  BranchInfo *branch_info = branch_manager->branch_info_map[branch_id];
  CHECK(branch_info);
  uint branch_idx = branch_info->branch_idx;
  if (parent_branch_id < 0) {
    return;
  }
  uint parent_branch_idx = branch_manager->get_branch_idx(parent_branch_id);
  CHECK_LT(parent_branch_idx, model_branches.size());
  ModelBranch& parent_branch = model_branches[parent_branch_idx];
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& child_branch = model_branches[branch_idx];
  // CHECK(parent_branch.store.size());
  if (parent_branch.store.size()) {
    child_branch.store.init_from(parent_branch.store);
    child_branch.store.copy_data_cpu(parent_branch.store);
  }
  if (parent_branch.history_store.size()) {
    child_branch.history_store.init_from(parent_branch.history_store);
    child_branch.history_store.copy_data_cpu(parent_branch.history_store);
  }
  if (parent_branch.temp_store.size()) {
    child_branch.temp_store.init_from(parent_branch.temp_store);
    child_branch.temp_store.copy_data_cpu(parent_branch.temp_store);
  }

  /* Push param data to clients after creating new branch */
  iter_t global_clock = data_tables[table_id].global_clock;
  iter_t internal_clock = -1;
  process_all_client_subscribed_reads(
      global_clock, table_id, branch_id, internal_clock);
}

void TabletStorage::inactivate_branch(
    int branch_id, iter_t clock_to_happen) {
  BranchInfo *branch_info = branch_manager->branch_info_map[branch_id];
  CHECK(branch_info);
  int optype = 1;
  BranchOpRefCountKey ref_count_key(optype, clock_to_happen);
  branch_info->branch_op_ref_count_map[ref_count_key]++;
  
  if (branch_info->branch_op_ref_count_map[ref_count_key] < num_processes) {
    /* Each client will send an inactivate_branch request to the server,
     * and the server only processes the request,
     * when it's received from each of the clients. */
    return;
  }

  for (uint table_id = 0; table_id < config.num_tables; table_id++) {
    DataTable& data_table = data_tables[table_id];
    if (data_table.global_clock == UNINITIALIZED_CLOCK) {
      CHECK_EQ(clock_to_happen, INITIAL_CLOCK);
    } else {
      CHECK_EQ(clock_to_happen, data_table.global_clock + 1);
    }
    inactivate_branch_for_table(branch_id, clock_to_happen, table_id);
  }
}

void TabletStorage::inactivate_branch_for_table(
    int branch_id, iter_t clock_to_happen, uint table_id) {
  branch_manager->free_branch(branch_id);
}

void TabletStorage::save_branch(
    int branch_id, iter_t clock_to_happen) {
  BranchInfo *branch_info = branch_manager->branch_info_map[branch_id];
  CHECK(branch_info);
  int optype = 2;
  BranchOpRefCountKey ref_count_key(optype, clock_to_happen);
  branch_info->branch_op_ref_count_map[ref_count_key]++;
  
  if (branch_info->branch_op_ref_count_map[ref_count_key] < num_processes) {
    /* Each client will send an inactivate_branch request to the server,
     * and the server only processes the request,
     * when it's received from each of the clients. */
    return;
  }

  for (uint table_id = 0; table_id < config.num_tables; table_id++) {
    DataTable& data_table = data_tables[table_id];
    if (data_table.global_clock == UNINITIALIZED_CLOCK) {
      CHECK_EQ(clock_to_happen, INITIAL_CLOCK);
    } else {
      CHECK_EQ(clock_to_happen, data_table.global_clock + 1);
    }
    save_branch_for_table(branch_id, clock_to_happen, table_id);
  }
}

void TabletStorage::save_branch_for_table(
    int branch_id, iter_t clock_to_happen, uint table_id) {
  CHECK(0);
}
