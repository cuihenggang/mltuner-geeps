/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/bind.hpp>

#include <tbb/tick_count.h>

#include <glog/logging.h>

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "common/work-pusher.hpp"
#include "common/background-worker.hpp"
#include "common/row-op-util.hpp"
#include "encoder-decoder.hpp"
#include "client-tuner-encoder-decoder.hpp"
#include "clientlib.hpp"

/* Commands for the pusher threads */
#define ITERATE_CMD         1
#define WORKER_STARTED_CMD  2
#define FORWARD_MSG_CMD     3

#define OPSEQ_CMD           1

ClientLib *client_lib = NULL;

ClientLib::ClientLib(
        uint process_id_i, const GeePsConfig& config) : config(config){
  /* Takes in configs */
  host_list = config.host_list;
  num_channels = config.num_channels;
  tcp_base_port = config.tcp_base_port;
  num_servers = config.num_processes;
  CHECK_EQ(num_servers, config.host_list.size());
  process_id = process_id_i;

  proc_stats.prefetch = config.prefetch;
  proc_stats.pp_policy = config.pp_policy;
  proc_stats.local_opt = config.local_opt;
  proc_stats.affinity = config.affinity;

  /* Init fields */
  all_table_global_clocks.resize(config.num_tables);
  for (uint table_id = 0; table_id < config.num_tables; table_id++) {
    all_table_global_clocks[table_id] = UNINITIALIZED_CLOCK;
  }
  local_data_clock = UNINITIALIZED_CLOCK;
  virtual_iteration_all_finished = false;

  nr_threads = 0;

  /* Init branch manager */
  CHECK_EQ(config.num_threads, 1);
  /* We don't need the branch request queues */
  uint num_queues = 0;
  branch_manager = make_shared<BranchManager>(
      config.num_branches, num_queues);

  /* Init the tablet server and communication modules */
  CHECK_GT(num_channels, 0);
  comm_channels.resize(num_channels);
  for (uint i = 0; i < num_channels; i++) {
    init_comm_channel(i, config);
  }

  /* Init scheduler communication channel */
  init_tuner_comm_channel(config);

  start_time = tbb::tick_count::now();
}

void ClientLib::init_static_cache(
    StaticCache& static_cache, uint num_servers) {
  DataCacheOfModelBranches& data_cache_of_model_branches =
      static_cache.data_cache_of_model_branches;
  data_cache_of_model_branches.resize(config.num_branches);
  for (uint branch_idx = 0; branch_idx < config.num_branches; branch_idx++) {
    DataCacheOfModelBranch& data_cache_of_model_branch =
        data_cache_of_model_branches[branch_idx];
    data_cache_of_model_branch.per_server_data_age.resize(num_servers);
    data_cache_of_model_branch.per_server_internal_data_age.resize(num_servers);
    data_cache_of_model_branch.reset();
  }
}

void ClientLib::init_comm_channel(
        uint channel_id,
        const GeePsConfig& config) {
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  comm_channel.numa_node_id = 0;
  if (config.affinity) {
    comm_channel.numa_node_id =
      get_numa_node_id(channel_id, config.num_channels, config.num_zones);
  }
  comm_channel.mutex = make_shared<boost::mutex>();
  comm_channel.cvar = make_shared<boost::condition_variable>();

  /* Init cuda stream and cublas handle */
  cudaStream_t& cuda_stream_recv = comm_channel.cuda_stream_recv;
  cublasHandle_t& cublas_handle_recv = comm_channel.cublas_handle_recv;
  cudaStream_t& cuda_stream_send = comm_channel.cuda_stream_send;
  cublasHandle_t& cublas_handle_send = comm_channel.cublas_handle_send;
  CUDA_CHECK(cudaStreamCreate(&cuda_stream_recv));
  CUBLAS_CHECK(cublasCreate(&cublas_handle_recv));
  CUBLAS_CHECK(cublasSetStream(cublas_handle_recv, cuda_stream_recv));
  CUDA_CHECK(cudaStreamCreate(&cuda_stream_send));
  CUBLAS_CHECK(cublasCreate(&cublas_handle_send));
  CUBLAS_CHECK(cublasSetStream(cublas_handle_send, cuda_stream_send));

  comm_channel.zmq_ctx = make_shared<zmq::context_t>(1);

  /* Start server thread */
  ServerThreadEntry server_entry(
      channel_id, num_channels, process_id, num_servers,
      comm_channel.zmq_ctx, config);
  comm_channel.server_thread = make_shared<boost::thread>(server_entry);

  /* Init client communication */
  string client_name = (format("client-%i") % process_id).str();
  vector<string> bind_list;   /* Empty bind_list vector */
  vector<string> connect_list;
  for (uint i = 0; i < host_list.size(); i ++) {
    uint port =
      channel_id + ((port_list.size() != 0) ? port_list[i] : tcp_base_port);
    string connect_endpoint =
        "tcp://" + host_list[i] + ":" + boost::lexical_cast<std::string>(port);
    connect_list.push_back(connect_endpoint);
  }
  comm_channel.router = make_shared<RouterHandler>(
      channel_id, comm_channel.zmq_ctx, connect_list, bind_list,
      client_name, config);
  comm_channel.encoder = make_shared<ClientServerEncode>(
      comm_channel.router, host_list.size(), process_id, config);

  bool work_in_background = true;
  comm_channel.decoder = make_shared<ServerClientDecode>(
      channel_id, comm_channel.zmq_ctx,
      this, work_in_background, config);
  comm_channel.router->start_handler_thread(
      comm_channel.decoder->get_recv_callback());

  /* Start background worker thread */
  string endpoint = "inproc://bg-worker";
  shared_ptr<WorkPuller> work_puller =
      make_shared<WorkPuller>(comm_channel.zmq_ctx, endpoint);
  BackgroundWorker bg_worker(work_puller);
  BackgroundWorker::WorkerCallback worker_started_callback =
      bind(&ClientLib::cbk_worker_started, this, channel_id, _1);
  bg_worker.add_callback(WORKER_STARTED_CMD, worker_started_callback);
  BackgroundWorker::WorkerCallback iterate_callback =
      bind(&ClientLib::cbk_iterate, this, channel_id, _1);
  bg_worker.add_callback(ITERATE_CMD, iterate_callback);
  BackgroundWorker::WorkerCallback forward_msg_callback =
      bind(&ClientLib::cbk_forward_msg,
          this, channel_id, _1);
  bg_worker.add_callback(
      FORWARD_MSG_CMD, forward_msg_callback);
  comm_channel.bg_worker_thread = make_shared<boost::thread>(bg_worker);

  /* Init work pusher */
  comm_channel.work_pusher = make_shared<WorkPusher>(
                comm_channel.zmq_ctx, endpoint);

  /* Init tables */
  CHECK(config.num_tables);
  comm_channel.cached_tables.resize(config.num_tables);
  for (uint table_id = 0; table_id < config.num_tables; table_id++) {
    CachedTable& cached_table = comm_channel.cached_tables[table_id];
    cached_table.server_clock_min = UNINITIALIZED_CLOCK;
    cached_table.server_clock.resize(num_servers);
    // cached_table.per_server_branch_id.resize(num_servers);
    for (uint server_id = 0; server_id < num_servers; server_id++) {
      cached_table.server_clock[server_id] = UNINITIALIZED_CLOCK;
    }
    init_static_cache(cached_table.static_cache_cpu, num_servers);
    init_static_cache(cached_table.static_cache_gpu, num_servers);
  }

  /* Init other fields */
  comm_channel.server_started.resize(num_servers);
  for (uint i = 0; i < num_servers; i++) {
    comm_channel.server_started[i] = 0;
  }
  comm_channel.all_server_started = 0;
}

void ClientLib::init_tuner_comm_channel(
    const GeePsConfig& config) {
  tuner_comm_channel.zmq_ctx = make_shared<zmq::context_t>(1);

  if (process_id == 0) {
    /* Only process 0 runs the scheduler thread */
    MltunerEntry tuner_entry(
        config.num_processes, tuner_comm_channel.zmq_ctx, config);
    tuner_comm_channel.tuner_thread = make_shared<boost::thread>(tuner_entry);
  }

  string client_name = (format("client-%i") % process_id).str();
  vector<string> bind_list;   /* Empty bind_list vector */
  vector<string> connect_list;
  /* The scheduler is at host[0] */
  CHECK(host_list.size());
  uint port = tcp_base_port + config.num_channels;
  string connect_endpoint =
      "tcp://" + host_list[0] + ":" + boost::lexical_cast<std::string>(port);
  connect_list.push_back(connect_endpoint);
  uint channel_id = 0;    /* Just for message printing of router handler */
  uint numa_node_id = 0;
  tuner_comm_channel.router = make_shared<RouterHandler>(
      channel_id, tuner_comm_channel.zmq_ctx, connect_list, bind_list,
      client_name, config);
  tuner_comm_channel.encoder = make_shared<ClientTunerEncoder>(
      tuner_comm_channel.router, process_id, config);

  tuner_comm_channel.decoder = make_shared<TunerClientDecoder>(
      channel_id, tuner_comm_channel.zmq_ctx,
      this, true /* Work in background */,
      numa_node_id, config);
  tuner_comm_channel.router->start_handler_thread(
      tuner_comm_channel.decoder->get_recv_callback());

  /* The scheduler communication channel shares the same
   * background pusher thread as comm_channel[0] */
}

void ClientLib::thread_start() {
  ScopedLock global_clock_lock(global_clock_mutex);

  /* Set affinities */
  uint thread_id = nr_threads;
  uint numa_node_id = 0;
  if (config.affinity) {
    CHECK_EQ(thread_id, 0);
    numa_node_id =
        get_numa_node_id(thread_id, 1 /* num_threads */, config.num_zones);
    set_cpu_affinity(numa_node_id, config.num_cores, config.num_zones);
    set_mem_affinity(numa_node_id);
  }

  thread_data.reset(new ThreadData);
  ThreadData& thread_data_ref = *thread_data;

  /* Set up thread number */
  thread_data_ref.thread_id = thread_id;
  thread_data_ref.numa_node_id = numa_node_id;
  nr_threads++;
  proc_stats.nr_threads++;

  /* Init clocks */
  thread_data_ref.current_clock = INITIAL_CLOCK;

  /* Init cuda stream and cublas handle */
  cudaStreamCreate(&thread_data_ref.cuda_stream);
  cublasCreate(&thread_data_ref.cublas_handle);
  cublasSetStream(thread_data_ref.cublas_handle, thread_data_ref.cuda_stream);

  /* Init other fields */
  thread_data_ref.bg_worker_started = false;

  /* Signal the background worker to send a worker started message
   * to all servers */
  for (uint channel_id = 0; channel_id < comm_channels.size(); channel_id++) {
    push_worker_started_work(channel_id);
  }

  /* Send a worker started message to the scheduler */
  tuner_comm_channel.encoder->worker_started();
}

void ClientLib::push_worker_started_work(uint channel_id) {
  comm_channels[channel_id].work_pusher->push_work(WORKER_STARTED_CMD);
}

void ClientLib::shutdown() {
  // for (uint i = 0; i < num_channels; i++) {
    // zmq_term(*(comm_channels[i].zmq_ctx));
  // }
  // TODO(hengganc): join threads
  for (uint i = 0; i < num_channels; i++) {
    CommunicationChannel& comm_channel = comm_channels[i];
    /* Shut down background worker thread */
    comm_channel.work_pusher->push_work(BackgroundWorker::STOP_CMD);
    (*comm_channel.bg_worker_thread).join();

    /* Shut down router thread */
    comm_channel.router->stop_handler_thread();

    /* Shut down decoder thread */
    comm_channel.decoder->stop_decoder();
  }
}

void ClientLib::thread_stop() {
  // /* Clean up itself */
  // ThreadData& thread_data_ref = *thread_data;
  // (*thread_data_ref.bg_worker_thread).join();
  // CUDA_CHECK(cudaFreeHost(thread_data_ref.staging_mem));
  // thread_data.release();

  unique_lock<mutex> global_clock_lock(global_clock_mutex);
  nr_threads--;
}

uint64_t ClientLib::get_hash(table_id_t table, row_idx_t row) {
  return 0;
}

uint ClientLib::get_machine_id(table_id_t table, row_idx_t row) {
  return 0;
}

uint ClientLib::get_machine_id(uint64_t hash) {
  return 0;
}

uint ClientLib::get_channel_id(table_id_t table, row_idx_t row) {
  // return row % num_channels;
  uint table_id = table;
  return row / rows_per_channel[table_id];
}

uint ClientLib::get_channel_id(const TableRow& table_row) {
  // return table_row.second % num_channels;
  uint table_id = table_row.first;
  return table_row.second / rows_per_channel[table_id];
}

uint ClientLib::get_channel_id(uint64_t hash) {
  CHECK(0);
  return 0;
}

string ClientLib::json_stats() {
  BgthreadStats bgthread_stats;
  for (uint channel_id = 0; channel_id < num_channels; channel_id++) {
    bgthread_stats += comm_channels[channel_id].bgthread_stats;
  }
  bgthread_stats /= num_channels;
  // TODO(hengganc): separate router stats
  CommunicationChannel& comm_channel = comm_channels[0];
  proc_stats.router_stats = comm_channel.router->get_stats();
  proc_stats.bgthread_stats = bgthread_stats.to_json();

  /* Get server stat */
  unique_lock<mutex> global_clock_lock(global_clock_mutex);
  bool call_server = true;
  proc_stats.server_stats_refreshed = false;
  while (!proc_stats.server_stats_refreshed) {
    if (call_server) {
      call_server = false;
      global_clock_lock.unlock();  /* Release the lock while sending messages */
      comm_channel.encoder->get_stats(process_id);
      global_clock_lock.lock();
    } else {
      global_clock_cvar.wait(global_clock_lock);  /* Wait is notified by get_stats_cbk() */
    }
  }

  string json = proc_stats.to_json();

  std::string out_path = config.output_dir;
  out_path += "/json_stats.";
  out_path += boost::lexical_cast<std::string>(process_id);
  std::ofstream json_out(out_path.c_str(),
                         std::ofstream::out | std::ofstream::app);
  json_out << json << endl;
  json_out.close();

  return json;
}

void ClientLib::get_stats_cbk(const string& server_stats) {
  ScopedLock global_clock_lock(global_clock_mutex);
  proc_stats.server_stats = server_stats;
  proc_stats.server_stats_refreshed = true;
  global_clock_cvar.notify_all();
}

bool ClientLib::find_row_static_cache(
      table_id_t table, row_idx_t row, uint channel_id,
      bool unique_request, bool non_blocking, bool force_refresh) {
  CHECK(0);
  return true;
}

void ClientLib::find_row(table_id_t table, row_idx_t row,
                                   bool unique_request,
    /* unique_request == true: don't send redundant requests */
                                   bool non_blocking,
    /* non_blocking == true: don't wait for the reply */
                                   bool force_refresh
    /* force_refresh == true: send request even we know the location */
                                   ) {
  uint channel_id = get_channel_id(table, row);
  /* Look at static cache */
  if (find_row_static_cache(
      table, row, channel_id, unique_request, non_blocking, force_refresh)) {
    return;
  }
  CHECK(0);
}

bool ClientLib::find_row_cbk_static_cache(
      table_id_t table, row_idx_t row, uint32_t server_id,
      uint channel_id) {
  CHECK(0);
  return true;
}

void ClientLib::find_row_cbk(
      table_id_t table, row_idx_t row, uint32_t server_id) {
  uint channel_id = get_channel_id(table, row);
  /* Look at static cache */
  if (find_row_cbk_static_cache(table, row, server_id, channel_id)) {
    return;
  }
  CHECK(0);
}

void ClientLib::reset_perf_counters() {
  start_time = tbb::tick_count::now();
  proc_stats.reset();

  for (uint i = 0; i < comm_channels.size(); i++) {
    comm_channels[i].bgthread_stats.reset();
  }
}

void ClientLib::create_oplog_entry(
    uint channel_id, size_t oplog_idx, uint table_id, bool gpu) {
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  if (config.affinity) {
    set_mem_affinity(comm_channel.numa_node_id);
  }
  CHECK_LT(table_id, comm_channel.cached_tables.size());
  CachedTable& cached_table = comm_channel.cached_tables[table_id];
  StaticCache& static_cache = gpu ?
      cached_table.static_cache_gpu :
      cached_table.static_cache_cpu;
  StaticOpLog& oplog = static_cache.oplog;
  OpMemBufferPool& opmem_buffer_pool = static_cache.opmem_buffer_pool;
  CHECK_LT(oplog_idx, oplog.size());
  FlatOps *new_flatops = opmem_buffer_pool.get();
  // cout << "got " << new_flatops << " at " << clock << endl;
  CHECK(new_flatops);
  new_flatops->reset();
  oplog[oplog_idx] = new_flatops;
  /* Reset memory affinity */
  if (config.affinity) {
    set_mem_affinity(thread_data->numa_node_id);
  }
}

void ClientLib::clock_all(iter_t clock) {
  for (uint table_id = 0; table_id < config.num_tables; table_id++) {
    clock_table(clock, table_id);
  }

  /* Process the branch requests for the local table */
  ScopedLock global_clock_lock(global_clock_mutex);
  ScopedLock branch_manager_lock(branch_manager->mutex);
  if (local_data_clock != UNINITIALIZED_CLOCK) {
    CHECK_EQ(clock, local_data_clock + 1);
  }
  local_data_clock = clock;
  global_clock_cvar.notify_all();
}

void ClientLib::clock_table(iter_t clock, uint table_id) {
  ScopedLock global_clock_lock(global_clock_mutex);
  ScopedLock branch_manager_lock(branch_manager->mutex);

  // /* Update process stats */
  // proc_stats += thread_data_ref.thread_stats;
  // thread_data_ref.thread_stats = Stats();

  CHECK_LT(table_id, all_table_global_clocks.size());
  iter_t& global_clock = all_table_global_clocks[table_id];
  global_clock = clock;

  /* Signal the server that we have finished clock -1 */
  ClockSchedule clock_schedule =
      branch_manager->get_clock_schedule(
          &branch_manager_lock, clock);
  int branch_id = clock_schedule.branch_id;
  iter_t internal_clock = clock_schedule.internal_clock;
  /* Let the background communication thread send updates to tablet servers */
  for (uint channel_id = 0; channel_id < comm_channels.size(); channel_id++) {
    // push_updates_static_cache(channel_id, clock - 1, table_id);
    push_clock_work(
        channel_id, clock, table_id,
        branch_id, internal_clock);
  }
  global_clock_cvar.notify_all();
}

void ClientLib::push_clock_work(
    uint channel_id, iter_t clock, uint table_id,
    int branch_id, iter_t internal_clock) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(1);
  msgs[0].init_size(sizeof(bgcomm_clock_msg_t));
  bgcomm_clock_msg_t *clock_msg =
    reinterpret_cast<bgcomm_clock_msg_t *>(msgs[0].data());
  clock_msg->clock = clock;
  clock_msg->table_id = table_id;
  clock_msg->branch_id = branch_id;
  clock_msg->internal_clock = internal_clock;
  comm_channels[channel_id].work_pusher->push_work(ITERATE_CMD, msgs);
}

void ClientLib::push_updates_static_cache(
      uint channel_id, iter_t clock, uint table_id,
      int branch_id, iter_t internal_clock) {
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  BgthreadStats& bgthread_stats = comm_channel.bgthread_stats;
  tbb::tick_count push_updates_start = tbb::tick_count::now();

  /* TODO: re-consider whether we need to grab the channel lock here */
  CHECK_LT(table_id, comm_channel.cached_tables.size());
  CachedTable& cached_table = comm_channel.cached_tables[table_id];

  StaticCache& static_cache_cpu = cached_table.static_cache_cpu;
  StaticOpLog& static_oplog_cpu = static_cache_cpu.oplog;
  StaticCache& static_cache_gpu = cached_table.static_cache_gpu;
  StaticOpLog& static_oplog_gpu = static_cache_gpu.oplog;
  bool cpu_op = static_oplog_cpu.size() != 0 &&
      static_oplog_cpu[static_cast<size_t>(clock) % static_oplog_cpu.size()]
          != NULL;
  bool gpu_op = static_oplog_gpu.size() != 0 &&
      static_oplog_gpu[static_cast<size_t>(clock) % static_oplog_gpu.size()]
          != NULL;
  if (!cpu_op && !gpu_op) {
    comm_channel.encoder->clock_broadcast(
        clock, table_id, branch_id, internal_clock);
    return;
  }

  const RowKey *row_keys_cpu = NULL;
  const RowOpVal *updates_cpu = NULL;
  if (cpu_op) {
    size_t oplog_idx = static_cast<size_t>(clock) % static_oplog_cpu.size();
    FlatOps& flat_ops_cpu = *static_oplog_cpu[oplog_idx];
    CHECK_EQ(flat_ops_cpu.branch_id, branch_id);
    row_keys_cpu = static_cache_cpu.row_keys.data();
    updates_cpu = flat_ops_cpu.data();
    CHECK_EQ(static_cache_cpu.per_server_row_start.size(), num_servers);
    CHECK_EQ(static_cache_cpu.per_server_num_rows.size(), num_servers);
  }

  const RowKey *row_keys_gpu = NULL;
  RowOpVal *updates_gpu = NULL;
  if (gpu_op) {
    /* Copy GPU updates to host memory buffer */
    // tbb::tick_count copy_to_buffer_start = tbb::tick_count::now();
    size_t oplog_idx = static_cast<size_t>(clock) % static_oplog_gpu.size();
    FlatOps& flat_ops_gpu = *static_oplog_gpu[oplog_idx];
    CHECK_EQ(flat_ops_gpu.branch_id, branch_id);
    row_keys_gpu = static_cache_gpu.row_keys.data();
    updates_gpu = comm_channel.send_buffer;
    CUDA_CHECK(cudaMemcpyAsync(updates_gpu, flat_ops_gpu.data(),
        flat_ops_gpu.memsize(), cudaMemcpyDefault,
        comm_channel.cuda_stream_send));
    CUDA_CHECK(cudaStreamSynchronize(comm_channel.cuda_stream_send));
    // const RowOpVal *updates = flat_ops.data();
    // bgthread_stats.push_updates_find_row_time +=
      // (tbb::tick_count::now() - copy_to_buffer_start).seconds();
    CHECK_EQ(static_cache_gpu.per_server_row_start.size(), num_servers);
    CHECK_EQ(static_cache_gpu.per_server_num_rows.size(), num_servers);
    if (!config.read_my_writes) {
      /* If we don't do read-my-updates, we can reclaim the oplogs here */
      reclaim_oplog(channel_id, clock, clock, table_id, true /* the GPU one */);
    }
  }

  /* Send to each server */
  for (uint server_id = 0; server_id < num_servers; server_id++) {
    const RowOpVal *updates_to_send_cpu = NULL;
    const RowKey *row_keys_to_send_cpu = NULL;
    uint num_rows_cpu = 0;
    const RowOpVal *updates_to_send_gpu = NULL;
    const RowKey *row_keys_to_send_gpu = NULL;
    uint num_rows_gpu = 0;
    if (cpu_op) {
      uint row_start_cpu = static_cache_cpu.per_server_row_start[server_id];
      updates_to_send_cpu = &updates_cpu[row_start_cpu];
      row_keys_to_send_cpu = &row_keys_cpu[row_start_cpu];
      num_rows_cpu = static_cache_cpu.per_server_num_rows[server_id];
    }
    if (gpu_op) {
      uint row_start_gpu = static_cache_gpu.per_server_row_start[server_id];
      updates_to_send_gpu = &updates_gpu[row_start_gpu];
      row_keys_to_send_gpu = &row_keys_gpu[row_start_gpu];
      num_rows_gpu = static_cache_gpu.per_server_num_rows[server_id];
    }
    comm_channel.encoder->clock_with_updates_batch(
        server_id, clock, table_id, branch_id, internal_clock,
        updates_to_send_cpu, row_keys_to_send_cpu, num_rows_cpu,
        updates_to_send_gpu, row_keys_to_send_gpu, num_rows_gpu);
  }

  if (cpu_op) {
    if (!config.read_my_writes) {
      /* If we don't do read-my-updates, we can reclaim the oplogs here */
      bool gpu = false;   /* the CPU one */
      reclaim_oplog(channel_id, clock, clock, table_id, gpu);
    }
  }

  /* TODO: I think we should pack GPU updates and CPU updates in the same message */
  bgthread_stats.tot_push_updates_time +=
    (tbb::tick_count::now() - push_updates_start).seconds();
}

void ClientLib::push_updates(
    uint channel_id, iter_t clock, uint table_id,
    int branch_id, iter_t internal_clock) {
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  /* There's only one push_updates() thread for each channel,
   * so we don't need to grab the channel lock */
  BgthreadStats& bgthread_stats = comm_channel.bgthread_stats;
  push_updates_static_cache(
      channel_id, clock, table_id, branch_id, internal_clock);
  bgthread_stats.push_updates_iter = clock - 1;
}

void ClientLib::cbk_iterate(
        uint channel_id, vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 1);
  bgcomm_clock_msg_t *clock_msg =
      reinterpret_cast<bgcomm_clock_msg_t *>(args[0].data());
  push_updates(
      channel_id, clock_msg->clock, clock_msg->table_id,
      clock_msg->branch_id, clock_msg->internal_clock);
  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ClientLib::worker_started(uint channel_id) {
  /* Wait for the server to reply the worker started message */
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  unique_lock<mutex> channel_lock(*comm_channel.mutex);
  comm_channel.encoder->worker_started();
  while (!comm_channel.all_server_started) {
    if (!comm_channel.cvar->timed_wait(channel_lock,
        boost::posix_time::milliseconds(2000))) {
      cerr << "*** worker " << process_id
           << " of channel " << channel_id
           << " waiting for server start timed out\n";
      /* Resend the worker started message */
      comm_channel.encoder->worker_started();
      cout << "*** client resends worker started to channel " << channel_id << endl;
    }
  }
}

void ClientLib::cbk_worker_started(
        uint channel_id, vector<ZmqPortableBytes>& args) {
  worker_started(channel_id);
  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

bool ClientLib::recv_row_batch_static_cache(
      uint channel_id, uint server_id, iter_t data_age, iter_t self_clock,
      int branch_id, uint branch_idx,
      iter_t internal_data_age, iter_t internal_self_clock,
      uint table_id, RowKey *row_keys, RowData *row_data, uint batch_size) {
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  // BgthreadStats& bgthread_stats = comm_channel.bgthread_stats;

  CHECK_LT(table_id, comm_channel.cached_tables.size());
  CachedTable& cached_table = comm_channel.cached_tables[table_id];
  StaticCache& static_cache_cpu = cached_table.static_cache_cpu;
  StaticCache& static_cache_gpu = cached_table.static_cache_gpu;
  uint cache_size_cpu = static_cache_cpu.per_server_num_rows[server_id];
  uint cache_size_gpu = static_cache_gpu.per_server_num_rows[server_id];
  CHECK_EQ(cache_size_cpu + cache_size_gpu, batch_size);

  /* Copy to pinned memory first */
  /* GPU data is stored after CPU data */
  RowData *row_data_gpu_to_be_copied = &row_data[cache_size_cpu];
  size_t size = cache_size_gpu * sizeof(RowData);
  if (size) {
    /* TODO: recv_buffer size is it enough? */
    memcpy(comm_channel.recv_buffer, row_data_gpu_to_be_copied, size);
  }

  unique_lock<mutex> channel_lock(*comm_channel.mutex);

  RowKey *row_keys_cpu = row_keys;
  RowData *row_data_cpu = row_data;
  RowKey *row_keys_gpu = &row_keys[cache_size_cpu];
  RowData *row_data_gpu = comm_channel.recv_buffer;
  recv_row_batch_cpu(channel_id, server_id, data_age, self_clock,
      branch_id, branch_idx, internal_data_age, internal_self_clock,
      table_id, row_keys_cpu, row_data_cpu, cache_size_cpu);
  recv_row_batch_gpu(channel_id, server_id, data_age, self_clock,
      branch_id, branch_idx, internal_data_age, internal_self_clock,
      table_id, row_keys_gpu, row_data_gpu, cache_size_gpu);

  /* FIXME: move the mutex and cvar to table level */
  comm_channel.cvar->notify_all();

  // bgthread_stats.recv_row_apply_oplog_time +=
    // (tbb::tick_count::now() - apply_oplog_start).seconds();

  return true;
}

void ClientLib::recv_row_batch_gpu(
    uint channel_id, uint server_id, iter_t data_age, iter_t self_clock,
    int branch_id, uint branch_idx,
    iter_t internal_data_age, iter_t internal_self_clock,
    uint table_id, RowKey *row_keys, RowData *row_data, uint batch_size) {
  CHECK_EQ(batch_size, 0);
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  CHECK_LT(table_id, comm_channel.cached_tables.size());
  CachedTable& cached_table = comm_channel.cached_tables[table_id];
  StaticCache& static_cache = cached_table.static_cache_gpu;
  CHECK_LT(server_id, static_cache.per_server_row_start.size());
  CHECK_LT(server_id, static_cache.per_server_num_rows.size());
  uint row_start = static_cache.per_server_row_start[server_id];
  uint num_rows = static_cache.per_server_num_rows[server_id];
  CHECK_EQ(num_rows, batch_size);
  DataCacheOfModelBranches& data_cache_of_model_branches =
      static_cache.data_cache_of_model_branches;
  CHECK_LT(branch_idx, data_cache_of_model_branches.size());
  DataCacheOfModelBranch& data_cache_of_model_branch =
      data_cache_of_model_branches[branch_idx];
  if (data_cache_of_model_branch.branch_id != branch_id) {
    /* Received data of an inactivated branch */
    return;
  }

  CHECK_LT(server_id, data_cache_of_model_branch.per_server_data_age.size());
  iter_t& server_data_age =
      data_cache_of_model_branch.per_server_data_age[server_id];
  iter_t& server_internal_data_age =
      data_cache_of_model_branch.per_server_internal_data_age[server_id];
  CHECK_GT(data_age, server_data_age);
  CHECK_LE(data_age, self_clock);
  server_data_age = data_age;
  server_internal_data_age = internal_data_age;
  data_cache_of_model_branch.data_age =
      clock_min(data_cache_of_model_branch.per_server_data_age);
  data_cache_of_model_branch.internal_data_age =
      clock_min(data_cache_of_model_branch.per_server_internal_data_age);

  DataCache& data_cache = data_cache_of_model_branch.data_cache;
  RowData *cached_data = &data_cache.data()[row_start];
  cudaStream_t& cuda_stream = comm_channel.cuda_stream_recv;
  // cublasHandle_t& cublas_handle = comm_channel.cublas_handle_recv;

  // tbb::tick_count memcpy_start = tbb::tick_count::now();
  CUDA_CHECK(cudaMemcpyAsync(cached_data, row_data,
      batch_size * sizeof(RowData), cudaMemcpyDefault, cuda_stream));
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  // bgthread_stats.recv_row_copy_data_time +=
      // (tbb::tick_count::now() - memcpy_start).seconds();

  // if (config.read_my_writes) {
    // /* Apply oplogs */
    // CHECK(0);
    // StaticOpLog& oplog = static_cache.oplog;
    // int64_t oplog_count = 0;
    // CHECK_LE(data_age, self_clock);
    // for (iter_t clock = self_clock + 1; clock <= fast_clock; clock++) {
      // if (oplog[clock] == NULL) {
        // continue;
      // }
      // FlatOps& flat_ops = *oplog[clock];
      // if (flat_ops.flag == FlatOps::INC) {
        // CHECK_EQ(flat_ops.size(), data_cache.size());
        // oplog_count++;
        // const RowOpVal *updates = &flat_ops.data()[row_start];
        // add_row_batch_gpu(cublas_handle, cached_data, updates, num_rows);
      // }
    // }
    // CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  // }
}

void ClientLib::recv_row_batch_cpu(
    uint channel_id, uint server_id, iter_t data_age, iter_t self_clock,
    int branch_id, uint branch_idx,
    iter_t internal_data_age, iter_t internal_self_clock,
    uint table_id, RowKey *row_keys, RowData *row_data, uint batch_size) {
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  CHECK_LT(table_id, comm_channel.cached_tables.size());
  CachedTable& cached_table = comm_channel.cached_tables[table_id];
  StaticCache& static_cache = cached_table.static_cache_cpu;
  CHECK_LT(server_id, static_cache.per_server_row_start.size());
  CHECK_LT(server_id, static_cache.per_server_num_rows.size());
  uint row_start = static_cache.per_server_row_start[server_id];
  uint num_rows = static_cache.per_server_num_rows[server_id];
  CHECK_EQ(num_rows, batch_size);
  DataCacheOfModelBranches& data_cache_of_model_branches =
      static_cache.data_cache_of_model_branches;
  CHECK_LT(branch_idx, data_cache_of_model_branches.size());
  DataCacheOfModelBranch& data_cache_of_model_branch =
      data_cache_of_model_branches[branch_idx];
  if (data_cache_of_model_branch.branch_id != branch_id) {
    /* Received data of an inactivated branch */
    return;
  }

  CHECK_LT(server_id, data_cache_of_model_branch.per_server_data_age.size());
  iter_t& server_data_age =
      data_cache_of_model_branch.per_server_data_age[server_id];
  iter_t& server_internal_data_age =
      data_cache_of_model_branch.per_server_internal_data_age[server_id];
  CHECK_GT(data_age, server_data_age);
  CHECK_LE(data_age, self_clock);
  server_data_age = data_age;
  server_internal_data_age = internal_data_age;
  data_cache_of_model_branch.data_age =
      clock_min(data_cache_of_model_branch.per_server_data_age);
  data_cache_of_model_branch.internal_data_age =
      clock_min(data_cache_of_model_branch.per_server_internal_data_age);

  DataCache& data_cache = data_cache_of_model_branch.data_cache;
  RowData *cached_data = &data_cache.data()[row_start];

  // tbb::tick_count memcpy_start = tbb::tick_count::now();
  memcpy(cached_data, row_data, batch_size * sizeof(RowData));
  // bgthread_stats.recv_row_copy_data_time +=
      // (tbb::tick_count::now() - memcpy_start).seconds();

  // if (config.read_my_writes) {
    // /* Apply oplogs */
    // CHECK(0);
    // StaticOpLog& oplog = static_cache.oplog;
    // int64_t oplog_count = 0;
    // CHECK_LE(data_age, self_clock);
    // for (iter_t clock = self_clock + 1; clock <= fast_clock; clock++) {
      // if (oplog[clock] == NULL) {
        // continue;
      // }
      // FlatOps& flat_ops = *oplog[clock];
      // if (flat_ops.flag == FlatOps::INC) {
        // CHECK_EQ(flat_ops.size(), data_cache.size());
        // oplog_count++;
        // const RowOpVal *updates = &flat_ops.data()[row_start];
        // add_row_batch(cached_data, updates, num_rows);
      // }
    // }
  // }
}

void ClientLib::recv_row_batch(
      uint channel_id, uint server_id, iter_t data_age, iter_t self_clock,
      int branch_id, iter_t internal_data_age, iter_t internal_self_clock,
      uint table_id, RowKey *row_keys, RowData *row_data, uint batch_size) {
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  BgthreadStats& bgthread_stats = comm_channel.bgthread_stats;
  tbb::tick_count recv_row_start;
  tbb::tick_count apply_op_start;
  tbb::tick_count recv_row_end;
  recv_row_start = tbb::tick_count::now();

  // cout << "Client recv row batch for branch " << branch_id << endl;

  uint branch_idx;
  {
    ScopedLock branch_manager_lock(branch_manager->mutex);
    branch_idx = branch_manager->get_branch_idx(branch_id);
  }
  if (branch_idx < config.num_branches) {
    recv_row_batch_static_cache(
        channel_id, server_id, data_age, self_clock,
        branch_id, branch_idx, internal_data_age, internal_self_clock,
        table_id, row_keys, row_data, batch_size);
  } else {
    /* Received data of an inactivated branch.
     * We decide to ignore it for now, and if we want to restore
     * an inactivated branch, we can read it again from the server */
    // cerr << "WARNING: received row of inactivated branch " << branch_id << endl;
  }

  recv_row_end = tbb::tick_count::now();
  bgthread_stats.tot_recv_row_time +=
    (recv_row_end - recv_row_start).seconds();
}

int ClientLib::read_row_batch_static_cache(
    RowData *buffer, OpInfo& op_info, int branch_id, uint branch_idx,
    iter_t required_internal_data_age,
    cudaStream_t& cuda_stream, RowData *staging_mem) {
  // tbb::tick_count wait_miss_start;
  // tbb::tick_count cache_copy_start;

  CHECK_EQ(op_info.type, OpInfo::READ);

  for (uint channel_id = 0; channel_id < num_channels; channel_id++) {
    // if (timing) {
      // wait_miss_start = tbb::tick_count::now();
    // }

    tbb::tick_count time_start = tbb::tick_count::now();
    CHECK(staging_mem);
    read_batch_cpu(
        staging_mem, op_info, channel_id,
        branch_id, branch_idx, required_internal_data_age);
    proc_stats.bg_read_read_time +=
        (tbb::tick_count::now() - time_start).seconds();
    // if (timing) {
      // thread_data_ref.thread_stats.read_make_cache_time +=
        // (tbb::tick_count::now() - cache_copy_start).seconds();
    // }

    // if (stat) {
      // if (clock > 0) {
        // thread_data_ref.thread_stats.tot_staleness +=
          // (clock - comm_channel.data_age);
      // }
    // }
  }

  tbb::tick_count time_start = tbb::tick_count::now();
  /* Copy the data from CPU buffer to GPU buffer */
  /* TODO: I don't think we need two copies here */
  size_t size;
  if (op_info.num_vals_limit >= 0) {
    size = op_info.num_vals_limit * sizeof(val_t);
  } else {
    size = op_info.rows.size() * sizeof(RowData);
  }
  CUDA_CHECK(cudaMemcpyAsync(buffer, staging_mem, size,
      cudaMemcpyDefault, cuda_stream));
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  proc_stats.bg_read_move_time +=
      (tbb::tick_count::now() - time_start).seconds();

  return 1;
}

void ClientLib::read_batch_gpu(
    RowData *buffer, OpInfo& op_info, uint channel_id,
    int branch_id, uint branch_idx, iter_t required_internal_data_age,
    cudaStream_t& cuda_stream) {
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  ScopedLock channel_lock(*comm_channel.mutex);
  uint table_id = op_info.table_id;
  CHECK_LT(table_id, comm_channel.cached_tables.size());
  CachedTable& cached_table = comm_channel.cached_tables[table_id];
  StaticCache& static_cache = cached_table.static_cache_gpu;
  DataCacheOfModelBranches& data_cache_of_model_branches =
      static_cache.data_cache_of_model_branches;
  CHECK_LT(branch_idx, data_cache_of_model_branches.size());
  DataCacheOfModelBranch& data_cache_of_model_branch =
      data_cache_of_model_branches[branch_idx];

  while (data_cache_of_model_branch.internal_data_age
      < required_internal_data_age) {
    // comm_channel.cvar->wait(channel_lock);
    if (!comm_channel.cvar->timed_wait(channel_lock,
          boost::posix_time::milliseconds(12000))) {
      if (channel_id == 0) {
        /* Read timeout */
        cerr << "machine " << process_id
              << " wait time out!" << endl;
        cerr << "Branch: " << branch_id
             << " Need: " << required_internal_data_age
             << " Data age: " << data_cache_of_model_branch.internal_data_age
             << std::endl;
         /* Read time out */
      }
    }
  }

  DataCache& data_cache = data_cache_of_model_branch.data_cache;
  CHECK(op_info.row_index_gpu);
  const DoubleIndex *row_index = op_info.row_index_gpu;
  PerChannelIndexInfo& index_info = op_info.index_infos[channel_id];
  size_t row_index_start = index_info.index_start;
  size_t row_index_size = index_info.index_size;
  size_t num_rows = static_cache.num_rows;
  CHECK_EQ(num_rows, data_cache.size());
  const DoubleIndex *row_channel_index = &row_index[row_index_start];
  DoubleIndex index_offset(0, 0) /* No offset needed */;
  assign_rows_to_double_index_gpu(
      buffer, data_cache.data(), row_channel_index, row_index_size,
      index_offset, ROW_DATA_SIZE, op_info.num_vals_limit, cuda_stream);
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
}

void ClientLib::read_batch_cpu(
    RowData *buffer, OpInfo& op_info, uint channel_id,
    int branch_id, uint branch_idx, iter_t required_internal_data_age) {
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  ScopedLock channel_lock(*comm_channel.mutex);
  uint table_id = op_info.table_id;
  CHECK_LT(table_id, comm_channel.cached_tables.size());
  CachedTable& cached_table = comm_channel.cached_tables[table_id];
  StaticCache& static_cache = cached_table.static_cache_cpu;
  DataCacheOfModelBranches& data_cache_of_model_branches =
      static_cache.data_cache_of_model_branches;
  CHECK_LT(branch_idx, data_cache_of_model_branches.size());
  DataCacheOfModelBranch& data_cache_of_model_branch =
      data_cache_of_model_branches[branch_idx];

  while (data_cache_of_model_branch.internal_data_age
      < required_internal_data_age) {
    // comm_channel.cvar->wait(channel_lock);
    if (!comm_channel.cvar->timed_wait(channel_lock,
          boost::posix_time::milliseconds(12000))) {
      if (channel_id == 0) {
        /* Read timeout */
        cerr << "machine " << process_id
              << " wait time out!" << endl;
        cerr << "Branch: " << branch_id
             << " Need: " << required_internal_data_age
             << " Data age: " << data_cache_of_model_branch.internal_data_age
             << std::endl;
         /* Read time out */
      }
    }
  }

  DataCache& data_cache = data_cache_of_model_branch.data_cache;
  CHECK(op_info.row_index_cpu);
  const DoubleIndex *row_index = op_info.row_index_cpu;
  PerChannelIndexInfo& index_info = op_info.index_infos[channel_id];
  size_t row_index_start = index_info.index_start;
  size_t row_index_size = index_info.index_size;
  size_t num_rows = static_cache.num_rows;
  CHECK_EQ(num_rows, data_cache.size());
  const DoubleIndex *row_channel_index = &row_index[row_index_start];
  DoubleIndex index_offset(0, 0) /* No offset needed */;
  assign_rows_to_double_index_cpu(
      buffer, data_cache.data(), row_channel_index, row_index_size,
      index_offset, ROW_DATA_SIZE, op_info.num_vals_limit);
}

bool ClientLib::update_batch_static_cache(
    OpInfo& op_info, int branch_id, const RowOpVal *updates, iter_t clock,
    cudaStream_t& cuda_stream, RowData *staging_mem) {
  // tbb::tick_count apply_proc_cache_start;
  // tbb::tick_count get_lock_end;
  // tbb::tick_count apply_proc_cache_end;

  CHECK_EQ(op_info.type, OpInfo::PRE_UPDATE);

  /* Copy the updates from GPU buffer to CPU buffer */
  /* TODO: I don't think we need two copies here */
  tbb::tick_count time_start = tbb::tick_count::now();
  CHECK(staging_mem);
  size_t size;
  if (op_info.num_vals_limit >= 0) {
    size = op_info.num_vals_limit * sizeof(val_t);
  } else {
    size = op_info.rows.size() * sizeof(RowData);
  }
  // cerr << "size = " << size << endl;
  CUDA_CHECK(cudaMemcpyAsync(staging_mem, updates, size,
      cudaMemcpyDefault, cuda_stream));
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  proc_stats.bg_update_move_time +=
      (tbb::tick_count::now() - time_start).seconds();

  time_start = tbb::tick_count::now();
  for (uint channel_id = 0; channel_id < num_channels; channel_id++) {
    // if (timing) {
      // apply_proc_cache_start = tbb::tick_count::now();
    // }

    CommunicationChannel& comm_channel = comm_channels[channel_id];
    unique_lock<mutex> channel_lock(*comm_channel.mutex);

    // if (timing) {
      // get_lock_end = tbb::tick_count::now();
      // thread_data_ref.thread_stats.update_get_lock_time +=
        // (get_lock_end - apply_proc_cache_start).seconds();
    // }

    update_batch_cpu(op_info, staging_mem, channel_id, clock, branch_id);

    // if (stat) {
      // thread_data_ref.thread_stats.tot_apply_proc_cache++;
    // }
    // if (timing) {
      // apply_proc_cache_end = tbb::tick_count::now();
      // thread_data_ref.thread_stats.update_apply_proc_cache_time +=
        // (apply_proc_cache_end - get_lock_end).seconds();
    // }
  }
  proc_stats.bg_update_update_time +=
      (tbb::tick_count::now() - time_start).seconds();

  return true;
}

void ClientLib::update_batch_gpu(
    OpInfo& op_info, const RowOpVal *updates, uint channel_id, iter_t clock,
    int branch_id, cudaStream_t& cuda_stream) {
  /* Channel lock is held while calling this function */
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  uint table_id = op_info.table_id;
  CHECK_LT(table_id, comm_channel.cached_tables.size());
  CachedTable& cached_table = comm_channel.cached_tables[table_id];
  StaticCache& static_cache = cached_table.static_cache_gpu;
  DataCacheOfModelBranches& data_cache_of_model_branches =
      static_cache.data_cache_of_model_branches;
  uint branch_idx = get_branch_idx(branch_id);
  CHECK_LT(branch_idx, data_cache_of_model_branches.size());
  DataCache& data_cache = data_cache_of_model_branches[branch_idx].data_cache;
  size_t oplog_idx = static_cast<size_t>(clock) % static_cache.oplog.size();
  CHECK_LT(oplog_idx, static_cache.oplog.size());
  if (static_cache.oplog[oplog_idx] == NULL) {
    create_oplog_entry(channel_id, oplog_idx, table_id, true);
    FlatOps *new_flatops = static_cache.oplog[oplog_idx];
    CHECK(new_flatops) << "clock = " << clock << ", oplog_idx = " << oplog_idx;
    /* The DataStorage class zerofies data using cudaMemsetAsync(),
     * which takes in a cudaStream_t argument.
     * a cudaStreamSynchronize() will be called on this stream.
     * There was a bug about this that I spent almost two weeks debugging.
     * Previously, I was using cudaMemset(),
     * and I saw "unspecified launch failure" in weird places.
     * Finally I realized that cudaMemset() is actually *asynchronous*,
     * in that it returns control to host code before finish.
     * So we actually need a cudaDeviceSynchronize() after it.
     * But any way, we should use cudaMemsetAsync() here. */
    new_flatops->zerofy_data_gpu(cuda_stream);
    new_flatops->branch_id = branch_id;
    new_flatops->clock = clock;
  }
  FlatOps& flat_ops = *static_cache.oplog[oplog_idx];
  CHECK_EQ(flat_ops.branch_id, branch_id);
  CHECK_EQ(flat_ops.clock, clock);
  CHECK(op_info.row_index_gpu);
  const DoubleIndex *row_index = op_info.row_index_gpu;
  PerChannelIndexInfo& index_info = op_info.index_infos[channel_id];
  size_t row_index_start = index_info.index_start;
  size_t row_index_size = index_info.index_size;
  size_t num_rows = static_cache.num_rows;
  CHECK_EQ(num_rows, data_cache.size());
  CHECK_EQ(num_rows, flat_ops.size());
  flat_ops.flag = FlatOps::INC;
  const DoubleIndex *row_channel_index = &row_index[row_index_start];
  DoubleIndex index_offset(0, 0) /* No offset needed */;
  add_rows_from_double_index_gpu(
      flat_ops.data(), updates, row_channel_index, row_index_size,
      index_offset, ROW_DATA_SIZE, op_info.num_vals_limit,
      cuda_stream);
  if (config.read_my_writes) {
    add_rows_from_double_index_gpu(
        data_cache.data(), updates, row_channel_index, row_index_size,
        index_offset, ROW_DATA_SIZE, op_info.num_vals_limit,
        cuda_stream);
  }
  // const float *ops_data = reinterpret_cast<const float *>(flat_ops.data());
  // float ops_dot = caffe::caffe_cpu_dot<float>(flat_ops.size() * ROW_DATA_SIZE, ops_data, ops_data);
  // cerr << "gpu_ops_dot = " << ops_dot << endl;
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
}

void ClientLib::update_batch_cpu(
    OpInfo& op_info, const RowOpVal *updates, uint channel_id, iter_t clock,
    int branch_id) {
  /* Channel lock is held while calling this function */
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  uint table_id = op_info.table_id;
  CHECK_LT(table_id, comm_channel.cached_tables.size());
  CachedTable& cached_table = comm_channel.cached_tables[table_id];
  StaticCache& static_cache = cached_table.static_cache_cpu;
  DataCacheOfModelBranches& data_cache_of_model_branches =
      static_cache.data_cache_of_model_branches;
  uint branch_idx = get_branch_idx(branch_id);
  CHECK_LT(branch_idx, data_cache_of_model_branches.size());
  DataCache& data_cache = data_cache_of_model_branches[branch_idx].data_cache;
  size_t oplog_idx = static_cast<size_t>(clock) % static_cache.oplog.size();
  CHECK_LT(oplog_idx, static_cache.oplog.size());
  if (static_cache.oplog[oplog_idx] == NULL) {
    create_oplog_entry(channel_id, oplog_idx, table_id, false);
    FlatOps *new_flatops = static_cache.oplog[oplog_idx];
    CHECK(new_flatops) << "clock = " << clock << ", oplog_idx = " << oplog_idx;
    new_flatops->zerofy_data_cpu();
    new_flatops->branch_id = branch_id;
    new_flatops->clock = clock;
  }
  FlatOps& flat_ops = *static_cache.oplog[oplog_idx];
  CHECK_EQ(flat_ops.branch_id, branch_id);
  CHECK_EQ(flat_ops.clock, clock);
  CHECK(op_info.row_index_cpu);
  const DoubleIndex *row_index = op_info.row_index_cpu;
  PerChannelIndexInfo& index_info = op_info.index_infos[channel_id];
  size_t row_index_start = index_info.index_start;
  size_t row_index_size = index_info.index_size;
  size_t num_rows = static_cache.num_rows;
  CHECK_EQ(num_rows, data_cache.size());
  CHECK_EQ(num_rows, flat_ops.size());
  flat_ops.flag = FlatOps::INC;
  CHECK_EQ(flat_ops.branch_id, branch_id);
  const DoubleIndex *row_channel_index = &row_index[row_index_start];
  DoubleIndex index_offset(0, 0) /* No offset needed */;
  add_rows_from_double_index_cpu(
      flat_ops.data(), updates, row_channel_index, row_index_size,
      index_offset, ROW_DATA_SIZE, op_info.num_vals_limit);
  if (config.read_my_writes) {
    add_rows_from_double_index_cpu(
        data_cache.data(), updates, row_channel_index, row_index_size,
        index_offset, ROW_DATA_SIZE, op_info.num_vals_limit);
    // const float *ops_data = reinterpret_cast<const float *>(flat_ops.data());
    // float ops_dot = caffe::caffe_cpu_dot<float>(flat_ops.size() * ROW_DATA_SIZE, ops_data, ops_data);
    // cerr << "cpu_ops_dot = " << ops_dot << endl;
  }
}

void ClientLib::server_clock(
        uint channel_id, uint server_id, iter_t clock, uint table_id) {
  CHECK_GE(clock, 0);
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  /* TODO: do we need to grab a lock here? */
  CHECK_LT(table_id, comm_channel.cached_tables.size());
  CachedTable& cached_table = comm_channel.cached_tables[table_id];
  // if (comm_channel.server_clock.count(server_id)) {
    // if (clock <= comm_channel.server_clock[server_id]) {
      // cerr << "WARNING: shared iteration went backwards "
           // << comm_channel.server_clock[server_id] << " to " << clock
           // << endl;
    // } else if (clock > comm_channel.server_clock[server_id] + 1) {
      // cerr << "WARNING: shared iteration jumped "
           // << comm_channel.server_clock[server_id] << " to " << clock
           // << endl;
    // }
  // }
  CHECK_LT(server_id, cached_table.server_clock.size());
  // iter_t cur_clock = comm_channel.server_clock[server_id];
  // if (cur_clock != UNINITIALIZED_CLOCK) {
    // /* We might receive duplicate CLOCK message,
     // * because currently the CLOCK message comes with READ_DATA.
     // * We should fix this. */
    // if (clock != cur_clock) {
      // CHECK_EQ(clock, cur_clock + 1);
    // }
  // }
  CHECK_LE(cached_table.server_clock[server_id], clock);
  cached_table.server_clock[server_id] = clock;

  iter_t min_clock = clock_min(cached_table.server_clock);
  if (min_clock > cached_table.server_clock_min) {
    /* Remove oplog entries.
     * We don't grab any locks because we believe there won't be any threads
     * accessing it. */
    iter_t start_clock = cached_table.server_clock_min + 1;
    iter_t end_clock = min_clock;
    reclaim_oplog(
        channel_id, start_clock, end_clock, table_id, true /* gpu */);
    reclaim_oplog(
        channel_id, start_clock, end_clock, table_id, false /* cpu */);
    cached_table.server_clock_min = min_clock;
  }
}

void ClientLib::server_started(uint channel_id, uint server_id) {
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  unique_lock<mutex> channel_lock(*comm_channel.mutex);
  CHECK_LT(server_id, comm_channel.server_started.size());
  comm_channel.server_started[server_id] = 1;
  comm_channel.all_server_started = clock_min(comm_channel.server_started);
  comm_channel.cvar->notify_all();
}

void ClientLib::reclaim_oplog(
    uint channel_id, iter_t start_clock, iter_t end_clock, uint table_id,
    bool gpu) {
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  CHECK_LT(table_id, comm_channel.cached_tables.size());
  CachedTable& cached_table = comm_channel.cached_tables[table_id];
  StaticCache& static_cache = gpu ?
      cached_table.static_cache_gpu :
      cached_table.static_cache_cpu;
  if (static_cache.oplog.size()) {
    for (iter_t clock = start_clock; clock <= end_clock; clock++) {
      if (clock < 0) {
        continue;
      }
      size_t oplog_idx = static_cast<size_t>(clock) % static_cache.oplog.size();
      CHECK_LT(oplog_idx, static_cache.oplog.size());
      FlatOps *oplog_entry_to_remove = static_cache.oplog[oplog_idx];
      if (oplog_entry_to_remove != NULL) {
        // cout << "put " << oplog_entry_to_remove << " at " << clock << endl;
        oplog_entry_to_remove->reset();
        static_cache.opmem_buffer_pool.put(oplog_entry_to_remove);
        static_cache.oplog[oplog_idx] = NULL;
      }
    }
  }
}

void ClientLib::read_batch_local(
    RowData *buffer, OpInfo& op_info,
    DataStorage& local_storage, size_t local_storage_offset,
    cudaStream_t& cuda_stream, RowData *staging_mem) {
  CHECK(buffer);
  CHECK(staging_mem);
  CHECK_EQ(op_info.type, OpInfo::LOCAL_ACCESS);
  CHECK_LT(local_storage_offset, local_storage.size());
  RowData *local_storage_ptr =
      &local_storage.data()[local_storage_offset];
  // size_t size = op_info.num_vals_limit * sizeof(val_t);
  size_t size = op_info.rows.size() * sizeof(RowData);
  // if (local_storage.type() == LocalStorage::PINNED_CPU) {
  if (true) {
    /* If local storage is in pinned memory, we just need one copying */
    CUDA_CHECK(cudaMemcpyAsync(buffer, local_storage_ptr,
        size, cudaMemcpyDefault, cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  } else {
    CHECK_EQ(local_storage.type(), LocalStorage::CPU);
    /* Copy to pinned CPU memory first */
    memcpy(staging_mem, local_storage_ptr, size);
    /* Copy to GPU buffer */
    CUDA_CHECK(cudaMemcpyAsync(buffer, staging_mem,
        size, cudaMemcpyDefault, cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  }
}

void ClientLib::update_batch_local(
    OpInfo& op_info, const RowOpVal *updates,
    DataStorage& local_storage, size_t local_storage_offset,
    cudaStream_t& cuda_stream, RowData *staging_mem) {
  CHECK(updates);
  CHECK(staging_mem);
  CHECK_EQ(op_info.type, OpInfo::LOCAL_ACCESS);
  CHECK_LT(local_storage_offset, local_storage.size());
  RowData *local_storage_ptr =
      &local_storage.data()[local_storage_offset];
  // size_t size = op_info.num_vals_limit * sizeof(val_t);
  size_t size = op_info.rows.size() * sizeof(RowData);
  // if (local_storage.type() == LocalStorage::PINNED_CPU) {
  if (true) {
    /* If local storage is in pinned memory, we just need one copying */
    CUDA_CHECK(cudaMemcpyAsync(local_storage_ptr, updates,
        size, cudaMemcpyDefault, cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  } else {
    CHECK_EQ(local_storage.type(), LocalStorage::CPU);
    /* Copy to pinned CPU memory first */
    CUDA_CHECK(cudaMemcpyAsync(staging_mem, updates,
        size, cudaMemcpyDefault, cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    /* Copy to CPU local storage */
    memcpy(local_storage_ptr, staging_mem, size);
  }
}

void ClientLib::app_acquire_routine(
    RowData **buffer_mem_ret, int handle) {
  // cout << "App acquire, handle #" << handle << endl;
  ThreadData& thread_data_ref = *thread_data;
  iter_t clock = thread_data_ref.current_clock;
  OpSeq& opseq = thread_data_ref.opseq;
  uint op_id = static_cast<uint>(handle);
  CHECK_LT(op_id, opseq.size());
  OpInfo& op_info = opseq[op_id];
  if (!thread_data_ref.bg_worker_started) {
    /* The background worker is not started yet, so do it myself */
    ThreadCache& thread_cache = thread_data_ref.thread_cache;
    LocalStorageOfModelBranches& local_storage_of_model_branches =
        thread_data_ref.local_storage_of_model_branches;
    cudaStream_t& cuda_stream = thread_data_ref.cuda_stream;
    RowData *staging_mem = thread_data_ref.staging_mem;
    bool is_from_bg_worker = false;
    if (op_info.type == OpInfo::READ) {
      /* Read */
      read_impl(
          op_info, clock, opseq, thread_cache,
          staging_mem, cuda_stream, is_from_bg_worker);
    }
    if (op_info.type == OpInfo::PRE_UPDATE) {
      /* PreUpdate */
      preupdate_impl(
          op_info, clock, opseq, thread_cache,
          staging_mem, cuda_stream, is_from_bg_worker);
    }
    if (op_info.type == OpInfo::LOCAL_ACCESS) {
      /* LocalAccess */
      localaccess_impl(
          op_info, clock, opseq, thread_cache,
          local_storage_of_model_branches,
          staging_mem, cuda_stream, is_from_bg_worker);
    }
  } else {
    CHECK_EQ(handle, thread_data_ref.last_handle + 1)
        << "handle mismatch in update_batch()";
    thread_data_ref.last_handle = handle;
  }
  tbb::tick_count time_start = tbb::tick_count::now();
  CHECK_GE(clock, 0);
  CHECK(op_info.op_data_buffers.size());
  size_t op_data_buffers_index =
      static_cast<size_t>(clock) % op_info.op_data_buffers.size();
  OpDataBuffer& op_data_buffer =
      op_info.op_data_buffers[op_data_buffers_index];
  unique_lock<mutex> buffer_lock(op_data_buffer.mutex);
  while (op_data_buffer.buffer == NULL) {
    // op_data_buffer.cvar.wait(buffer_lock);
    if (!op_data_buffer.cvar.timed_wait(buffer_lock,
        boost::posix_time::milliseconds(12000))) {
       cerr << "app acquire waiting timed out for op #" << op_id
            << ", clock = " << clock << endl;
    }
  }
  CHECK(op_data_buffer.buffer);
  CHECK(!op_data_buffer.buffer_being_used);
  CHECK(!op_data_buffer.updates_ready);
  *buffer_mem_ret = op_data_buffer.buffer;
  op_data_buffer.buffer_being_used = true;
  proc_stats.app_read_time +=
      (tbb::tick_count::now() - time_start).seconds();
  CHECK(*buffer_mem_ret);
}

void ClientLib::app_release_routine(int handle) {
  // cout << "App release, handle #" << handle << endl;
  ThreadData& thread_data_ref = *thread_data;
  iter_t clock = thread_data_ref.current_clock;
  OpSeq& opseq = thread_data_ref.opseq;
  uint op_id = static_cast<uint>(handle);
  CHECK_LT(op_id, opseq.size());
  OpInfo& op_info = opseq[op_id];
  CHECK_GE(op_info.prestep_handle, 0);
  uint prestep_op_id = static_cast<uint>(op_info.prestep_handle);
  CHECK_LT(prestep_op_id, opseq.size());
  OpInfo& prestep_op_info = opseq[prestep_op_id];
  CHECK_GE(clock, 0);
  CHECK(prestep_op_info.op_data_buffers.size());
  size_t op_data_buffers_index =
      static_cast<size_t>(clock) % prestep_op_info.op_data_buffers.size();
  OpDataBuffer& op_data_buffer =
      prestep_op_info.op_data_buffers[op_data_buffers_index];
  tbb::tick_count time_start = tbb::tick_count::now();
  unique_lock<mutex> buffer_lock(op_data_buffer.mutex);
  op_data_buffer.buffer_being_used = false;
  op_data_buffer.updates_ready = true;
  /* Notify the background thread to reclaim the read buffer */
  op_data_buffer.cvar.notify_all();
  buffer_lock.unlock();
  proc_stats.app_postread_time +=
      (tbb::tick_count::now() - time_start).seconds();
 
  if (!thread_data_ref.bg_worker_started) {
    /* The background worker is not started yet, so do it myself */
    ThreadCache& thread_cache = thread_data_ref.thread_cache;
    LocalStorageOfModelBranches& local_storage_of_model_branches =
        thread_data_ref.local_storage_of_model_branches;
    cudaStream_t& cuda_stream = thread_data_ref.cuda_stream;
    RowData *staging_mem = thread_data_ref.staging_mem;
    bool is_from_bg_worker = false;
    if (op_info.type == OpInfo::POST_READ) {
      /* PostRead */
      postread_impl(
          op_info, clock, opseq, thread_cache,
          staging_mem, cuda_stream, is_from_bg_worker);
    }
    if (op_info.type == OpInfo::UPDATE) {
      /* Update */
      update_impl(
        op_info, clock, opseq, thread_cache,
        staging_mem, cuda_stream, is_from_bg_worker);
    }
    if (op_info.type == OpInfo::POST_LOCAL_ACCESS) {
      /* PostLocalAccess */
      postlocalaccess_impl(
          op_info, clock, opseq, thread_cache,
          local_storage_of_model_branches,
          staging_mem, cuda_stream, is_from_bg_worker);
    }
  } else {
    CHECK_EQ(handle, thread_data_ref.last_handle + 1)
        << "handle mismatch in update_batch()";
    thread_data_ref.last_handle = handle;
  }
}

void ClientLib::iterate() {
  ThreadData& thread_data_ref = *thread_data;
  if (!thread_data_ref.bg_worker_started) {
    clock_all(thread_data_ref.current_clock);
  }
  thread_data_ref.current_clock++;
  thread_data_ref.last_handle = -1;
}

void ClientLib::start_opseq() {
  /* Create a background worker thread for read/update.
   * The background worker thread will do this operation
   * starting from this clock. */
  ThreadData& thread_data_ref = *thread_data;
  OpSeq& opseq = thread_data_ref.opseq;
  iter_t current_clock = thread_data_ref.current_clock;
  for (uint i = 0; i < opseq.size(); i++) {
    opseq[i].last_finished_clock = current_clock - 1;
  }

  /* Free its CPU buffer */
  CUDA_CHECK(cudaFreeHost(thread_data_ref.staging_mem));

  /* Start alloc worker */
  thread_data_ref.alloc_worker_thread = make_shared<boost::thread>(bind(
      &ClientLib::alloc_worker_entry, this));
  /* Start reclaim worker */
  thread_data_ref.alloc_worker_thread = make_shared<boost::thread>(bind(
      &ClientLib::reclaim_worker_entry, this));
  thread_data_ref.bg_worker_started = true;
  thread_data_ref.last_handle = -1;
}

void ClientLib::alloc_worker_entry() {
  ThreadData& thread_data_ref = *thread_data;
  OpSeq& opseq = thread_data_ref.opseq;
  ThreadCache& thread_cache = thread_data_ref.thread_cache;
  LocalStorageOfModelBranches& local_storage_of_model_branches =
      thread_data_ref.local_storage_of_model_branches;
  RowData *staging_mem;
  mallocHost(&staging_mem, thread_data_ref.staging_mem_size);
  cudaStream_t cuda_stream;
  cublasHandle_t cublas_handle;
  CUDA_CHECK(cudaStreamCreate(&cuda_stream));
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSetStream(cublas_handle, cuda_stream));
  bool is_from_bg_worker = true;
  while (true) {
    for (uint i = 0; i < opseq.size(); i++) {
      OpInfo& op_info = opseq[i];
      iter_t clock = op_info.last_finished_clock + 1;
      if (op_info.type == OpInfo::READ) {
        // cout << "BG Read, handle #" << i << endl;
        read_impl(
            op_info, clock, opseq, thread_cache,
            staging_mem, cuda_stream, is_from_bg_worker);
        continue;
      }
      if (op_info.type == OpInfo::PRE_UPDATE) {
        // cout << "BG PreUpdate, handle #" << i << endl;
        preupdate_impl(
            op_info, clock, opseq, thread_cache,
            staging_mem, cuda_stream, is_from_bg_worker);
        continue;
      }
      if (op_info.type == OpInfo::LOCAL_ACCESS) {
        // cout << "BG LocalAccess, handle #" << i << endl;
        localaccess_impl(
            op_info, clock, opseq, thread_cache,
            local_storage_of_model_branches,
            staging_mem, cuda_stream, is_from_bg_worker);
        continue;
      }
      if (op_info.type == OpInfo::CLOCK) {
        // cout << "BG Clock, handle #" << i;
        /* Do not need to wait.
         * Because we have a set of OpDataBuffer for every clock,
         * the alloc_worker can start the next clock before
         * reclaim_worker finishes */
        break;
        /* Break out of the for loop because the operations after CLOCK
         * are unrepeated ones */
      }
    }
  }
  /* TODO: free CPU buffer */
  /* TODO: destroy stream and handle */
}

void ClientLib::reclaim_worker_entry() {
  ThreadData& thread_data_ref = *thread_data;
  OpSeq& opseq = thread_data_ref.opseq;
  ThreadCache& thread_cache = thread_data_ref.thread_cache;
  LocalStorageOfModelBranches& local_storage_of_model_branches =
      thread_data_ref.local_storage_of_model_branches;
  RowData *staging_mem;
  mallocHost(&staging_mem, thread_data_ref.staging_mem_size);
  cudaStream_t cuda_stream;
  cublasHandle_t cublas_handle;
  CUDA_CHECK(cudaStreamCreate(&cuda_stream));
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSetStream(cublas_handle, cuda_stream));
  bool is_from_bg_worker = true;
  iter_t clock = UNINITIALIZED_CLOCK;
  while (true) {
    for (uint i = 0; i < opseq.size(); i++) {
      OpInfo& op_info = opseq[i];
      clock = op_info.last_finished_clock + 1;
      if (op_info.type == OpInfo::POST_READ) {
        // cout << "BG PostRead, handle #" << i << endl;
        postread_impl(
            op_info, clock, opseq, thread_cache,
            staging_mem, cuda_stream, is_from_bg_worker);
        continue;
      }
      if (op_info.type == OpInfo::UPDATE) {
        // cout << "BG Update, handle #" << i << endl;
        update_impl(
            op_info, clock, opseq, thread_cache,
            staging_mem, cuda_stream, is_from_bg_worker);
        continue;
      }
      if (op_info.type == OpInfo::POST_LOCAL_ACCESS) {
        // cout << "BG PostLocalAccess, handle #" << i << endl;
        postlocalaccess_impl(
            op_info, clock, opseq, thread_cache,
            local_storage_of_model_branches,
            staging_mem, cuda_stream, is_from_bg_worker);
        continue;
      }
      if (op_info.type == OpInfo::CLOCK) {
        // cout << "BG Clock, handle #" << i;
        reclaim_worker_clock(
            op_info, staging_mem, cuda_stream, cublas_handle);
        break;
        /* Break out of the loop because the operations after CLOCK
         * are unrepeated ones */
      }
    }
    {
      /* Update the local data clock */
      /* TODO: do we need to wait for the alloc thread? */
      ScopedLock global_clock_lock(global_clock_mutex);
      CHECK_NE(clock, UNINITIALIZED_CLOCK);
      if (local_data_clock != UNINITIALIZED_CLOCK) {
        CHECK_EQ(clock, local_data_clock + 1);
      }
      local_data_clock = clock;
      global_clock_cvar.notify_all();
    }
  }
  /* TODO: free CPU buffer */
  /* TODO: destroy stream and handle */
}

void ClientLib::localaccess_impl(
    OpInfo& op_info, iter_t clock, OpSeq& opseq, ThreadCache& thread_cache,
    LocalStorageOfModelBranches& local_storage_of_model_branches,
    RowData *staging_mem, cudaStream_t cuda_stream, bool is_from_bg_worker) {
  tbb::tick_count tick_start;
  tbb::tick_count time_start = tbb::tick_count::now();

  CHECK_GE(clock, 0);
  int branch_id = get_branch_id(clock);
  uint branch_idx = get_branch_idx(branch_id);
  CHECK_LT(branch_idx, local_storage_of_model_branches.size());
  LocalStorageOfModelBranch& local_storage_of_model_branch =
      local_storage_of_model_branches[branch_idx];
  CHECK_EQ(local_storage_of_model_branch.branch_id, branch_id);
  LocalStorage& local_storage = local_storage_of_model_branch.local_storage;

  CHECK(op_info.op_data_buffers.size());
  size_t op_data_buffers_index =
      static_cast<size_t>(clock) % op_info.op_data_buffers.size();
  OpDataBuffer& op_data_buffer =
      op_info.op_data_buffers[op_data_buffers_index];
  // cout << "alloc worker at op #" << i << "clock " << clock << endl;
  // cout << "need lock " << &(op_data_buffer.mutex) << endl;
  unique_lock<mutex> buffer_lock(op_data_buffer.mutex);
  CHECK(op_data_buffer.buffer == NULL);
  CHECK(!op_data_buffer.buffer_being_used);
  CHECK(!op_data_buffer.updates_ready);
  CHECK(op_info.shared_buffer);
  SharedBuffer& shared_buffer = *op_info.shared_buffer;
  uint num_rows = op_info.rows.size();
  unique_lock<mutex> shared_buffer_lock(shared_buffer.mutex);
  while (shared_buffer.buffer && shared_buffer.branch_id != -1 &&
         shared_buffer.branch_id != branch_id) {
    /* The shared buffer is used by some other branches,
     * so we wait for it to be released.
     * We can actually also make one shared buffer for each branch,
     * but I don't think it's really necessary. */
    /* Since we have multiple sets of op_buffers,
     * we won't have a deadlock here. */
    if (!shared_buffer.cvar.timed_wait(shared_buffer_lock,
        boost::posix_time::milliseconds(12000))) {
       cerr << "opseq worker of machine " << process_id
            << " waiting timed out for shared buffer to be released\n";
    }
  }
  bool new_buffer = false;
  if (shared_buffer.buffer) {
    /* Buffer already exists */
    if (shared_buffer.ref_count == 0) {
      CHECK(shared_buffer.pinned_entry);
      /* When we swap data to CPU memory in postlocalaccess_impl(),
       * we will reset shared_buffer.branch_id to -1 */
      CHECK(shared_buffer.branch_id == -1 ||
            shared_buffer.branch_id == branch_id);
      new_buffer = shared_buffer.branch_id == -1;
      /* This pinned entry is not being used, so we just overupdate it */
      shared_buffer.branch_id = branch_id;
    }
    CHECK_EQ(shared_buffer.branch_id, branch_id);
    shared_buffer.ref_count++;
    op_data_buffer.buffer = shared_buffer.buffer;
    op_data_buffer.branch_id = branch_id;
    CHECK_EQ(shared_buffer.num_rows, num_rows);
  } else {
    /* Allocate a buffer */
    CHECK_EQ(shared_buffer.ref_count, 0);
    uint num_rows = op_info.rows.size();
    tick_start = tbb::tick_count::now();
    bool wait = true;
    int tag = clock;  /* just for debugging message printing */
    while ((shared_buffer.buffer =
        thread_cache.get(num_rows, wait, tag)) == NULL) {
      cerr << "opseq worker of machine " << process_id
            << " has no more space, op #" //<< i
            << endl;
       cerr << "clock = " << clock << endl;
    }
    op_info.alloc_wait_time +=
          (tbb::tick_count::now() - tick_start).seconds();
    CHECK(shared_buffer.buffer);
    shared_buffer.num_rows = num_rows;
    shared_buffer.branch_id = branch_id;
    shared_buffer.ref_count++;
    op_data_buffer.buffer = shared_buffer.buffer;
    op_data_buffer.branch_id = branch_id;
    new_buffer = true;
  }
  if (new_buffer && op_info.fetch_local) {
    /* Copy CPU local storage to GPU buffer if the app needs to
     * read the data */
    CHECK(shared_buffer.has_local_storage);
    tick_start = tbb::tick_count::now();
    read_batch_local(
        op_data_buffer.buffer, op_info,
        local_storage, shared_buffer.storage_offset,
        cuda_stream, staging_mem);
    op_info.fetch_time +=
        (tbb::tick_count::now() - tick_start).seconds();
    op_info.rows_fetched += num_rows;
  }
  shared_buffer.cvar.notify_all();
  op_data_buffer.buffer_being_used = false;
  op_data_buffer.updates_ready = false;
  op_info.last_finished_clock = clock;
  op_data_buffer.cvar.notify_all();
  proc_stats.bg_read_time +=
      (tbb::tick_count::now() - time_start).seconds();
}

void ClientLib::read_impl(
    OpInfo& op_info, iter_t clock, OpSeq& opseq, ThreadCache& thread_cache,
    RowData *staging_mem, cudaStream_t cuda_stream, bool is_from_bg_worker) {
  tbb::tick_count time_start = tbb::tick_count::now();

  CHECK_GE(clock, 0);
  CHECK(op_info.op_data_buffers.size());
  size_t op_data_buffers_index =
      static_cast<size_t>(clock) % op_info.op_data_buffers.size();
  OpDataBuffer& op_data_buffer =
      op_info.op_data_buffers[op_data_buffers_index];
  // cout << "alloc worker at op #" << i << "clock " << clock << endl;
  // cout << "need lock " << &(op_data_buffer.mutex) << endl;
  unique_lock<mutex> buffer_lock(op_data_buffer.mutex);
  CHECK(op_data_buffer.buffer == NULL);
  CHECK(!op_data_buffer.buffer_being_used);
  CHECK(!op_data_buffer.updates_ready);
  uint num_rows = op_info.rows.size();
  tbb::tick_count tick_start = tbb::tick_count::now();
  bool wait = true;
  int tag = clock;
  while ((op_data_buffer.buffer =
      thread_cache.get(num_rows, wait, tag)) == NULL) {
    cerr << "opseq worker of machine " << process_id
          << " has no more space, op #" //<< i
          << endl;
    cerr << "clock = " << clock << endl;
  }
  op_info.alloc_wait_time +=
        (tbb::tick_count::now() - tick_start).seconds();
  CHECK(op_data_buffer.buffer);
  int branch_id;
  iter_t internal_clock;
  iter_t required_internal_data_age;
  uint branch_idx;
  {
    ScopedLock branch_manager_lock(branch_manager->mutex);
    ClockSchedule clock_schedule =
        branch_manager->get_clock_schedule(&branch_manager_lock, clock);
    branch_id = clock_schedule.branch_id;
    internal_clock = clock_schedule.internal_clock;
    const Tunable& tunable = branch_manager->get_tunable(branch_id);
    int slack = tunable.slack;
    CHECK_LE(slack, config.max_slack);
    required_internal_data_age = internal_clock - slack - 1;
    branch_idx = branch_manager->get_branch_idx(branch_id);
  }
  tick_start = tbb::tick_count::now();
  /* We will keep prefetching data from process cache to the op_data_buffer,
   * as long as we know the clock schedule and we haven't hit the slack bound.*/
  read_row_batch_static_cache(
      op_data_buffer.buffer, op_info, branch_id, branch_idx,
      required_internal_data_age,
      cuda_stream, staging_mem);
  op_info.read_time +=
      (tbb::tick_count::now() - tick_start).seconds();
  op_data_buffer.buffer_being_used = false;
  op_data_buffer.updates_ready = false;
  op_data_buffer.branch_id = branch_id;
  op_info.last_finished_clock = clock;
  op_data_buffer.cvar.notify_all();
  proc_stats.bg_read_time +=
      (tbb::tick_count::now() - time_start).seconds();
}

void ClientLib::preupdate_impl(
    OpInfo& op_info, iter_t clock, OpSeq& opseq, ThreadCache& thread_cache,
    RowData *staging_mem, cudaStream_t cuda_stream, bool is_from_bg_worker) {
  tbb::tick_count tick_start;
  tbb::tick_count time_start = tbb::tick_count::now();
  CHECK_GE(clock, 0);
  CHECK(op_info.op_data_buffers.size());
  size_t op_data_buffers_index =
      static_cast<size_t>(clock) % op_info.op_data_buffers.size();
  OpDataBuffer& op_data_buffer =
      op_info.op_data_buffers[op_data_buffers_index];
  // cout << "alloc worker at op #" << i << "clock " << clock << endl;
  // cout << "need lock " << &(op_data_buffer.mutex) << endl;
  unique_lock<mutex> buffer_lock(op_data_buffer.mutex);
  CHECK(op_data_buffer.buffer == NULL);
  CHECK(!op_data_buffer.buffer_being_used);
  CHECK(!op_data_buffer.updates_ready);
  tick_start = tbb::tick_count::now();
  uint num_rows = op_info.rows.size();
  bool wait = true;
  while ((op_data_buffer.buffer =
      thread_cache.get(num_rows, wait, clock)) == NULL) {
     cerr << "opseq worker of machine " << process_id
          << " has no more space, op #" //<< i
          << endl;
     cerr << "clock = " << clock << endl;
  }
  op_info.alloc_wait_time +=
        (tbb::tick_count::now() - tick_start).seconds();
  CHECK(op_data_buffer.buffer);
  op_data_buffer.buffer_being_used = false;
  op_data_buffer.updates_ready = false;
  // op_data_buffer.branch_id = branch_id;
  op_info.last_finished_clock = clock;
  op_data_buffer.cvar.notify_all();
  proc_stats.bg_preupdate_time +=
      (tbb::tick_count::now() - time_start).seconds();
}

void ClientLib::postlocalaccess_impl(
    OpInfo& op_info, iter_t clock, OpSeq& opseq, ThreadCache& thread_cache,
    LocalStorageOfModelBranches& local_storage_of_model_branches,
    RowData *staging_mem, cudaStream_t cuda_stream, bool is_from_bg_worker) {
  tbb::tick_count tick_start;
  tbb::tick_count time_start = tbb::tick_count::now();
  CHECK_GE(op_info.prestep_handle, 0);
  uint prestep_op_id = static_cast<uint>(op_info.prestep_handle);
  CHECK_LT(prestep_op_id, opseq.size());
  OpInfo& prestep_op_info = opseq[prestep_op_id];
  CHECK_EQ(prestep_op_info.type, OpInfo::LOCAL_ACCESS);

  CHECK(prestep_op_info.op_data_buffers.size());
  size_t op_data_buffers_index =
      static_cast<size_t>(clock) % prestep_op_info.op_data_buffers.size();
  OpDataBuffer& op_data_buffer =
      prestep_op_info.op_data_buffers[op_data_buffers_index];
  // cout << "reclaim worker at op #" << i
       // << "(" << op_info.prestep_handle << "), clock " << clock << endl;
  // cout << "need lock " << &(op_data_buffer.mutex) << endl;
  unique_lock<mutex> buffer_lock(op_data_buffer.mutex);
  /* Wait for the application to finish using the read buffer */
  tick_start = tbb::tick_count::now();
  while (!op_data_buffer.updates_ready) {
    // op_data_buffer.cvar.wait(buffer_lock);
    if (!op_data_buffer.cvar.timed_wait(buffer_lock,
        boost::posix_time::milliseconds(12000))) {
       cerr << "opseq worker of machine " << process_id
            << " waiting for updates ready timed out, op #"
            << prestep_op_id << endl;
       // cerr << "POST_READ op #" << i << endl;
       cerr << "clock = " << clock << endl;
       cerr << "preop.updates_ready = " << op_data_buffer.updates_ready << endl;
       cerr << "preop.buffer = " << op_data_buffer.buffer << endl;
       cerr << "preop.buffer_being_used = "
            << op_data_buffer.buffer_being_used << endl;
    }
  }
  op_info.reclaim_wait_time +=
      (tbb::tick_count::now() - tick_start).seconds();
  CHECK(op_data_buffer.updates_ready);
  CHECK(op_data_buffer.buffer);
  CHECK(!op_data_buffer.buffer_being_used);

  /* Release cache */
  CHECK(prestep_op_info.shared_buffer);
  SharedBuffer& shared_buffer = *prestep_op_info.shared_buffer;
  unique_lock<mutex> shared_buffer_lock(shared_buffer.mutex);
  CHECK_EQ(shared_buffer.buffer, op_data_buffer.buffer);
  CHECK_EQ(shared_buffer.branch_id, op_data_buffer.branch_id);
  CHECK_GT(shared_buffer.ref_count, 0);
  shared_buffer.ref_count--;
  bool reclaim_buffer = false;
  if (shared_buffer.ref_count == 0) {
    if (!shared_buffer.pinned_entry) {
      reclaim_buffer = true;
    } else {
      /* For pinned entries, we will look at the branch id of the next clock.
       * We only swap out pinned entry, when it's the last access of this clock
       * and we will switch branch for the next clock */
      if (op_info.batch_last_update) {
        /* We try to peek at the schedule of the next clock.
         * If we will run the same branch for the next clock,
         * we can have the pinned entries stay in the thread cache.
         * If we don't know the schedule of the next clock,
         * or if we are going to run a different branch for the next clock,
         * we will need to evict this entry. */
        /* Note that the recv_make_branch() function will wait
         * for the current clock to finish, before processing
         * the make branch request, and the clock schedule is added after
         * the new branch is made.
         * So we might not necessarily be able to know
         * the schedule of the next clock,
         * but we will know it when the branch scheduler schedules
         * multiple clocks for us to run. */
        int branch_id_of_next_clock = get_branch_id_no_wait(clock + 1);
        if (branch_id_of_next_clock == -1
            || branch_id_of_next_clock != shared_buffer.branch_id) {
          reclaim_buffer = true;
        }
      }
    }
  }
  if (reclaim_buffer) {
    uint num_rows = prestep_op_info.rows.size();
    CHECK_EQ(shared_buffer.num_rows, num_rows);
    /* Copy GPU buffer to CPU local storage if the app wants to keep it */
    /* The last access decides whether it needs keep or not */
    shared_buffer.need_keep = op_info.keep_local;
    if (shared_buffer.need_keep) {
      int branch_id = op_data_buffer.branch_id;
      CHECK_GE(branch_id, 0);
      uint branch_idx = get_branch_idx(branch_id);
      CHECK_LT(branch_idx, local_storage_of_model_branches.size());
      LocalStorageOfModelBranch& local_storage_of_model_branch =
          local_storage_of_model_branches[branch_idx];
      CHECK_EQ(local_storage_of_model_branch.branch_id, branch_id);
      LocalStorage& local_storage = local_storage_of_model_branch.local_storage;
      tick_start = tbb::tick_count::now();
      update_batch_local(
          prestep_op_info, op_data_buffer.buffer,
          local_storage, shared_buffer.storage_offset,
          cuda_stream, staging_mem);
      op_info.keep_time +=
          (tbb::tick_count::now() - tick_start).seconds();
      op_info.rows_kept += num_rows;
    }
    thread_cache.put(op_data_buffer.buffer, num_rows);
    shared_buffer.buffer = NULL;
    shared_buffer.branch_id = -1;
  }
  shared_buffer.cvar.notify_all();

  op_data_buffer.buffer = NULL;
  op_data_buffer.branch_id = -1;
  op_data_buffer.buffer_being_used = false;
  op_data_buffer.updates_ready = false;
  op_info.last_finished_clock = clock;
  /* Don't need to notify, no one is waiting for this */
  proc_stats.bg_postread_time +=
      (tbb::tick_count::now() - time_start).seconds();
}

void ClientLib::postread_impl(
    OpInfo& op_info, iter_t clock, OpSeq& opseq, ThreadCache& thread_cache,
    RowData *staging_mem, cudaStream_t cuda_stream, bool is_from_bg_worker) {
  tbb::tick_count tick_start;
  tbb::tick_count time_start = tbb::tick_count::now();
  CHECK_GE(op_info.prestep_handle, 0);
  uint prestep_op_id = static_cast<uint>(op_info.prestep_handle);
  CHECK_LT(prestep_op_id, opseq.size());
  OpInfo& prestep_op_info = opseq[prestep_op_id];
  CHECK_EQ(prestep_op_info.type, OpInfo::READ);
  CHECK_GE(clock, 0);
  CHECK(prestep_op_info.op_data_buffers.size());
  size_t op_data_buffers_index =
      static_cast<size_t>(clock) % prestep_op_info.op_data_buffers.size();
  OpDataBuffer& op_data_buffer =
      prestep_op_info.op_data_buffers[op_data_buffers_index];
  // cout << "reclaim worker at op #" << i
       // << "(" << op_info.prestep_handle << "), clock " << clock << endl;
  // cout << "need lock " << &(op_data_buffer.mutex) << endl;
  unique_lock<mutex> buffer_lock(op_data_buffer.mutex);
  /* Wait for the application to finish using the read buffer */
  tick_start = tbb::tick_count::now();
  while (!op_data_buffer.updates_ready) {
    // op_data_buffer.cvar.wait(buffer_lock);
    if (!op_data_buffer.cvar.timed_wait(buffer_lock,
        boost::posix_time::milliseconds(12000))) {
       cerr << "opseq worker of machine " << process_id
            << " waiting for updates ready timed out, op #"
            << prestep_op_id << endl;
       // cerr << "POST_READ op #" << i << endl;
       cerr << "clock = " << clock << endl;
       cerr << "preop.last_finished_clock = "
            << prestep_op_info.last_finished_clock << endl;
       cerr << "preop.updates_ready = "
            << op_data_buffer.updates_ready << endl;
       cerr << "preop.buffer = " << op_data_buffer.buffer << endl;
       cerr << "preop.buffer_being_used = "
            << op_data_buffer.buffer_being_used << endl;
    }
  }
  op_info.reclaim_wait_time +=
      (tbb::tick_count::now() - tick_start).seconds();
  CHECK(op_data_buffer.updates_ready);
  CHECK(op_data_buffer.buffer);
  CHECK(!op_data_buffer.buffer_being_used);
  /* Release cache */
  uint num_rows = prestep_op_info.rows.size();
  thread_cache.put(op_data_buffer.buffer, num_rows);
  op_data_buffer.buffer = NULL;
  op_data_buffer.branch_id = -1;
  op_data_buffer.buffer_being_used = false;
  op_data_buffer.updates_ready = false;
  op_info.last_finished_clock = clock;
  /* Don't need to notify, no one is waiting for this */
  proc_stats.bg_postread_time +=
      (tbb::tick_count::now() - time_start).seconds();
}

void ClientLib::update_impl(
    OpInfo& op_info, iter_t clock, OpSeq& opseq, ThreadCache& thread_cache,
    RowData *staging_mem, cudaStream_t cuda_stream, bool is_from_bg_worker) {
  tbb::tick_count tick_start;
  tbb::tick_count time_start = tbb::tick_count::now();
  int branch_id = get_branch_id(clock);
  CHECK_GE(op_info.prestep_handle, 0);
  uint prestep_op_id = static_cast<uint>(op_info.prestep_handle);
  CHECK_LT(prestep_op_id, opseq.size());
  OpInfo& prestep_op_info = opseq[prestep_op_id];
  CHECK_EQ(prestep_op_info.type, OpInfo::PRE_UPDATE);
  CHECK_GE(clock, 0);
  CHECK(prestep_op_info.op_data_buffers.size());
  size_t op_data_buffers_index =
      static_cast<size_t>(clock) % prestep_op_info.op_data_buffers.size();
  OpDataBuffer& op_data_buffer =
      prestep_op_info.op_data_buffers[op_data_buffers_index];
  // cout << "reclaim worker at op #" << i
       // << "(" << op_info.prestep_handle << "), clock " << clock << endl;
  // cout << "need lock " << &(op_data_buffer.mutex) << endl;
  unique_lock<mutex> buffer_lock(op_data_buffer.mutex);
  /* Wait for the updates to be ready */
  tick_start = tbb::tick_count::now();
  while (!op_data_buffer.updates_ready) {
    // op_data_buffer.cvar.wait(buffer_lock);
    if (!op_data_buffer.cvar.timed_wait(buffer_lock,
        boost::posix_time::milliseconds(12000))) {
       cerr << "opseq worker of machine " << process_id
            << " waiting for updates ready timed out, op #"
            << prestep_op_id << endl;
       // cerr << "UPDATE op #" << i << endl;
       cerr << "clock = " << clock << endl;
       cerr << "last_finished_clock = "
            << op_info.last_finished_clock << endl;
       cerr << "preop.last_finished_clock = "
            << prestep_op_info.last_finished_clock << endl;
       cerr << "preop.updates_ready = " << op_data_buffer.updates_ready << endl;
       cerr << "preop.buffer = " << op_data_buffer.buffer << endl;
       cerr << "preop.buffer_being_used = "
            << op_data_buffer.buffer_being_used << endl;
    }
    op_info.reclaim_wait_time +=
        (tbb::tick_count::now() - tick_start).seconds();
  }
  CHECK(op_data_buffer.updates_ready);
  CHECK(op_data_buffer.buffer);
  CHECK(!op_data_buffer.buffer_being_used);
  /* Apply updates */
  tick_start = tbb::tick_count::now();
  update_batch_static_cache(
      prestep_op_info, branch_id, op_data_buffer.buffer, clock,
      cuda_stream, staging_mem);
  op_info.update_time +=
      (tbb::tick_count::now() - tick_start).seconds();
  /* Release cache */
  uint num_rows = prestep_op_info.rows.size();
  thread_cache.put(op_data_buffer.buffer, num_rows);
  op_data_buffer.buffer = NULL;
  op_data_buffer.branch_id = -1;
  op_data_buffer.buffer_being_used = false;
  op_data_buffer.updates_ready = false;
  op_info.last_finished_clock = clock;
  /* Signal a table_clock if that's the last update of this table */
  if (is_from_bg_worker && op_info.table_last_update) {
    uint table_id = op_info.table_id;
    /* Signalling a "clock" means that we have finished "clock" */
    clock_table(clock, table_id);
  }
  /* Don't need to notify, no one is waiting for this */
  proc_stats.bg_update_time +=
      (tbb::tick_count::now() - time_start).seconds();
}

void ClientLib::reclaim_worker_clock(
    OpInfo& op_info, RowData *staging_mem,
    cudaStream_t cuda_stream, cublasHandle_t cublas_handle) {
  OpSeq& opseq = thread_data->opseq;

  // cout << ", last_finished_clock = " << op_info.last_finished_clock << endl;
  // tbb::tick_count time_start = tbb::tick_count::now();
  iter_t clock = op_info.last_finished_clock + 1;
  op_info.last_finished_clock = clock;
  // /* Signalling a "clock + 1" means that we have finished "clock" */
  // tick_start = tbb::tick_count::now();
  // clock_all(clock + 1);
  // op_info.update_time +=
      // (tbb::tick_count::now() - tick_start).seconds();
  // proc_stats.bg_clock_time +=
      // (tbb::tick_count::now() - time_start).seconds();
  if (config.log_interval > 0 &&
      clock > 0 && clock % config.log_interval == 0) {
    cout << "bg_worker_times:" << endl;
    size_t total_fetch = 0;
    size_t total_keep = 0;
    for (uint i = 0; i < opseq.size(); i++) {
      OpInfo& op_info = opseq[i];
      total_fetch += op_info.rows_fetched;
      total_keep += op_info.rows_kept;
      cerr << i
           << "," << op_info.read_time
           << "," << op_info.fetch_time
           << "," << op_info.alloc_wait_time
           << "," << op_info.update_time
           << "," << op_info.keep_time
           << "," << op_info.reclaim_wait_time
           << "," << op_info.rows_fetched
           << "," << op_info.rows_kept
           << endl;
    }
    cerr << "total_fetch=" << total_fetch << endl;
    cerr << "total_keep=" << total_keep << endl;
  }
}

void ClientLib::report(double progress) {
  iter_t clock = thread_data->current_clock;
  int branch_id = get_branch_id(clock);
  tuner_comm_channel.encoder->report_progress(
        clock, branch_id, progress);
}

int ClientLib::get_branch_id(iter_t clock) {
  ScopedLock branch_manager_lock(branch_manager->mutex);
  ClockSchedule clock_schedule =
      branch_manager->get_clock_schedule(&branch_manager_lock, clock);
  int branch_id = clock_schedule.branch_id;
  return branch_id;
}

/* This function does not wait.
 * If the clock schedule of the next clock is unknown,
 * the clock_schedule will be filled with (-1, 0) */
int ClientLib::get_branch_id_no_wait(iter_t clock) {
  ScopedLock branch_manager_lock(branch_manager->mutex);
  ClockSchedule clock_schedule =
      branch_manager->get_clock_schedule_no_wait(clock);
  int branch_id = clock_schedule.branch_id;
  return branch_id;
}

Tunable ClientLib::get_tunable() {
  ScopedLock branch_manager_lock(branch_manager->mutex);
  ClockSchedule clock_schedule = branch_manager->get_clock_schedule(
      &branch_manager_lock, thread_data->current_clock);
  int branch_id = clock_schedule.branch_id;
  /* TODO: we can actually cache this in thread_data */
  Tunable tunable = branch_manager->get_tunable(branch_id);
  return tunable;
}

void ClientLib::get_current_branch_info(Tunable *tunable, int *flag) {
  ScopedLock branch_manager_lock(branch_manager->mutex);
  ClockSchedule clock_schedule = branch_manager->get_clock_schedule(
      &branch_manager_lock, thread_data->current_clock);
  int branch_id = clock_schedule.branch_id;
  BranchInfo *branch_info = branch_manager->branch_info_map[branch_id];
  CHECK(branch_info);
  /* TODO: we can actually cache this in thread_data */
  *tunable = branch_info->tunable;
  *flag = branch_info->flag;
}

shared_ptr<BranchManager> ClientLib::get_branch_manager() {
  return branch_manager;
}

uint ClientLib::get_branch_idx(int branch_id) {
  ScopedLock branch_manager_lock(branch_manager->mutex);
  uint branch_idx = branch_manager->get_branch_idx(branch_id);
  return branch_idx;
}

void ClientLib::recv_branch_schedules(
    uint batch_size, iter_t *clocks, int *branch_ids) {
  ScopedLock branch_manager_lock(branch_manager->mutex);
  branch_manager->add_clock_schedules(batch_size, clocks, branch_ids);
}

void ClientLib::wait_for_all_clocks(
    ScopedLock *global_clock_lock_ptr, iter_t clock) {
  while (true) {
    bool time_to_process_request = true;
    if (clock == INITIAL_CLOCK) {
      for (uint table_id = 0; table_id < config.num_tables; table_id++) {
        CHECK_LT(table_id, all_table_global_clocks.size());
        iter_t global_clock = all_table_global_clocks[table_id];
        CHECK_EQ(global_clock, UNINITIALIZED_CLOCK);
      }
      CHECK_EQ(local_data_clock, UNINITIALIZED_CLOCK);
    } else {
      for (uint table_id = 0; table_id < config.num_tables; table_id++) {
        CHECK_LT(table_id, all_table_global_clocks.size());
        iter_t global_clock = all_table_global_clocks[table_id];
        if (clock != global_clock + 1) {
          CHECK_GT(clock, global_clock + 1);
          time_to_process_request = false;
        }
      }
      if (clock != local_data_clock + 1) {
        CHECK_GT(clock, local_data_clock + 1);
        time_to_process_request = false;
      }
    }
    if (time_to_process_request) {
      break;  /* Break out the while loop */
    }
    if (!global_clock_cvar.timed_wait(*global_clock_lock_ptr,
        boost::posix_time::milliseconds(30000))) {
      cerr << "recv_make_branch() waiting for global clock timed out\n";
    }
  }
}

void ClientLib::signal_bgworker_forward_msg(
    vector<ZmqPortableBytes>& args) {
  for (uint channel_id = 0; channel_id < comm_channels.size(); channel_id++) {
    CommunicationChannel& comm_channel = comm_channels[channel_id];
    vector<ZmqPortableBytes> args_copy(args.size());
    for (uint j = 0; j < args_copy.size(); j ++) {
      args_copy[j].copy(args[j]);
    }
    comm_channel.work_pusher->push_work(FORWARD_MSG_CMD, args_copy);
  }
}

void ClientLib::cbk_forward_msg(
    uint channel_id, vector<ZmqPortableBytes>& args) {
  CHECK_LT(channel_id, comm_channels.size());
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  comm_channel.encoder->broadcast_msg(args);
  /* The message will be copied by the encoder,
   * so we can safely close our message object */
  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ClientLib::recv_make_branch(
    int branch_id, const Tunable& tunable, int flag,
    int parent_branch_id, iter_t clock_to_happen,
    vector<ZmqPortableBytes>& args) {
  {
    /* Wait for the virtual iteration to finish,
     * because we need to set the branch_id field of all static cache rows */
    ScopedLock vilock(virtual_iteration_mutex);
    while (!virtual_iteration_all_finished) {
      if (!virtual_iteration_cvar.timed_wait(vilock,
          boost::posix_time::milliseconds(30000))) {
        cerr << "recv_make_branch() waiting for virutal iteration finish timed out\n";
      }
    }
  }

  {
    /* Wait for all threads to finish the previous clocks */
    ScopedLock global_clock_lock(global_clock_mutex);
    wait_for_all_clocks(&global_clock_lock, clock_to_happen);

    ScopedLock branch_manager_lock(branch_manager->mutex);
    BranchInfo *branch_info =
        new BranchInfo(branch_id, tunable, flag, parent_branch_id);
    CHECK(!branch_manager->branch_info_map[branch_id])
        << "branch " << branch_id << " already exists\n";
    branch_manager->branch_info_map[branch_id] = branch_info;
    uint ref_count = 1;
    uint branch_idx = branch_manager->alloc_branch_idx(ref_count);
    branch_info->branch_idx = branch_idx;

    if (process_id == 0) {
      cout << "Make branch " << branch_id
           << " at clock " << clock_to_happen
           << " got idx " << branch_idx << endl;
    }

    /* Set the branch_id field of all static cache tables.
     * We need to do that before forwarding the make branch request
     * to the server, because the server will push param data
     * to the process cache. */
    for (uint channel_id = 0; channel_id < comm_channels.size(); channel_id++) {
      CommunicationChannel& comm_channel = comm_channels[channel_id];
      ScopedLock channel_lock(*comm_channel.mutex);
      CachedTables& cached_tables = comm_channel.cached_tables;
      for (uint table_id = 0; table_id < cached_tables.size(); table_id++) {
        CachedTable& cached_table = cached_tables[table_id];
        /* NOTE: we assume we have only CPU cache */
        DataCacheOfModelBranches& data_cache_of_model_branches =
            cached_table.static_cache_cpu.data_cache_of_model_branches;
        CHECK_LT(branch_idx, data_cache_of_model_branches.size()); 
        DataCacheOfModelBranch& data_cache_of_model_branch =
            data_cache_of_model_branches[branch_idx];
        CHECK_EQ(data_cache_of_model_branch.branch_id, -1);
        CHECK_EQ(data_cache_of_model_branch.internal_data_age,
            UNINITIALIZED_CLOCK);
        data_cache_of_model_branch.branch_id = branch_id;
      }
    }

    /* Make branch for each table and the local data */
    for (uint table_id = 0; table_id < config.num_tables; table_id++) {
      make_branch_for_table(branch_id, table_id);
    }
    make_branch_for_local_data(branch_id);
  }   /* Unlock global_clock_lock */

  /* Forward the branch operation to the servers */
  signal_bgworker_forward_msg(args);

  branch_manager->cvar.notify_all();
}

void ClientLib::make_branch_for_table(int branch_id, uint table_id) {
  /* Do nothing for the static cache */
}

void ClientLib::make_branch_for_local_data(int branch_id) {
  /* The branch manager lock is held while calling this function */
  BranchInfo *branch_info = branch_manager->branch_info_map[branch_id];
  CHECK(branch_info);
  int parent_branch_id = branch_info->parent_branch_id;
  /* Copy the local storage */
  /* We are sure that the local data in the thread cache must
   * have been flushed to the local storage,
   * because the reclaim worker only keeps pinned entries in the thread cache,
   * when the schedule of the next clock is available,
   * and this information won't be available,
   * when we plan to do a make_branch for the next clock. */
  CHECK(thread_data.get());
  ThreadData& thread_data_ref = *thread_data;
  LocalStorageOfModelBranches& local_storage_of_model_branches =
      thread_data_ref.local_storage_of_model_branches;
  // cout << "Client " << process_id << " process make branch" << endl;
  uint child_branch_idx = branch_manager->get_branch_idx(branch_id);
  CHECK_LT(child_branch_idx, local_storage_of_model_branches.size());
  LocalStorageOfModelBranch& child_branch_storage =
      local_storage_of_model_branches[child_branch_idx];
  CHECK_EQ(child_branch_storage.branch_id, -1);
  child_branch_storage.branch_id = branch_id;
  if (parent_branch_id < 0) {
    return;
  }
  uint parent_branch_idx = branch_manager->get_branch_idx(parent_branch_id);
  CHECK_LT(parent_branch_idx, local_storage_of_model_branches.size());
  LocalStorageOfModelBranch& parent_branch_storage =
      local_storage_of_model_branches[parent_branch_idx];
  LocalStorage& child_local_storage = child_branch_storage.local_storage;
  LocalStorage& parent_local_storage = parent_branch_storage.local_storage;
  if (parent_local_storage.size()) {
    child_local_storage.init_from(parent_local_storage);
    child_local_storage.copy_data_cpu(parent_local_storage);
    /* TODO: actually we don't need to copy all the local data.
     * Instead, only the momentum needs to be copied */
  }
}

void ClientLib::recv_inactivate_branch(
    int branch_id, iter_t clock_to_happen,
    vector<ZmqPortableBytes>& args) {
  // cout << "Client received inactivate branch " << branch_id
       // << " at clock " << clock_to_happen << endl;
  {
    /* Wait for all threads to finish the previous clocks */
    ScopedLock global_clock_lock(global_clock_mutex);
    wait_for_all_clocks(&global_clock_lock, clock_to_happen);

    ScopedLock branch_manager_lock(branch_manager->mutex);

    /* Inactivate branch for each table and the local data */
    for (uint table_id = 0; table_id < config.num_tables; table_id++) {
      inactivate_branch_for_table(branch_id, table_id);
    }
    inactivate_branch_for_local_data(branch_id);

    branch_manager->free_branch(branch_id);
  }

  /* Forward the branch operation to the servers */
  signal_bgworker_forward_msg(args);
}

void ClientLib::inactivate_branch_for_table(int branch_id, uint table_id) {
  /* The branch manager lock is held while calling this function */
  uint branch_idx = branch_manager->get_branch_idx(branch_id);
  /* Reset the data_age and branch_id fields of this static cache table */
  /* TODO: snapshot to files */
  for (uint channel_id = 0; channel_id < comm_channels.size(); channel_id++) {
    CommunicationChannel& comm_channel = comm_channels[channel_id];
    ScopedLock channel_lock(*comm_channel.mutex);
    CachedTables& cached_tables = comm_channel.cached_tables;
    CachedTable& cached_table = cached_tables[table_id];
    /* NOTE: we assume we have only CPU cache */
    CHECK_EQ(cached_table.static_cache_gpu.num_rows, 0);
    DataCacheOfModelBranches& data_cache_of_model_branches =
        cached_table.static_cache_cpu.data_cache_of_model_branches;
    CHECK_LT(branch_idx, data_cache_of_model_branches.size()); 
    DataCacheOfModelBranch& data_cache_of_model_branch =
        data_cache_of_model_branches[branch_idx];
    CHECK_EQ(data_cache_of_model_branch.branch_id, branch_id);
    data_cache_of_model_branch.reset();
  }
}

void ClientLib::inactivate_branch_for_local_data(int branch_id) {
  /* The branch manager lock is held while calling this function */
  uint branch_idx = branch_manager->get_branch_idx(branch_id);
  /* Reset the branch_id field of local storage */
  LocalStorageOfModelBranches& local_storage_of_model_branches =
      thread_data->local_storage_of_model_branches;
  CHECK_LT(branch_idx, local_storage_of_model_branches.size());
  LocalStorageOfModelBranch& local_storage_of_model_branch =
      local_storage_of_model_branches[branch_idx];
  CHECK_EQ(local_storage_of_model_branch.branch_id, branch_id);
  local_storage_of_model_branch.branch_id = -1;
}

void ClientLib::recv_save_branch(
    int branch_id, iter_t clock_to_happen,
    vector<ZmqPortableBytes>& args) {
  // cout << "Client received save branch " << branch_id
       // << " at clock " << clock_to_happen << endl;
  {
    /* Wait for all threads to finish the previous clocks */
    ScopedLock global_clock_lock(global_clock_mutex);
    wait_for_all_clocks(&global_clock_lock, clock_to_happen);

    ScopedLock branch_manager_lock(branch_manager->mutex);

    save_branch_for_local_data(branch_id);
  }

  /* Forward the branch operation to the servers */
  signal_bgworker_forward_msg(args);
}

void ClientLib::save_branch_for_local_data(int branch_id) {
  /* The branch manager lock is held while calling this function */
  uint branch_idx = branch_manager->get_branch_idx(branch_id);
  /* Reset the branch_id field of local storage */
  LocalStorageOfModelBranches& local_storage_of_model_branches =
      thread_data->local_storage_of_model_branches;
  CHECK_LT(branch_idx, local_storage_of_model_branches.size());
  LocalStorageOfModelBranch& local_storage_of_model_branch =
      local_storage_of_model_branches[branch_idx];
  CHECK_EQ(local_storage_of_model_branch.branch_id, branch_id);
  local_storage_of_model_branch.branch_id = -1;
  CHECK(0);
}
