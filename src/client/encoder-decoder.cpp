/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

// Encode and decode messages to/fro the server

#include <string>
#include <vector>

#include "common/work-puller.hpp"
#include "common/background-worker.hpp"
#include "common/row-op-util.hpp"
#include "encoder-decoder.hpp"

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using boost::shared_ptr;
using boost::make_shared;

void ClientServerEncode::find_row(
      table_id_t table, row_idx_t row, uint metadata_sever_id) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(cs_find_row_msg_t));
  cs_find_row_msg_t *cs_find_row_msg =
      reinterpret_cast<cs_find_row_msg_t *>(msgs[0].data());
  cs_find_row_msg->cmd = FIND_ROW;
  cs_find_row_msg->client_id = client_id;
  cs_find_row_msg->table = table;
  cs_find_row_msg->row = row;

  /* Currently, the tablet servers are also metadata servers */
  string metadata_sever_name = server_names[metadata_sever_id];

  router_handler->send_to(metadata_sever_name, msgs);
}

void ClientServerEncode::read_row_batch(
      uint server_id, RowKeys& row_keys, iter_t data_age,
      bool prioritized) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(2);

  msgs[0].init_size(sizeof(cs_read_row_batch_msg_t));
  cs_read_row_batch_msg_t *cs_read_row_batch_msg =
      reinterpret_cast<cs_read_row_batch_msg_t *>(msgs[0].data());
  cs_read_row_batch_msg->cmd = READ_ROW_BATCH;
  cs_read_row_batch_msg->client_id = client_id;
  cs_read_row_batch_msg->data_age = data_age;
  cs_read_row_batch_msg->prioritized = prioritized;

  msgs[1].pack_vector<RowKey>(row_keys);

  CHECK_LT(server_id, server_names.size());
  string tablet_name = server_names[server_id];
  router_handler->send_to(tablet_name, msgs);
}

void ClientServerEncode::clock_broadcast(
    iter_t clock, uint table_id, int branch_id, iter_t internal_clock) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(cs_clock_msg_t));
  cs_clock_msg_t *cs_clock_msg =
      reinterpret_cast<cs_clock_msg_t *>(msgs[0].data());
  cs_clock_msg->cmd = CLOCK;
  cs_clock_msg->client_id = client_id;
  cs_clock_msg->clock = clock;
  cs_clock_msg->table_id = table_id;
  cs_clock_msg->branch_id = branch_id;
  cs_clock_msg->internal_clock = internal_clock;

  /* Broadcast to all tablet servers */
  router_handler->send_to(server_names, msgs);
}

void ClientServerEncode::clock_with_updates_batch(
      uint server_id, iter_t clock, uint table_id,
      int branch_id, iter_t internal_clock,
      const RowOpVal *updates, const RowKey *row_keys, uint batch_size) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(3);

  msgs[0].init_size(sizeof(cs_clock_with_updates_batch_msg_t));
  cs_clock_with_updates_batch_msg_t *cs_clock_with_updates_batch_msg =
      reinterpret_cast<cs_clock_with_updates_batch_msg_t *>(msgs[0].data());
  cs_clock_with_updates_batch_msg->cmd = CLOCK_WITH_UPDATES_BATCH;
  cs_clock_with_updates_batch_msg->client_id = client_id;
  cs_clock_with_updates_batch_msg->clock = clock;
  cs_clock_with_updates_batch_msg->table_id = table_id;
  cs_clock_with_updates_batch_msg->branch_id = branch_id;
  cs_clock_with_updates_batch_msg->internal_clock = internal_clock;

  msgs[1].pack_memory(row_keys, batch_size * sizeof(RowKey));
  msgs[2].pack_memory(updates, batch_size * sizeof(RowOpVal));

  CHECK_LT(server_id, server_names.size());
  string tablet_name = server_names[server_id];
  router_handler->send_to(tablet_name, msgs);
}

void ClientServerEncode::clock_with_updates_batch(
      uint server_id, iter_t clock, uint table_id,
      int branch_id, int internal_clock,
      const RowOpVal *updates0, const RowKey *row_keys0, uint batch_size0,
      const RowOpVal *updates1, const RowKey *row_keys1, uint batch_size1) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(3);

  msgs[0].init_size(sizeof(cs_clock_with_updates_batch_msg_t));
  cs_clock_with_updates_batch_msg_t *cs_clock_with_updates_batch_msg =
      reinterpret_cast<cs_clock_with_updates_batch_msg_t *>(msgs[0].data());
  cs_clock_with_updates_batch_msg->cmd = CLOCK_WITH_UPDATES_BATCH;
  cs_clock_with_updates_batch_msg->client_id = client_id;
  cs_clock_with_updates_batch_msg->clock = clock;
  cs_clock_with_updates_batch_msg->table_id = table_id;
  cs_clock_with_updates_batch_msg->branch_id = branch_id;
  cs_clock_with_updates_batch_msg->internal_clock = internal_clock;

  msgs[1].pack_memory(row_keys0, batch_size0 * sizeof(RowKey),
      row_keys1, batch_size1 * sizeof(RowKey));
  msgs[2].pack_memory(updates0, batch_size0 * sizeof(RowOpVal),
      updates1, batch_size1 * sizeof(RowOpVal));

  CHECK_LT(server_id, server_names.size());
  string tablet_name = server_names[server_id];
  router_handler->send_to(tablet_name, msgs);
}

void ClientServerEncode::add_access_info(
      uint metadata_server_id, const std::vector<RowAccessInfo>& access_info) {
  /* Currently, the tablet servers are also metadata servers */
  CHECK_LT(metadata_server_id, server_names.size());
  string metadata_sever_name = server_names[metadata_server_id];

  vector<ZmqPortableBytes> msgs;
  msgs.resize(2);

  msgs[0].init_size(sizeof(cs_add_access_info_msg_t));
  cs_add_access_info_msg_t *cs_add_access_info_msg =
      reinterpret_cast<cs_add_access_info_msg_t *>(msgs[0].data());
  cs_add_access_info_msg->cmd = ADD_ACCESS_INFO;
  cs_add_access_info_msg->client_id = client_id;

  msgs[1].pack_vector<RowAccessInfo>(access_info);

  router_handler->send_to(metadata_sever_name, msgs);
}

void ClientServerEncode::get_stats(uint server_id) {
  CHECK_LT(server_id, server_names.size());
  string sever_name = server_names[server_id];

  vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(cs_get_stats_msg_t));
  cs_get_stats_msg_t *cs_get_stats_msg =
      reinterpret_cast<cs_get_stats_msg_t *>(msgs[0].data());
  cs_get_stats_msg->cmd = GET_STATS;
  cs_get_stats_msg->client_id = client_id;

  router_handler->send_to(sever_name, msgs);
}

void ClientServerEncode::worker_started() {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(cs_worker_started_msg_t));
  cs_worker_started_msg_t *cs_worker_started_msg =
      reinterpret_cast<cs_worker_started_msg_t *>(msgs[0].data());
  cs_worker_started_msg->cmd = WORKER_STARTED;
  cs_worker_started_msg->client_id = client_id;

  router_handler->send_to(server_names, msgs);
}

void ClientServerEncode::broadcast_msg(vector<ZmqPortableBytes>& msgs) {
  /* Make a copy of the message for sending */
  vector<ZmqPortableBytes> msgs_copy(msgs.size());
  for (uint j = 0; j < msgs_copy.size(); j ++) {
    msgs_copy[j].copy(msgs[j]);
  }
  /* Broadcast to all servers */
  router_handler->send_to(server_names, msgs_copy);
}

ServerClientDecode::ServerClientDecode(
    uint channel_id, shared_ptr<zmq::context_t> ctx,
    ClientLib *client_lib, bool work_in_bg,
    const Config& config) :
      channel_id(channel_id), zmq_ctx(ctx),
      client_lib(client_lib), work_in_background(work_in_bg),
      config(config) {
  if (work_in_background) {
    /* Start background worker thread */
    string endpoint = "inproc://bg-recv-worker";
    shared_ptr<WorkPuller> work_puller =
        make_shared<WorkPuller>(zmq_ctx, endpoint);
    BackgroundWorker::WorkerCallback worker_callback =
        bind(&ServerClientDecode::decode_msg, this, _1);
    BackgroundWorker bg_worker(work_puller);
    bg_worker.add_callback(DECODE_CMD, worker_callback);
    bg_decode_worker_thread = make_shared<boost::thread>(bg_worker);

    /* Init work pusher */
    decode_work_pusher = make_shared<WorkPusher>(zmq_ctx, endpoint);
  }
}

void ServerClientDecode::find_row(vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(sc_find_row_msg_t));

  sc_find_row_msg_t *sc_find_row_msg =
    reinterpret_cast<sc_find_row_msg_t *>(args[0].data());
  table_id_t table = sc_find_row_msg->table;
  row_idx_t row = sc_find_row_msg->row;
  uint server_id = sc_find_row_msg->server_id;
  client_lib->find_row_cbk(table, row, server_id);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ServerClientDecode::read_row_batch(vector<ZmqPortableBytes>& args) {
  CHECK_GE(args.size(), 3);
  CHECK_EQ(args[0].size(), sizeof(sc_read_row_batch_msg_t));
  sc_read_row_batch_msg_t *sc_read_row_batch_msg =
      reinterpret_cast<sc_read_row_batch_msg_t *>(args[0].data());
  uint server_id = sc_read_row_batch_msg->server_id;
  iter_t data_age = sc_read_row_batch_msg->data_age;
  iter_t self_clock = sc_read_row_batch_msg->self_clock;
  int branch_id = sc_read_row_batch_msg->branch_id;
  iter_t internal_data_age = sc_read_row_batch_msg->internal_data_age;
  iter_t internal_self_clock = sc_read_row_batch_msg->internal_self_clock;
  uint table_id = sc_read_row_batch_msg->table_id;

  RowKey *row_keys =
      reinterpret_cast<RowKey *>(args[1].data());
  uint batch_size = args[1].size() / sizeof(RowKey);
  RowData *row_data =
      reinterpret_cast<RowData *>(args[2].data());
  CHECK_EQ(batch_size, args[2].size() / sizeof(RowData));
  client_lib->server_clock(channel_id, server_id, data_age, table_id);
  client_lib->recv_row_batch(
      channel_id, server_id, data_age, self_clock,
      branch_id, internal_data_age, internal_self_clock,
      table_id, row_keys, row_data, batch_size);
  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ServerClientDecode::clock(vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(sc_clock_msg_t));

  sc_clock_msg_t *sc_clock_msg =
      reinterpret_cast<sc_clock_msg_t *>(args[0].data());
  uint server_id = sc_clock_msg->server_id;
  iter_t clock = sc_clock_msg->clock;
  uint table_id = sc_clock_msg->table_id;

  client_lib->server_clock(channel_id, server_id, clock, table_id);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ServerClientDecode::server_started(vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(sc_server_started_msg_t));

  sc_server_started_msg_t *sc_server_started_msg =
      reinterpret_cast<sc_server_started_msg_t *>(args[0].data());
  uint server_id = sc_server_started_msg->server_id;

  client_lib->server_started(channel_id, server_id);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ServerClientDecode::get_stats(vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 2);

  string stats;
  args[1].unpack_string(stats);
  client_lib->get_stats_cbk(stats);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void ServerClientDecode::decode_msg(vector<ZmqPortableBytes>& msgs) {
  CHECK_GE(msgs.size(), 1);
  CHECK_GE(msgs[0].size(), sizeof(command_t));
  command_t cmd;
  msgs[0].unpack<command_t>(cmd);
  switch (cmd) {
  case FIND_ROW:
    find_row(msgs);
    break;
  case READ_ROW_BATCH:
    read_row_batch(msgs);
    break;
  case CLOCK:
    clock(msgs);
    break;
  case GET_STATS:
    get_stats(msgs);
    break;
  case SERVER_STARTED:
    server_started(msgs);
    break;
  default:
    CHECK(0) << "Client received unknown command!";
  }
}

void ServerClientDecode::router_callback(const string& src,
    vector<ZmqPortableBytes>& msgs) {
  /* The "src" field is not send to the background worker, because we don't
   * want to construct another string vector object.
   */
  if (work_in_background) {
    /* Push to the background thread */
    decode_work_pusher->push_work(DECODE_CMD, msgs);
  } else {
    /* Do it myself */
    decode_msg(msgs);
  }
}

RouterHandler::RecvCallback ServerClientDecode::get_recv_callback() {
  return bind(&ServerClientDecode::router_callback, this, _1, _2);
}

void ServerClientDecode::stop_decoder() {
  if (work_in_background) {
    /* Shut down background worker thread */
    vector<ZmqPortableBytes> args;  /* Args is empty */
    decode_work_pusher->push_work(BackgroundWorker::STOP_CMD, args);
    (*bg_decode_worker_thread).join();

    /* Set "work_in_background" to false, so that we won't do that again. */
    work_in_background = false;
  }
}

ServerClientDecode::~ServerClientDecode() {
  stop_decoder();
}
