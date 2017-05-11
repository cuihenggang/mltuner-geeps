/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <utility>
#include <string>
#include <vector>

#include "mltuner-encoder-decoder.hpp"
#include "common/portable-bytes.hpp"

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

MltunerDecoder::MltunerDecoder(shared_ptr<Mltuner> mltuner) :
    mltuner(mltuner) {
}

void MltunerDecoder::worker_started(
    const string& src, vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(cs_worker_started_msg_t));
  cs_worker_started_msg_t *cs_worker_started_msg =
      reinterpret_cast<cs_worker_started_msg_t *>(args[0].data());
  uint client_id = cs_worker_started_msg->client_id;

  mltuner->recv_worker_started(client_id);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void MltunerDecoder::report_progress(
    const string& src, vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(report_progress_msg_t));
  report_progress_msg_t *cs_report_progress_msg =
      reinterpret_cast<report_progress_msg_t *>(args[0].data());
  uint client_id = cs_report_progress_msg->client_id;
  iter_t clock = cs_report_progress_msg->clock;
  int branch_id = cs_report_progress_msg->branch_id;
  double progress = cs_report_progress_msg->progress;

  mltuner->recv_worker_progress(client_id, clock, branch_id, progress);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void MltunerDecoder::decode_msg(
    const string& src, vector<ZmqPortableBytes>& msgs) {
  if (msgs.size() < 1) {
    cerr << "Received message has parts missing!" << endl;
    assert(msgs.size() >= 1);
  }

  if (msgs[0].size() < sizeof(command_t)) {
    cerr << "Received message misformed: size = " << msgs[0].size() << endl;
    assert(msgs[0].size() >= sizeof(command_t));
  }
  command_t cmd;
  msgs[0].unpack<command_t>(cmd);
  switch (cmd) {
  case WORKER_STARTED:
    // cout << "server receives WORKER_STARTED\n";
    worker_started(src, msgs);
    break;
  case REPORT_PROGRESS:
    // cout << "server receives REPORT_PROGRESS\n";
    report_progress(src, msgs);
    break;
  default:
    cerr << "Server received unknown command: " << static_cast<int>(cmd)
         << " size: " << msgs[0].size()
         << endl;
    assert(0);
  }
}

void MltunerDecoder::router_callback(const string& src,
    vector<ZmqPortableBytes>& msgs) {
  decode_msg(src, msgs);
}

RouterHandler::RecvCallback MltunerDecoder::get_recv_callback() {
  return bind(&MltunerDecoder::router_callback, this, _1, _2);
}


// void MltunerEncoder::server_started(uint server_id) {
  // vector<ZmqPortableBytes> msgs;
  // msgs.resize(1);

  // msgs[0].init_size(sizeof(sc_server_started_msg_t));
  // sc_server_started_msg_t *sc_server_started_msg =
      // reinterpret_cast<sc_server_started_msg_t *>(msgs[0].data());
  // sc_server_started_msg->cmd = SERVER_STARTED;
  // sc_server_started_msg->server_id = server_id;

  // /* Broadcast to all clients */
  // router_handler->direct_send_to(client_names, msgs);
// }

void MltunerEncoder::schedule_branches(
      uint batch_size, const iter_t *clocks, const int *branch_ids) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(3);

  msgs[0].init_size(sizeof(schedule_branches_msg_t));
  schedule_branches_msg_t *schedule_branches_msg =
      reinterpret_cast<schedule_branches_msg_t *>(msgs[0].data());
  schedule_branches_msg->cmd = SCHEDULE_BRANCHES;
  schedule_branches_msg->batch_size = batch_size;

  msgs[1].pack_memory(clocks, sizeof(iter_t) * batch_size);
  msgs[2].pack_memory(branch_ids, sizeof(int) * batch_size);

  CHECK(batch_size);

  /* Broadcast to all clients */
  router_handler->direct_send_to(client_names, msgs);
}

void MltunerEncoder::make_branch(
      int branch_id, const Tunable& tunable, int flag,
      int parent_branch_id, iter_t clock_to_make) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(2);

  msgs[0].init_size(sizeof(make_branch_msg_t));
  make_branch_msg_t *make_branch_msg =
      reinterpret_cast<make_branch_msg_t *>(msgs[0].data());
  make_branch_msg->cmd = MAKE_BRANCH;
  make_branch_msg->branch_id = branch_id;
  make_branch_msg->flag = flag;
  make_branch_msg->parent_branch_id = parent_branch_id;
  make_branch_msg->clock_to_make = clock_to_make;

  msgs[1].pack_memory(&tunable, sizeof(Tunable));

  /* Broadcast to all clients */
  router_handler->direct_send_to(client_names, msgs);
}

void MltunerEncoder::inactivate_branch(
    int branch_id, iter_t clock_to_happen) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(inactivate_branch_msg_t));
  inactivate_branch_msg_t *inactivate_branch_msg =
      reinterpret_cast<inactivate_branch_msg_t *>(msgs[0].data());
  inactivate_branch_msg->cmd = INACTIVATE_BRANCH;
  inactivate_branch_msg->branch_id = branch_id;
  inactivate_branch_msg->clock_to_happen = clock_to_happen;

  /* Broadcast to all clients */
  router_handler->direct_send_to(client_names, msgs);
}

void MltunerEncoder::save_branch(
    int branch_id, iter_t clock_to_happen) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(save_branch_msg_t));
  save_branch_msg_t *save_branch_msg =
      reinterpret_cast<save_branch_msg_t *>(msgs[0].data());
  save_branch_msg->cmd = SAVE_BRANCH;
  save_branch_msg->branch_id = branch_id;
  save_branch_msg->clock_to_happen = clock_to_happen;

  /* Broadcast to all clients */
  router_handler->direct_send_to(client_names, msgs);
}
