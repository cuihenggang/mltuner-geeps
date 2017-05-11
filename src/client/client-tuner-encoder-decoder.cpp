/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

// Encode and decode messages to/from the scheduler

#include <string>
#include <vector>

#include "common/work-puller.hpp"
#include "common/background-worker.hpp"
#include "client-tuner-encoder-decoder.hpp"

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using boost::shared_ptr;
using boost::make_shared;

void ClientTunerEncoder::worker_started() {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(cs_worker_started_msg_t));
  cs_worker_started_msg_t *cs_worker_started_msg =
      reinterpret_cast<cs_worker_started_msg_t *>(msgs[0].data());
  cs_worker_started_msg->cmd = WORKER_STARTED;
  cs_worker_started_msg->client_id = client_id;

  router_handler->send_to(scheduler_name, msgs);
}

void ClientTunerEncoder::report_progress(
    iter_t clock, int branch_id, double progress) {
  vector<ZmqPortableBytes> msgs;
  msgs.resize(1);

  msgs[0].init_size(sizeof(report_progress_msg_t));
  report_progress_msg_t *report_progress_msg =
      reinterpret_cast<report_progress_msg_t *>(msgs[0].data());
  report_progress_msg->cmd = REPORT_PROGRESS;
  report_progress_msg->client_id = client_id;
  report_progress_msg->clock = clock;
  report_progress_msg->branch_id = branch_id;
  report_progress_msg->progress = progress;

  router_handler->send_to(scheduler_name, msgs);
}


TunerClientDecoder::TunerClientDecoder(
        uint channel_id, shared_ptr<zmq::context_t> ctx,
        ClientLib *client_lib, bool work_in_bg,
        uint numa_node_id, const Config& config)
  : channel_id(channel_id), zmq_ctx(ctx),
    client_lib(client_lib), work_in_background(work_in_bg),
    numa_node_id(numa_node_id), config(config) {
  if (work_in_background) {
    /* Start background worker thread */
    string endpoint = "inproc://bg-recv-worker";
    shared_ptr<WorkPuller> work_puller =
        make_shared<WorkPuller>(zmq_ctx, endpoint);
    BackgroundWorker::WorkerCallback worker_callback =
        bind(&TunerClientDecoder::decode_msg, this, _1);
    BackgroundWorker bg_worker(work_puller);
    bg_worker.add_callback(DECODE_CMD, worker_callback);
    bg_decode_worker_thread = make_shared<boost::thread>(bg_worker);

    /* Init work pusher */
    decode_work_pusher = make_shared<WorkPusher>(zmq_ctx, endpoint);
  }
}

void TunerClientDecoder::schedule_branches(vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 3);
  CHECK_EQ(args[0].size(), sizeof(schedule_branches_msg_t));
  schedule_branches_msg_t *schedule_branches_msg =
      reinterpret_cast<schedule_branches_msg_t *>(args[0].data());
  uint batch_size = schedule_branches_msg->batch_size;

  CHECK_EQ(args[1].size(), sizeof(iter_t) * batch_size);
  iter_t *clocks = reinterpret_cast<iter_t *>(args[1].data());
  CHECK_EQ(args[2].size(), sizeof(int) * batch_size);
  int *branch_ids = reinterpret_cast<int *>(args[2].data());

  client_lib->recv_branch_schedules(batch_size, clocks, branch_ids);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void TunerClientDecoder::make_branch(vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 2);
  CHECK_EQ(args[0].size(), sizeof(make_branch_msg_t));
  make_branch_msg_t *make_branch_msg =
      reinterpret_cast<make_branch_msg_t *>(args[0].data());
  int branch_id = make_branch_msg->branch_id;
  int flag = make_branch_msg->flag;
  int parent_branch_id = make_branch_msg->parent_branch_id;
  int clock_to_make = make_branch_msg->clock_to_make;

  Tunable tunable = *(reinterpret_cast<Tunable *>(args[1].data()));

  client_lib->recv_make_branch(
      branch_id, tunable, flag, parent_branch_id, clock_to_make, args);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void TunerClientDecoder::inactivate_branch(vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(inactivate_branch_msg_t));
  inactivate_branch_msg_t *inactivate_branch_msg =
      reinterpret_cast<inactivate_branch_msg_t *>(args[0].data());
  int branch_id = inactivate_branch_msg->branch_id;
  int clock_to_happen = inactivate_branch_msg->clock_to_happen;

  client_lib->recv_inactivate_branch(branch_id, clock_to_happen, args);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void TunerClientDecoder::save_branch(vector<ZmqPortableBytes>& args) {
  CHECK_EQ(args.size(), 1);
  CHECK_EQ(args[0].size(), sizeof(save_branch_msg_t));
  save_branch_msg_t *save_branch_msg =
      reinterpret_cast<save_branch_msg_t *>(args[0].data());
  int branch_id = save_branch_msg->branch_id;
  int clock_to_happen = save_branch_msg->clock_to_happen;

  client_lib->recv_save_branch(branch_id, clock_to_happen, args);

  for (uint i = 0; i < args.size(); i++) {
    args[i].close();
  }
}

void TunerClientDecoder::decode_msg(vector<ZmqPortableBytes>& msgs) {
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
  case SCHEDULE_BRANCHES:
    schedule_branches(msgs);
    break;
  case MAKE_BRANCH:
    make_branch(msgs);
    break;
  case INACTIVATE_BRANCH:
    inactivate_branch(msgs);
    break;
  default:
    cerr << "Client received unknown command!" << endl;
    assert(0);
  }
}

void TunerClientDecoder::router_callback(const string& src,
    vector<ZmqPortableBytes>& msgs) {
  /* The "src" field is not send to the background worker, because we don't
   * want to construct another string vector object. */
  if (work_in_background) {
    /* Push to the background thread */
    decode_work_pusher->push_work(DECODE_CMD, msgs);
  } else {
    /* Do it myself */
    decode_msg(msgs);
  }
}

RouterHandler::RecvCallback TunerClientDecoder::get_recv_callback() {
  return bind(&TunerClientDecoder::router_callback, this, _1, _2);
}

void TunerClientDecoder::stop_decoder() {
  if (work_in_background) {
    /* Shut down background worker thread */
    vector<ZmqPortableBytes> args;  /* Args is empty */
    decode_work_pusher->push_work(BackgroundWorker::STOP_CMD, args);
    (*bg_decode_worker_thread).join();

    /* Set "work_in_background" to false, so that we won't do that again. */
    work_in_background = false;
  }
}

TunerClientDecoder::~TunerClientDecoder() {
  stop_decoder();
}
