#ifndef __client_tuner_encoder_decoder_hpp__
#define __client_tuner_encoder_decoder_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

// Encode and decode messages to/fro the server

#include <vector>
#include <string>

#include "common/wire-protocol.hpp"
#include "common/common-utils.hpp"
#include "common/router-handler.hpp"
#include "client/clientlib.hpp"

using std::vector;

class ClientTunerEncoder {
  boost::shared_ptr<RouterHandler> router_handler;
  uint client_id;
  std::string scheduler_name;

 public:
  ClientTunerEncoder(
      boost::shared_ptr<RouterHandler> router_handler, uint client_id,
      const Config& config) :
          router_handler(router_handler), client_id(client_id) {
    scheduler_name = "scheduler";
  }
  void worker_started();
  void report_progress(iter_t clock, int branch_id, double progress);
};

class TunerClientDecoder {
  static const uint DECODE_CMD = 1;

  uint channel_id;
  boost::shared_ptr<zmq::context_t> zmq_ctx;
  ClientLib *client_lib;
  bool work_in_background;

  boost::shared_ptr<boost::thread> bg_decode_worker_thread;
  boost::shared_ptr<WorkPusher> decode_work_pusher;

  uint numa_node_id;
  Config config;

 public:
  TunerClientDecoder(
      uint channel_id, boost::shared_ptr<zmq::context_t> ctx,
      ClientLib *client_lib,
      bool work_in_bg, uint numa_node_id,
      const Config& config);
  ~TunerClientDecoder();
  void schedule_branches(vector<ZmqPortableBytes>& args);
  void make_branch(vector<ZmqPortableBytes>& args);
  void inactivate_branch(vector<ZmqPortableBytes>& args);
  void save_branch(vector<ZmqPortableBytes>& args);
  void decode_msg(vector<ZmqPortableBytes>& args);
  void router_callback(const string& src, vector<ZmqPortableBytes>& msgs);
  RouterHandler::RecvCallback get_recv_callback();
  void stop_decoder();
};

#endif  // defined __client_tuner_encoder_decoder_hpp__
