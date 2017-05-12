#ifndef __mltuner_encoder_decoder_hpp__
#define __mltuner_encoder_decoder_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

// Lazy table shards server

#include <tbb/tick_count.h>

#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/format.hpp>

#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <set>
#include <map>

#include "geeps.hpp"
#include "mltuner.hpp"
// #include "include/lazy-table-module-types.hpp"
#include "common/wire-protocol.hpp"
#include "common/zmq-portable-bytes.hpp"
#include "common/router-handler.hpp"

using std::string;
using std::vector;

using boost::shared_ptr;

/* Encodes messages to client */
class MltunerEncoder {
  shared_ptr<RouterHandler> router_handler;
  vector<string> client_names;

 public:
  explicit MltunerEncoder(
      shared_ptr<RouterHandler> rh, uint num_machines,
      const Config& config)
        : router_handler(rh) {
    for (uint i = 0; i < num_machines; i++) {
      std::string cname = (boost::format("client-%i") % i).str();
      client_names.push_back(cname);
    }
  }
  void schedule_branches(
      uint batch_size, const iter_t *clocks, const int *branch_ids);
  void make_branch(
      int branch_id, const Tunable& tunable, int flag,
      int parent_branch_id, iter_t clock_to_make);
  void inactivate_branch(int branch_id, iter_t clock_to_happen);
  void save_branch(int branch_id, iter_t clock_to_happen);
};

/* Decodes messages from client */
class MltunerDecoder {
  shared_ptr<Mltuner> mltuner;

 public:
  explicit MltunerDecoder(shared_ptr<Mltuner> mltuner);
  void worker_started(const string& src, vector<ZmqPortableBytes>& msgs);
  void report_progress(const string& src, vector<ZmqPortableBytes>& msgs);
  void decode_msg(const string& src, vector<ZmqPortableBytes>& msgs);
  void router_callback(const string& src, vector<ZmqPortableBytes>& msgs);

  RouterHandler::RecvCallback get_recv_callback();
};

#endif  // defined __mltuner_encoder_decoder_hpp__
