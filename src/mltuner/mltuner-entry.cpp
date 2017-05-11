/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <boost/format.hpp>
#include <boost/make_shared.hpp>

#include <string>
#include <vector>

#include "mltuner-entry.hpp"
#include "mltuner-encoder-decoder.hpp"
#include "mltuner.hpp"

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using boost::format;
using boost::shared_ptr;
using boost::make_shared;

class MltunerCommunicator : public Communicator {
 private:
  shared_ptr<MltunerEncoder> encoder;

 public:
  MltunerCommunicator(shared_ptr<MltunerEncoder> encoder)
      : encoder(encoder) {}

  virtual void make_branch(
      int branch_id, const Tunable& tunable, int flag,
      int parent_branch_id, int clock_to_happen) {
    encoder->make_branch(
        branch_id, tunable, flag, parent_branch_id, clock_to_happen);
  }

  virtual void inactivate_branch(int branch_id, int clock_to_happen) {
    encoder->inactivate_branch(branch_id, clock_to_happen);
  }

  virtual void schedule_branches(
      uint batch_size, const int *clocks, const int *branch_ids) {
    encoder->schedule_branches(batch_size, clocks, branch_ids);
  }
};

void MltunerEntry::mltuner_entry(
    uint num_processes, shared_ptr<zmq::context_t> zmq_ctx,
    const Config& config) {
  uint port = config.tcp_base_port + config.num_channels;
  string request_url = "tcp://*:" + boost::lexical_cast<std::string>(port);

  vector<string> connect_list;   /* Empty connect to */
  vector<string> bind_list;
  bind_list.push_back(request_url);
  string server_name = "scheduler";
  uint channel_id = 0;
#if defined(ITERSTORE)
  uint numa_node_id = 0;
  shared_ptr<RouterHandler> router_handler = make_shared<RouterHandler>(
      channel_id, zmq_ctx, connect_list, bind_list, server_name,
      numa_node_id, config);
#else
  shared_ptr<RouterHandler> router_handler = make_shared<RouterHandler>(
      channel_id, zmq_ctx, connect_list, bind_list, server_name,
      config);
#endif

  shared_ptr<MltunerEncoder> encoder = make_shared<MltunerEncoder>(
      router_handler, num_processes, config);

  shared_ptr<MltunerCommunicator> mltuner_communicator =
      make_shared<MltunerCommunicator>(encoder);
  shared_ptr<Mltuner> mltuner = make_shared<Mltuner>(
      mltuner_communicator, config);

  MltunerDecoder decoder(mltuner);

  router_handler->do_handler(decoder.get_recv_callback());
}
