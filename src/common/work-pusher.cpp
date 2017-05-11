/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>

#include "work-pusher.hpp"

WorkPusher::WorkPusher(
    boost::shared_ptr<zmq::context_t> ctx, std::string connection_endpoint)
      : zmq_ctx(ctx), endpoint(connection_endpoint) {
  socket.reset(new zmq::socket_t(*zmq_ctx, ZMQ_PUSH));
  socket->connect(endpoint.c_str());
}

void WorkPusher::push_work(uint cmd, std::vector<ZmqPortableBytes>& args) {
  boost::unique_lock<boost::mutex> lock(mutex);
  bool more = args.size() > 0;
  ZmqPortableBytes msg;
  msg.pack<uint>(cmd);
  send_msg(*socket, msg, more);
  if (more) {
    send_msgs(*socket, args);
  }
}

void WorkPusher::push_work(uint cmd) {
  std::vector<ZmqPortableBytes> args(0);    /* Empty args */
  push_work(cmd, args);
}

