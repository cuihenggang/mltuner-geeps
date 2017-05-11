#ifndef __work_pusher_hpp__
#define __work_pusher_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <boost/thread.hpp>

#include <vector>
#include <string>

#include "common/zmq-util.hpp"

class WorkPusher {
  boost::shared_ptr<zmq::context_t> zmq_ctx;
  boost::shared_ptr<zmq::socket_t> socket;
  std::string endpoint;
  boost::mutex mutex;

 public:
  WorkPusher(
      boost::shared_ptr<zmq::context_t> ctx, std::string connection_endpoint);
  void push_work(uint cmd, std::vector<ZmqPortableBytes>& args);
  void push_work(uint cmd);
};

#endif  // defined __work_pusher_hpp__
