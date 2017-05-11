/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

// socket layer that handles ZMQ sockets, both in-process and over the net

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <tbb/tick_count.h>

#include <string>
#include <vector>

#include "router-handler.hpp"

using std::string;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using boost::shared_ptr;
using boost::make_shared;
using boost::thread;
using boost::bind;


int64_t msg_size(const vector<string>& msgs) {
  int64_t size = 0;
  for (uint i = 0; i < msgs.size(); i++) {
    size += msgs[i].size();
  }

  return size;
}

int64_t msg_size(vector<ZmqPortableBytes>& msgs) {
  int64_t size = 0;
  for (uint i = 0; i < msgs.size(); i++) {
    size += msgs[i].size();
  }

  return size;
}

void zmq_connect_with_retry(
    zmq::socket_t *socket, const char *connect_point) {
  CHECK(socket);
  while (true) {
    try {
      socket->connect(connect_point);
    } catch (...) {
      cerr << "Connection error for " << connect_point << ", try again...\n";
      sleep(2);
      continue;
    }
    return;
  }
}

RouterHandler::RouterHandler(uint channel_id,
    shared_ptr<zmq::context_t> ctx,
    const vector<string>& connect_list, const vector<string>& bind_list,
    const string& identity, const Config& config) :
      channel_id(channel_id), zmq_ctx(ctx),
      connect_to(connect_list), bind_to(bind_list), identity(identity),
      router_socket(*zmq_ctx, ZMQ_ROUTER), shutdown_socket(*zmq_ctx, ZMQ_PULL),
      pull_socket(*zmq_ctx, ZMQ_PULL), local_recv_socket(*zmq_ctx, ZMQ_PULL),
      config(config) {
  if (identity != "") {
    router_socket.setsockopt(ZMQ_IDENTITY, identity.c_str(),
        identity.size());
  }

  BOOST_FOREACH(string s, connect_to) {
    router_socket.connect(s.c_str());
  }
  BOOST_FOREACH(string s, bind_to) {
    router_socket.bind(s.c_str());
  }

  if (connect_to.size()) {
    /* CLient */
    client = true;
    shutdown_socket.bind("inproc://client-rh-shutdown");
    pull_socket.bind("inproc://inproc-client-msg");
    local_recv_socket.bind("inproc://inproc-local-client-recv");
  } else {
    /* Server */
    client = false;
    shutdown_socket.bind("inproc://server-rh-shutdown");
    pull_socket.bind("inproc://inproc-server-msg");
    local_recv_socket.bind("inproc://inproc-local-server-recv");
  }
}

string RouterHandler::get_stats() {
  return stats.to_json();
}

void RouterHandler::send_to(
      const string& dest, vector<ZmqPortableBytes>& msgs) {
  stats.total_send += msg_size(msgs);
  if (dest.compare("local") != 0) {
    if (snd_msg_socket.get() == NULL) {
      snd_msg_socket.reset(new zmq::socket_t(*zmq_ctx, ZMQ_PUSH));
      if (client) {
        snd_msg_socket->connect("inproc://inproc-client-msg");
      } else {
        snd_msg_socket->connect("inproc://inproc-server-msg");
      }
    }
    send_msg(*snd_msg_socket, dest, true);
    send_msgs(*snd_msg_socket, msgs);
  } else {
    stats.total_local_send += msg_size(msgs);
    if (local_snd_msg_socket.get() == NULL) {
      local_snd_msg_socket.reset(new zmq::socket_t(*zmq_ctx, ZMQ_PUSH));
      if (client) {
        local_snd_msg_socket->connect(
            "inproc://inproc-local-server-recv");
      } else {
        local_snd_msg_socket->connect(
            "inproc://inproc-local-client-recv");
      }
    }
    send_msgs(*local_snd_msg_socket, msgs);
  }
  for (uint j = 0; j < msgs.size(); j ++) {
    assert(!msgs[j].size());
  }
}

void RouterHandler::direct_send_to(
      const string& dest, vector<ZmqPortableBytes>& msgs) {
  stats.total_send += msg_size(msgs);
  if (dest.compare("local") != 0) {
    send_msg(router_socket, dest, true);
    send_msgs(router_socket, msgs);
  } else {
    stats.total_local_send += msg_size(msgs);
    if (local_snd_msg_socket.get() == NULL) {
      local_snd_msg_socket.reset(new zmq::socket_t(*zmq_ctx, ZMQ_PUSH));
      if (client) {
        local_snd_msg_socket->connect(
            "inproc://inproc-local-server-recv");
      } else {
        local_snd_msg_socket->connect(
            "inproc://inproc-local-client-recv");
      }
    }
    send_msgs(*local_snd_msg_socket, msgs);
  }
}

void RouterHandler::send_to(
    const vector<string>& dests, vector<ZmqPortableBytes>& msgs) {
  for (uint i = 0; i < dests.size() - 1; i++) {
    vector<ZmqPortableBytes> msgs_copy(msgs.size());
    for (uint j = 0; j < msgs.size(); j ++) {
      msgs_copy[j].copy(msgs[j]);
    }
    send_to(dests[i], msgs_copy);
  }
  /* Send the original message for the last one */
  send_to(dests[dests.size() - 1], msgs);
}

void RouterHandler::direct_send_to(
    const vector<string>& dests, vector<ZmqPortableBytes>& msgs) {
  for (uint i = 0; i < dests.size() - 1; i++) {
    vector<ZmqPortableBytes> msgs_copy(msgs.size());
    for (uint j = 0; j < msgs_copy.size(); j ++) {
      msgs_copy[j].copy(msgs[j]);
    }
    direct_send_to(dests[i], msgs_copy);
  }
  /* Send the original message for the last one */
  direct_send_to(dests[dests.size() - 1], msgs);
}

void RouterHandler::do_handler(RecvCallback recv_callback) {
  zmq::pollitem_t pollitems[4];
  pollitems[0].socket = shutdown_socket;
  pollitems[0].events = ZMQ_POLLIN;
  pollitems[1].socket = pull_socket;
  pollitems[1].events = ZMQ_POLLIN;
  pollitems[2].socket = router_socket;
  pollitems[2].events = ZMQ_POLLIN;
  pollitems[3].socket = local_recv_socket;
  pollitems[3].events = ZMQ_POLLIN;

  while (true) {
    try {
      zmq::poll(pollitems, 4);
    } catch(...) {
      cerr << "exception in router handler" << endl;
      break;
    }

    if (pollitems[0].revents) {
      /* Shut down router */
      string msg;
      recv_msg(shutdown_socket, msg);

      /* Break out the loop */
      break;
    }
    if (pollitems[1].revents) {
      /* Pass PULL to ROUTER */
      // tbb::tick_count timing_start = tbb::tick_count::now();
      forward_msgs(pull_socket, router_socket);
      // stats.pull_to_router_time +=
          // (tbb::tick_count::now() - timing_start).seconds();
    }
    if (pollitems[2].revents) {
      /* router_socket */
      // tbb::tick_count timing_start = tbb::tick_count::now();
      string src;
      bool more = recv_msg(router_socket, src);
      assert(more);
      vector<ZmqPortableBytes> msgs;
      recv_msgs(router_socket, msgs);
      stats.total_receive += msg_size(msgs);
      recv_callback(src, msgs);
      // stats.router_recv_time +=
          // (tbb::tick_count::now() - timing_start).seconds();
    }
    if (pollitems[3].revents) {
      /* local_recv_socket */
      // tbb::tick_count timing_start = tbb::tick_count::now();
      string src("local");
      vector<ZmqPortableBytes> msgs;
      recv_msgs(local_recv_socket, msgs);
      stats.total_local_receive += msg_size(msgs);
      stats.total_receive += msg_size(msgs);
      recv_callback(src, msgs);
      // stats.router_local_recv_time +=
          // (tbb::tick_count::now() - timing_start).seconds();
    }
  }
}

void RouterHandler::start_handler_thread(RecvCallback recv_callback) {
  handler_thread = make_shared <thread>(
      bind(&RouterHandler::do_handler, this, recv_callback));
  sleep(1);
      /* ZMQ 2.2 does not support connect before bind, so we have to wait
       * long enough for the handler thread to start working.
       */
}

void RouterHandler::stop_handler_thread() {
  zmq::socket_t s(*zmq_ctx, ZMQ_PUSH);
  if (client) {
    s.connect("inproc://client-rh-shutdown");
  } else {
    s.connect("inproc://server-rh-shutdown");
  }
  send_msg(s, "stop");
  (*handler_thread).join();
}

RouterHandler::~RouterHandler() {
}
