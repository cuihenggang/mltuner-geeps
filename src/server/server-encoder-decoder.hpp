#ifndef __server_encoder_decoder_hpp__
#define __server_encoder_decoder_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <set>
#include <map>

#include <boost/shared_ptr.hpp>

#include "common/wire-protocol.hpp"
#include "common/router-handler.hpp"

using boost::shared_ptr;

class TabletStorage;
class MetadataServer;
class BranchScheduler;

/* Encodes messages to client */
class ServerClientEncode {
  shared_ptr<RouterHandler> router_handler;
  vector<string> client_names;
  cudaStream_t cuda_stream;
  cublasHandle_t cublas_handle;
  uint num_machines;
  uint server_id;
  Config config;

 public:
  explicit ServerClientEncode(
      shared_ptr<RouterHandler> router_handler,
      cudaStream_t cuda_stream, cublasHandle_t cublas_handle,
      uint num_machines, uint server_id, const Config& config)
        : router_handler(router_handler),
          cuda_stream(cuda_stream), cublas_handle(cublas_handle),
          num_machines(num_machines), server_id(server_id), config(config) {
    for (uint i = 0; i < num_machines; i++) {
      std::string client_name("local");
      if (!config.local_opt || i != server_id) {
        client_name = (boost::format("client-%i") % i).str();
      }
      client_names.push_back(client_name);
    }
  }
  void find_row(
      uint client_id, table_id_t table, row_idx_t row, uint server_id);
  void read_row_batch_reply(
      uint client_id, uint server_id, iter_t data_age, iter_t self_clock,
      int branch_id, iter_t internal_data_age, iter_t self_internal_clock,
      uint table_id, RowKey *row_keys, RowData *row_data, uint batch_size);
  // void clock(uint server_id, iter_t clock);
  void get_stats(uint client_id, const string& stats);
  string get_router_stats();
  void server_started(uint server_id);
};

/* Decodes messages from client */
class ClientServerDecode {
  uint channel_id;
  shared_ptr<TabletStorage> storage;
  shared_ptr<MetadataServer> metadata_server;

 public:
  explicit ClientServerDecode(
      uint channel_id,
      shared_ptr<TabletStorage> storage,
      shared_ptr<MetadataServer> metadata_server);
  void find_row(const string& src, vector<ZmqPortableBytes>& msgs);
  void clock(const string& src, vector<ZmqPortableBytes>& msgs);
  void clock_with_updates_batch(
      const string& src, vector<ZmqPortableBytes>& msgs);
  void read_row_batch(const string& src, vector<ZmqPortableBytes>& msgs);
  void add_access_info(const string& src, vector<ZmqPortableBytes>& msgs);
  void get_stats(const string& src, vector<ZmqPortableBytes>& msgs);
  void worker_started(const string& src, vector<ZmqPortableBytes>& msgs);
  void make_branch(const string& src, vector<ZmqPortableBytes>& msgs);
  void inactivate_branch(
      const string& src, vector<ZmqPortableBytes>& msgs);
  void decode_msg(const string& src, vector<ZmqPortableBytes>& msgs);
  void router_callback(const string& src, vector<ZmqPortableBytes>& msgs);

  RouterHandler::RecvCallback get_recv_callback();
};

#endif  // defined __server_encoder_decoder_hpp__
