#ifndef __wire_protocol_hpp__
#define __wire_protocol_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

// Common data types shared by the client and the server

#include <stdint.h>

#include <vector>

#include "lazytable-user-defined-types.hpp"

enum Command {
  FIND_ROW,
  READ_ROW_BATCH,
  CLOCK,
  CLOCK_WITH_UPDATES_BATCH,
  ADD_ACCESS_INFO,
  GET_STATS,
  WORKER_STARTED,
  SERVER_STARTED,
  REPORT_PROGRESS,
  SCHEDULE_BRANCHES,
  MAKE_BRANCH,
  INACTIVATE_BRANCH,
  SAVE_BRANCH,
  SHUTDOWN
};
typedef uint8_t command_t;

struct RowAccessInfo {
  table_id_t tid;     /* Table ID */
  row_idx_t rid;      /* Row ID */
  uint32_t nr_read;   /* Read frequency */
  uint32_t nr_write;  /* Write frequency */
};

struct RowKey {
  table_id_t table;
  row_idx_t row;
  RowKey(table_id_t table_i = 0, row_idx_t row_i = 0) :
    table(table_i), row(row_i) {}
};
typedef std::vector<RowKey> RowKeys;

struct cs_find_row_msg_t {
  command_t cmd;
  uint32_t client_id;
  table_id_t table;
  row_idx_t row;
};

struct cs_read_row_batch_msg_t {
  command_t cmd;
  uint32_t client_id;
  iter_t data_age;
  bool prioritized;
};

struct cs_clock_msg_t {
  command_t cmd;
  uint32_t client_id;
  iter_t clock;
  uint32_t table_id;
  int branch_id;
  iter_t internal_clock;
};

struct cs_clock_with_updates_batch_msg_t {
  command_t cmd;
  uint32_t client_id;
  iter_t clock;
  uint32_t table_id;
  int branch_id;
  iter_t internal_clock;
};

struct cs_add_access_info_msg_t {
  command_t cmd;
  uint32_t client_id;
};

struct cs_get_stats_msg_t {
  command_t cmd;
  uint32_t client_id;
};

struct cs_worker_started_msg_t {
  command_t cmd;
  uint32_t client_id;
};


struct sc_clock_msg_t {
  command_t cmd;
  uint32_t server_id;
  iter_t clock;
  uint32_t table_id;
};

struct sc_find_row_msg_t {
  command_t cmd;
  table_id_t table;
  row_idx_t row;
  uint32_t server_id;
};

struct sc_read_row_batch_msg_t {
  command_t cmd;
  uint32_t server_id;
  iter_t data_age;
  iter_t self_clock;
  int branch_id;
  iter_t internal_data_age;
  iter_t internal_self_clock;
  uint32_t table_id;
};

struct sc_server_started_msg_t {
  command_t cmd;
  uint32_t server_id;
};

struct sc_get_stats_msg_t {
  command_t cmd;
};


struct report_progress_msg_t {
  command_t cmd;
  uint32_t client_id;
  iter_t clock;
  int branch_id;
  double progress;
};

struct schedule_branches_msg_t {
  command_t cmd;
  uint32_t batch_size;
};

struct make_branch_msg_t {
  command_t cmd;
  int branch_id;
  int flag;
  int parent_branch_id;
  iter_t clock_to_make;
};

struct inactivate_branch_msg_t {
  command_t cmd;
  int branch_id;
  iter_t clock_to_happen;
};

struct save_branch_msg_t {
  command_t cmd;
  int branch_id;
  iter_t clock_to_happen;
};

#endif  // defined __wire_protocol_hpp__
