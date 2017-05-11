/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/bind.hpp>

#include <tbb/tick_count.h>

#include <glog/logging.h>

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "common/row-op-util.hpp"
#include "encoder-decoder.hpp"
#include "clientlib.hpp"

using std::string;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using boost::bind;
using boost::make_shared;
using boost::shared_ptr;
using boost::unique_lock;
using boost::mutex;
using boost::condition_variable;

int ClientLib::virtual_read_batch(
    table_id_t table, const vector<row_idx_t>& row_ids,
    size_t num_val_limit) {
  OpInfo opinfo(OpInfo::READ, table, row_ids, num_val_limit);
  return virtual_op(opinfo);
}

int ClientLib::virtual_postread_batch(int prestep_handle) {
  OpInfo opinfo(OpInfo::POST_READ, prestep_handle);
  copy_row_keys_from_preop(opinfo);
  return virtual_op(opinfo);
}

int ClientLib::virtual_preupdate_batch(
    table_id_t table, const vector<row_idx_t>& row_ids,
    size_t num_val_limit) {
  OpInfo opinfo(OpInfo::PRE_UPDATE, table, row_ids, num_val_limit);
  return virtual_op(opinfo);
}

int ClientLib::virtual_update_batch(int prestep_handle) {
  OpInfo opinfo(OpInfo::UPDATE, prestep_handle);
  copy_row_keys_from_preop(opinfo);
  return virtual_op(opinfo);
}

int ClientLib::virtual_localaccess_batch(
    table_id_t table, const vector<row_idx_t>& row_ids,
    size_t num_val_limit, bool fetch) {
  OpInfo opinfo(OpInfo::LOCAL_ACCESS, table, row_ids, num_val_limit, fetch);
  return virtual_op(opinfo);
}

int ClientLib::virtual_postlocalaccess_batch(
    int prestep_handle, bool keep) {
  OpInfo opinfo(OpInfo::POST_LOCAL_ACCESS, prestep_handle, keep);
  copy_row_keys_from_preop(opinfo);
  return virtual_op(opinfo);
}

int ClientLib::virtual_clock() {
  OpInfo opinfo(OpInfo::CLOCK);
  return virtual_op(opinfo);
}

void ClientLib::copy_row_keys_from_preop(OpInfo& op_info) {
  OpSeq& opseq = thread_data->opseq;
  CHECK_GE(op_info.prestep_handle, 0);
  uint prestep_op_id = static_cast<uint>(op_info.prestep_handle);
  CHECK_LT(prestep_op_id, opseq.size());
  OpInfo& prestep_op_info = opseq[prestep_op_id];
  op_info.table = prestep_op_info.table;
  op_info.table_id = prestep_op_info.table_id;
  op_info.rows = prestep_op_info.rows;
  op_info.num_vals_limit = prestep_op_info.num_vals_limit;
}

int ClientLib::virtual_op(const OpInfo& opinfo) {
  ThreadData& thread_data_ref = *thread_data;
  /* Record to operation sequence */
  thread_data_ref.opseq.push_back(opinfo);
  return static_cast<int>(thread_data_ref.opseq.size() - 1);
}

void ClientLib::finish_virtual_iteration() {
  ScopedLock vilock(virtual_iteration_mutex);

  /* Summarize information */
  vi_thread_summarize();

  vi_process_finalize();

  /* Make decisions based on the virtual iteration for the current thread */
  vi_thread_finalize();

  virtual_iteration_all_finished = true;
  virtual_iteration_cvar.notify_all();
}

void ClientLib::vi_thread_summarize() {
  ngr_capacity = config.gpu_memory_capacity / sizeof(RowData);
  cout << "ngr_capacity = " << ngr_capacity << endl;

  /* Create op_data_buffers */
  OpSeq& opseq = thread_data->opseq;
  CHECK_GE(config.max_slack, 0);
  /* We want to have two data buffers for slack 0 */
  size_t op_data_buffer_size = static_cast<size_t>(config.max_slack) + 2;
  for (size_t i = 0; i < opseq.size(); i++) {
    OpInfo& opinfo = opseq[i];
    if (opinfo.type == OpInfo::READ || opinfo.type == OpInfo::PRE_UPDATE ||
        opinfo.type == OpInfo::LOCAL_ACCESS) {
      opinfo.op_data_buffers.resize(op_data_buffer_size);
    }
  }

  KeyBatchInfoMap key_batch_info_map;

  /* Create thread cache */
  vi_create_thread_cache();

  /* Create local storage */
  vi_create_local_storage(key_batch_info_map);

  /* Decide parameter cache placement */
  vi_decide_param_cache();

  /* Decide pinned data */
  vi_decide_pinned_data(key_batch_info_map);
}

void ClientLib::vi_create_thread_cache() {
  ThreadData& thread_data_ref = *thread_data;
  /* Use all the GPU memory as thread cache */
  CHECK_GE(ngr_capacity, 0);
  thread_data_ref.thread_cache_size = ngr_capacity;
  cout << "thread_cache_size = " << thread_data_ref.thread_cache_size << endl;
  ThreadCache& thread_cache = thread_data_ref.thread_cache;
  thread_cache.init(thread_data_ref.thread_cache_size);
}

void ClientLib::vi_create_local_storage(
    KeyBatchInfoMap& key_batch_info_map) {
  ThreadData& thread_data_ref = *thread_data;
  OpSeq& opseq = thread_data_ref.opseq;
  for (size_t i = 0; i < opseq.size(); i++) {
    OpInfo& opinfo = opseq[i];
    if (opinfo.type == OpInfo::LOCAL_ACCESS) {
      CHECK(opinfo.rows.size());
      TableRow first_key(opinfo.table, opinfo.rows[0]);
      if (!key_batch_info_map.count(first_key)) {
        /* New key batch */
        KeyBatchInfo& key_batch_info = key_batch_info_map[first_key];
        key_batch_info.table = opinfo.table;
        key_batch_info.rows = opinfo.rows;
      }
      KeyBatchInfo& key_batch_info = key_batch_info_map[first_key];
      CHECK_EQ(key_batch_info.rows.size(), opinfo.rows.size());
      if (opinfo.fetch_local) {
        key_batch_info.num_fetches++;
      }
    }
    if (opinfo.type == OpInfo::POST_LOCAL_ACCESS) {
      CHECK(opinfo.rows.size());
      TableRow first_key(opinfo.table, opinfo.rows[0]);
      CHECK(key_batch_info_map.count(first_key));
      KeyBatchInfo& key_batch_info = key_batch_info_map[first_key];
      CHECK_EQ(key_batch_info.rows.size(), opinfo.rows.size());
      if (opinfo.keep_local) {
        key_batch_info.num_keeps++;
      }
    }
  }

  size_t total_local_data_size = 0;
  thread_data_ref.fetchkeep_local_data_size = 0;
  for (KeyBatchInfoMap::iterator it = key_batch_info_map.begin();
       it != key_batch_info_map.end(); it++) {
    KeyBatchInfo& key_batch_info = it->second;
    total_local_data_size += key_batch_info.rows.size();
    if (!key_batch_info.num_fetches && key_batch_info.num_keeps) {
      /* Keep but no fetch */
    }
    if (key_batch_info.num_fetches && !key_batch_info.num_keeps) {
      /* Fetch but no keep */
      CHECK(0) << "Fetch keep mismatch: "
           << "rows[0] = " << key_batch_info.rows[0]
           << ", rows.size() = " << key_batch_info.rows.size()
           << ", num_fetches = " << key_batch_info.num_fetches
           << ", num_keeps = " << key_batch_info.num_keeps;
    }
    if (key_batch_info.num_fetches && key_batch_info.num_keeps) {
      /* Only create local storage for rows that are both kept and fetched */
      key_batch_info.fetchkeep = true;
      thread_data_ref.fetchkeep_local_data_size += key_batch_info.rows.size();
    } else {
      key_batch_info.fetchkeep = false;
    }
  }
  for (size_t i = 0; i < opseq.size(); i++) {
    OpInfo& opinfo = opseq[i];
    if (opinfo.type == OpInfo::LOCAL_ACCESS ||
        opinfo.type == OpInfo::POST_LOCAL_ACCESS) {
      CHECK(opinfo.rows.size());
      TableRow first_key(opinfo.table, opinfo.rows[0]);
      CHECK(key_batch_info_map.count(first_key));
      KeyBatchInfo& key_batch_info = key_batch_info_map[first_key];
      if (opinfo.fetch_local && !key_batch_info.fetchkeep) {
        CHECK(0);
      }
      if (opinfo.keep_local && !key_batch_info.fetchkeep) {
        /* This entry is never fetched, so we don't need to keep it */
        opinfo.keep_local = false;
      }
    }
  }
  // cout << "total_local_data_size = "
       // << total_local_data_size << endl;
  // cout << "fetchkeep_local_data_size = "
       // << thread_data_ref.fetchkeep_local_data_size << endl;

  /* Create local storage */
  LocalStorageOfModelBranches& local_storage_of_model_branches =
      thread_data_ref.local_storage_of_model_branches;
  local_storage_of_model_branches.resize(config.num_branches);

  /* Make local storage key list and allocate pinned entries */
  RowKeys local_row_keys;
  for (KeyBatchInfoMap::iterator it = key_batch_info_map.begin();
       it != key_batch_info_map.end(); it++) {
    KeyBatchInfo& key_batch_info = it->second;
    key_batch_info.shared_buffer = new SharedBuffer();
    key_batch_info.shared_buffer->has_local_storage = key_batch_info.fetchkeep;
    if (key_batch_info.fetchkeep) {
      /* Create a local storage in CPU memory for all fetchkeep local data.
       * Even for the pinned local data, it could be swapped to CPU memory,
       * when we swap branches. */
      key_batch_info.shared_buffer->storage_offset = local_row_keys.size();
      for (size_t j = 0; j < key_batch_info.rows.size(); j++) {
        RowKey row_key(key_batch_info.table, key_batch_info.rows[j]);
        local_row_keys.push_back(row_key);
      }
    }
  }

  /* Create CPU local storage */
  cout << "local_row_keys.size() = " << local_row_keys.size() << endl;
  CHECK_GT(local_storage_of_model_branches.size(), 0);
  DataStorage& local_storage0 =
      local_storage_of_model_branches[0].local_storage;
  if (config.pinned_cpu_memory) {
    local_storage0.init(local_row_keys.size(), DataStorage::PINNED_CPU);
  } else {
    local_storage0.init(local_row_keys.size(), DataStorage::CPU);
  }

  /* Clone the storage to other model branches */
  /* NOTE: here we assume all the branches use the same set of rows,
   * but this assumption is only true when we use the same batch size
   * for all branches. */
  for (uint branch_idx = 1; branch_idx < local_storage_of_model_branches.size();
      branch_idx++) {
    cout << "make local storage for branch idx " << branch_idx << endl;
    LocalStorageOfModelBranch& local_storage_of_model_branch =
        local_storage_of_model_branches[branch_idx];
    local_storage_of_model_branch.branch_id = -1;
    local_storage_of_model_branch.local_storage.init_from(local_storage0);
  }
  /* Reset branch id field */
  for (uint branch_idx = 0; branch_idx < local_storage_of_model_branches.size();
      branch_idx++) {
    LocalStorageOfModelBranch& local_storage_of_model_branch =
        local_storage_of_model_branches[branch_idx];
    local_storage_of_model_branch.branch_id = -1;
  }

  /* Create index for accessing local storage */
  for (size_t i = 0; i < opseq.size(); i++) {
    OpInfo& opinfo = opseq[i];
    if (opinfo.type == OpInfo::LOCAL_ACCESS) {
      CHECK(opinfo.rows.size());
      TableRow first_key(opinfo.table, opinfo.rows[0]);
      CHECK(key_batch_info_map.count(first_key));
      KeyBatchInfo& key_batch_info = key_batch_info_map[first_key];
      opinfo.shared_buffer = key_batch_info.shared_buffer;
    }
  }
}

void ClientLib::vi_decide_pinned_data(
    KeyBatchInfoMap& key_batch_info_map) {
  ThreadData& thread_data_ref = *thread_data;
  OpSeq& opseq = thread_data_ref.opseq;

  /* Decide pinned local data rows */
  size_t pinned_size = 0;
  size_t thread_cache_size = thread_data_ref.thread_cache_size;
  size_t fetchkeep_local_data_size = thread_data_ref.fetchkeep_local_data_size;
  size_t nr_being_used_peak;
  size_t nr_used_all;
  size_t max_nr_each_access;
  bool all_data_pinned = false;
  while (true) {
    size_t pinned_size_updated = 0;
    /* Calculate the peak size */
    size_t nr_being_used_now = 0;
    nr_being_used_peak = 0;
    nr_used_all = 0;
    max_nr_each_access = 0;
    size_t nr_being_used_second_peak = 0;
    int max_peak_start = -1;
    int this_peak_start = -1;
    int second_peak_start = -1;
    /* Decide peak size */
    for (size_t i = 0; i < opseq.size(); i++) {
      OpInfo& opinfo = opseq[i];
      if (opinfo.type == OpInfo::READ || opinfo.type == OpInfo::PRE_UPDATE ||
          opinfo.type == OpInfo::LOCAL_ACCESS) {
        if (opinfo.type == OpInfo::LOCAL_ACCESS) {
          CHECK(opinfo.rows.size());
          TableRow first_key(opinfo.table, opinfo.rows[0]);
          CHECK(key_batch_info_map.count(first_key));
          KeyBatchInfo& key_batch_info = key_batch_info_map[first_key];
          if (key_batch_info.pinned) {
            /* Don't count the local data against peak size if it's pinned */
            continue;
          }
        }
        if (this_peak_start == -1) {
          this_peak_start = i;
        }
        nr_being_used_now += opinfo.rows.size();
        nr_used_all += opinfo.rows.size();
        if (opinfo.rows.size() > max_nr_each_access) {
          max_nr_each_access = opinfo.rows.size();
        }
      }
      if (opinfo.type == OpInfo::POST_READ || opinfo.type == OpInfo::UPDATE
          || opinfo.type == OpInfo::POST_LOCAL_ACCESS) {
        if (opinfo.type == OpInfo::POST_LOCAL_ACCESS) {
          CHECK(opinfo.rows.size());
          TableRow first_key(opinfo.table, opinfo.rows[0]);
          CHECK(key_batch_info_map.count(first_key));
          KeyBatchInfo& key_batch_info = key_batch_info_map[first_key];
          if (key_batch_info.pinned) {
            /* Don't count the local data against peak size if it's pinned */
            continue;
          }
        }
        if (this_peak_start != -1) {
          /* End of a peak */
          if (nr_being_used_now > nr_being_used_peak) {
            nr_being_used_second_peak = nr_being_used_peak;
            second_peak_start = max_peak_start;
            nr_being_used_peak = nr_being_used_now;
            max_peak_start = this_peak_start;
          } else if (nr_being_used_now > nr_being_used_second_peak) {
            nr_being_used_second_peak = nr_being_used_now;
            second_peak_start = this_peak_start;
          }
        }
        nr_being_used_now -= opinfo.rows.size();
        this_peak_start = -1;
        CHECK_GE(nr_being_used_now, 0);
      }
    }
    CHECK_NE(second_peak_start, second_peak_start + 1);
    // cout << "nr_being_used_peak = " << nr_being_used_peak << endl;
    // cout << "max_peak_start = " << max_peak_start << endl;
    // cout << "nr_being_used_second_peak = " << nr_being_used_second_peak << endl;
    // cout << "second_peak_start = " << second_peak_start << endl;
    if (all_data_pinned) {
      /* All data has been assigned to GPU memory */
      break;
    }
    /* The unpinned thread cache will be at least twice the size
     * of peak access */
    CHECK_GE(thread_cache_size, pinned_size + nr_being_used_peak * 2);
    if (fetchkeep_local_data_size + nr_being_used_peak * 2
        <= thread_cache_size) {
      /* This means that the thread cache size is large enough for us
       * to pin all the fetchkeep local data in it */
      /* Pin all the fetchkeep local data in GPU */
      // cout << "Keep all fetchkeep local storage in GPU\n";
      for (KeyBatchInfoMap::iterator it = key_batch_info_map.begin();
           it != key_batch_info_map.end(); it++) {
        KeyBatchInfo& key_batch_info = it->second;
        if (key_batch_info.pinned == true) {
          continue;
        }
        if (key_batch_info.fetchkeep) {
          /* Only consider pinning the fetchkeep entries */
          key_batch_info.pinned = true;
          pinned_size += key_batch_info.rows.size();
        }
      }
      all_data_pinned = true;
      /* Do not break out of the while(),
       * because we need to recalculate the peak */
    } else {
      /* Pin a part of fetchkeep local data in GPU */
      /* Try to reduce the peak */
      bool peak_changed = false;
      CHECK_GE(max_peak_start, 0);
      CHECK_LT(max_peak_start, opseq.size());
      for (int i = max_peak_start; i < opseq.size(); i++) {
        OpInfo& opinfo = opseq[i];
        if (opinfo.type == OpInfo::LOCAL_ACCESS) {
          CHECK(opinfo.rows.size());
          TableRow first_key(opinfo.table, opinfo.rows[0]);
          CHECK(key_batch_info_map.count(first_key));
          KeyBatchInfo& key_batch_info = key_batch_info_map[first_key];
          if (key_batch_info.pinned) {
            /* Skip the entries that are already pinned */
            continue;
          }
          if (key_batch_info.fetchkeep) {
            /* Only consider pinning the fetchkeep entries */
            size_t size_after_changing_peak =
                nr_being_used_peak - key_batch_info.rows.size();
            CHECK_GE(size_after_changing_peak, 0);
            if (size_after_changing_peak < nr_being_used_second_peak) {
              size_after_changing_peak = nr_being_used_second_peak;
            }
            if (pinned_size + key_batch_info.rows.size() +
                + size_after_changing_peak * 2 <= thread_cache_size) {
              key_batch_info.pinned = true;
              pinned_size += key_batch_info.rows.size();
              pinned_size_updated++;
              peak_changed = true;
              // cout << "peak changed" << endl;
              break;
            }
          }
        }
        if (opinfo.type == OpInfo::POST_READ || opinfo.type == OpInfo::UPDATE
            || opinfo.type == OpInfo::POST_LOCAL_ACCESS) {
          /* End of this peak */
          break;
        }
      }
      if (!peak_changed) {
        /* Randomly pick some */
        for (KeyBatchInfoMap::iterator it = key_batch_info_map.begin();
             it != key_batch_info_map.end(); it++) {
          KeyBatchInfo& key_batch_info = it->second;
          if (key_batch_info.pinned) {
            /* Skip the entries that are already pinned */
            continue;
          }
          if (key_batch_info.fetchkeep) {
            /* Only consider pinning the fetchkeep entries */
            if (pinned_size + key_batch_info.rows.size() +
                nr_being_used_peak * 2 <= thread_cache_size) {
              key_batch_info.pinned = true;
              pinned_size += key_batch_info.rows.size();
              pinned_size_updated++;
            }
          }
        }
      }
      if (!pinned_size_updated) {
        /* Exit the loop with some entries not pinned in GPU memory */
        CHECK_LT(config.mm_warning_level, 2)
            << " thread_cache_size = " << thread_cache_size
            << " nr_being_used_peak = " << nr_being_used_peak
            << " fetchkeep_local_data_size = " << fetchkeep_local_data_size
            << endl;
        break;
      }
    }
  }

  /* Make local storage key list and allocate pinned entries */
  for (KeyBatchInfoMap::iterator it = key_batch_info_map.begin();
       it != key_batch_info_map.end(); it++) {
    KeyBatchInfo& key_batch_info = it->second;
    key_batch_info.shared_buffer->pinned_entry = key_batch_info.pinned;
    if (key_batch_info.pinned) {
      /* Pin this entry in thread cache, by simply allocating the entry */
      /* We allocate the pinned entries now, instead of allocating
       * when it's used, because we want the pinned the entries to be
       * contiguous in the thread cache */
      size_t num_rows = key_batch_info.rows.size();
      bool wait = false;
      key_batch_info.shared_buffer->buffer =
          thread_data_ref.thread_cache.get(num_rows, wait);
      key_batch_info.shared_buffer->num_rows = num_rows;
    }
  }

  /* Allocate staging memory for GPU/CPU data transfer */
  size_t size = max_nr_each_access * sizeof(RowData);
  mallocHost(&thread_data_ref.staging_mem, size);
  thread_data_ref.staging_mem_size = size;
  // cout << "max_nr_each_access = " << max_nr_each_access << endl;
}

void ClientLib::vi_decide_param_cache() {
  ThreadData& thread_data_ref = *thread_data;
  OpSeq& opseq = thread_data_ref.opseq;

  size_t num_oplog_entries = 1;
  if (config.read_my_writes) {
    CHECK(0);
    /* If we don't do read-my-updates, just one entry is enough */
    /* If we do read-my-updates, we need slack + 1 entries.
     * We always create the oplog entry of the current clock
     * at the first INC operation.
     * If the application only starts calling INC after finishing reading all
     * parameter data for the current clock, we need only (slack + 1)
     * oplog entries. Otherwise, we need (slack + 2). */
    // num_oplog_entries = static_cast<size_t>(config.max_slack) + 1;
  }
  /* Set info for all channels */
  for (size_t i = 0; i < comm_channels.size(); i++) {
    comm_channels[i].num_oplog_entries = num_oplog_entries;
  }

  /* We keep all param cache entries in CPU memory */
  RowKeys& param_row_keys = thread_data_ref.param_row_keys_cpu;
  boost::unordered_set<TableRow> existing_keys;
  param_row_keys.clear();
  for (size_t i = 0; i < opseq.size(); i++) {
    OpInfo& opinfo = opseq[i];
    if (opinfo.type == OpInfo::CLOCK) {
      continue;
    }
    if (opinfo.type != OpInfo::LOCAL_ACCESS
        && opinfo.type != OpInfo::POST_LOCAL_ACCESS
        && opinfo.rows.size()) {
      TableRow first_key(opinfo.table, opinfo.rows[0]);
      if (!existing_keys.count(first_key)) {
        /* New keys */
        /* Insert to key list */
        for (size_t j = 0; j < opinfo.rows.size(); j++) {
          TableRow table_row(opinfo.table, opinfo.rows[j]);
          CHECK(!existing_keys.count(table_row));
          RowKey row_key(opinfo.table, opinfo.rows[j]);
          param_row_keys.push_back(row_key);
          existing_keys.insert(table_row);
        }
      } else {
        /* The keys exist */
        for (size_t j = 0; j < opinfo.rows.size(); j++) {
          TableRow table_row(opinfo.table, opinfo.rows[j]);
          CHECK(existing_keys.count(table_row));
        }
      }
    }
  }

  /* Divide rows to channels.
   * Here we assume that all workers access the same set of rows,
   * so even though they decide row_id to channel_id mapping independently,
   * this decision is consistent across all workers. */
  cout << "param_row_keys.size() = " << param_row_keys.size() << endl;
  rows_per_channel.resize(config.num_tables);
  vector<size_t> row_counts(config.num_tables);
  for (uint table_id = 0; table_id < config.num_tables; table_id++) {
    row_counts[table_id] = 0;
  }
  for (size_t i = 0; i < param_row_keys.size(); i++) {
    CHECK_LT(param_row_keys[i].table, row_counts.size());
    row_counts[param_row_keys[i].table]++;
  }
  for (uint table_id = 0; table_id < config.num_tables; table_id++) {
    rows_per_channel[table_id] =
      (row_counts[table_id] + comm_channels.size() - 1)
          / comm_channels.size();
    // cout << "rows_per_channel of " << table_id
         // << " = " << rows_per_channel[table_id] << endl;
  }
}

void ClientLib::vi_process_channel_table_finalize(
    ThreadData& thread_data_ref, uint channel_id, uint table_id, bool gpu) {
  RowKeys& param_row_keys =
      gpu ? thread_data_ref.param_row_keys_gpu :
            thread_data_ref.param_row_keys_cpu;
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  CHECK_LT(table_id, comm_channel.cached_tables.size());
  CachedTable& cached_table = comm_channel.cached_tables[table_id];
  StaticCache& static_cache = gpu ?
      cached_table.static_cache_gpu :
      cached_table.static_cache_cpu;
  StaticCacheIndex& static_cache_index = static_cache.static_cache_index;
  StaticOpLog& oplog = static_cache.oplog;

  static_cache.num_rows = 0;
  for (size_t i = 0; i < param_row_keys.size(); i++) {
    table_id_t table = param_row_keys[i].table;
    row_idx_t row = param_row_keys[i].row;
    if (table != table_id) {
      /* Consider only the current table */
      continue;
    }
    if (get_channel_id(table, row) != channel_id) {
      /* Consider only the current channel */
      continue;
    }
    TableRow table_row(table, row);
    StaticCacheMetadata& static_cache_metadata =
        static_cache_index[table_row];
    /* !!!! FIXME: now I assume cache_idx is the same as row_id */
    // static_cache_metadata.cache_idx = row_count;
    static_cache_metadata.cache_idx = static_cache.num_rows;
    static_cache_metadata.tablet_server_id = -2;
    static_cache.num_rows++;
  }
  // if (gpu) {
    // cout << "Channel #" << channel_id << " Table #" << table_id
         // << " GPU cache size " << static_cache.num_rows << endl;
  // } else {
    // cout << "Channel #" << channel_id << " Table #" << table_id
         // << " CPU cache size " << static_cache.num_rows << endl;
  // }

  /* Create static cache */
  static_cache.row_keys.resize(static_cache.num_rows);
  DataCacheOfModelBranches& data_cache_of_model_branches =
      static_cache.data_cache_of_model_branches;
  data_cache_of_model_branches.resize(config.num_branches);
  CHECK_GT(data_cache_of_model_branches.size(), 0);
  DataCache& data_cache0 =
      data_cache_of_model_branches[0].data_cache;
  if (gpu) {
    data_cache0.init(static_cache.num_rows, DataStorage::GPU);
    data_cache0.zerofy_data_gpu(thread_data_ref.cuda_stream);
  } else {
    data_cache0.init(static_cache.num_rows, DataStorage::CPU);
    data_cache0.zerofy_data_cpu();
  }

  /* Clone the storage to other model branches */
  /* NOTE: here we assume all the branches use the same set of rows */
  for (uint branch_idx = 1; branch_idx < data_cache_of_model_branches.size();
      branch_idx++) {
    // cout << "make data cache for branch idx " << branch_idx << endl;
    data_cache_of_model_branches[branch_idx].data_cache.init_from(data_cache0);
  }
  /* Reset branch id field */
  for (uint branch_idx = 0; branch_idx < data_cache_of_model_branches.size();
      branch_idx++) {
    data_cache_of_model_branches[branch_idx].branch_id = -1;
  }

  CHECK(comm_channel.num_oplog_entries);
  static_cache.opmem_buffer_pool.init(
      gpu, comm_channel.num_oplog_entries, static_cache.num_rows);
  /* FIXME: this only works for the case where all clients access
   * the same set of rows. */
  static_cache.per_server_row_start.resize(num_servers);
  static_cache.per_server_num_rows.resize(num_servers);
  size_t div = static_cache.num_rows / num_servers;
  size_t res = static_cache.num_rows % num_servers;
  for (size_t i = 0; i < num_servers; i++) {
    static_cache.per_server_row_start[i] = div * i + (res > i ? i : res);
    static_cache.per_server_num_rows[i] = div + (res > i ? 1 : 0);
  }
  /* The oplog is a circular array */
  size_t num_oplog_entries = static_cast<size_t>(config.max_slack) + 2;
  cout << "num_oplog_entries = " << num_oplog_entries << endl;
  oplog.resize(num_oplog_entries);
  for (StaticCacheIndex::iterator table_row_it
         = static_cache_index.begin();
       table_row_it != static_cache_index.end(); table_row_it++) {
    const TableRow& table_row = table_row_it->first;
    table_id_t table = table_row.first;
    row_idx_t row = table_row.second;
    StaticCacheMetadata& static_cache_metadata =
        static_cache_index[table_row];
    size_t row_idx = static_cache_metadata.cache_idx;
    static_cache_metadata.row_idx = row_idx;
    static_cache.row_keys[row_idx].table = table;
    static_cache.row_keys[row_idx].row = row;
  }
}

void ClientLib::vi_process_channel_finalize(
    ThreadData& thread_data_ref, uint channel_id, bool gpu) {
  CommunicationChannel& comm_channel = comm_channels[channel_id];
  for (uint table_id = 0; table_id < config.num_tables; table_id++) {
    vi_process_channel_table_finalize(
        thread_data_ref, channel_id, table_id, gpu);
  }

  /* Create send/receive buffers (only necessary for the GPU cache) */
  if (gpu) {
    /* Decide the size of the biggest table */
    size_t biggest_table_size = 0;
    for (uint table_id = 0; table_id < config.num_tables; table_id++) {
      size_t table_size =
          comm_channel.cached_tables[table_id].static_cache_gpu.num_rows;
      biggest_table_size =
          table_size > biggest_table_size ? table_size : biggest_table_size;
    }
    size_t size = biggest_table_size * sizeof(RowData);
    comm_channel.comm_buffer_size = size;
    mallocHost(&comm_channel.send_buffer, size);
    mallocHost(&comm_channel.recv_buffer, size);
  }
}

void ClientLib::vi_process_finalize() {
  ThreadData& thread_data_ref = *thread_data;
  size_t thread_id = thread_data_ref.thread_id;
  if (nr_threads >= num_channels) {
    size_t threads_per_channel = nr_threads / num_channels;
    CHECK(threads_per_channel);
    bool is_leading = (thread_id % threads_per_channel == 0);
    if (!is_leading) {
      return;
    }
    uint channel_id = thread_id / threads_per_channel;
    if (channel_id >= num_channels) {
      return;
    }
    vi_process_channel_finalize(
        thread_data_ref, channel_id, false /* cpu */);
    vi_process_channel_finalize(
        thread_data_ref, channel_id, true /* gpu */);
  } else {
    size_t div = num_channels / nr_threads;
    size_t res = num_channels % nr_threads;
    size_t start = div * thread_id + (res > thread_id ? thread_id : res);
    size_t size = div + (res > thread_id ? 1 : 0);
    for (uint channel_id = start; channel_id < start + size; channel_id++) {
      vi_process_channel_finalize(
          thread_data_ref, channel_id, false /* cpu */);
      vi_process_channel_finalize(
          thread_data_ref, channel_id, true /* gpu */);
    }
  }
}

void ClientLib::vi_thread_finalize() {
  ThreadData& thread_data_ref = *thread_data;
  OpSeq& opseq = thread_data_ref.opseq;

  /* Decide last update of each row batch and each table */
  boost::unordered_map<uint, bool> table_last_update_map;
  boost::unordered_map<TableRow, bool> batch_last_update_map;
  bool clock_op_seen = false;
  for (size_t i = opseq.size() - 1; i >= 0 && i < opseq.size(); i--) {
    OpInfo& op_info = opseq[i];
    if (op_info.type == OpInfo::CLOCK) {
      clock_op_seen = true;
    }
    if (!clock_op_seen) {
      /* Operations after CLOCK are all unrepeated ones */
      CHECK(0);
      continue;
    }
    if (op_info.type == OpInfo::UPDATE) {
      uint table_id = op_info.table_id;
      if (!table_last_update_map.count(table_id)) {
        table_last_update_map[table_id] = true;
        op_info.table_last_update = true;
      } else {
        op_info.table_last_update = false;
      }
    }
    if (op_info.type == OpInfo::POST_LOCAL_ACCESS ||
        op_info.type == OpInfo::UPDATE) {
      CHECK(op_info.rows.size());
      TableRow table_row(op_info.table, op_info.rows[0]);
      if (!batch_last_update_map.count(table_row)) {
        batch_last_update_map[table_row] = true;
        op_info.batch_last_update = true;
      } else {
        op_info.batch_last_update = false;
      }
    }
  }
  for (uint table_id = 0; table_id < config.num_tables; table_id++) {
    CHECK(table_last_update_map.count(table_id))
        << "No one updates table " << table_id;
  }

  /* Create index for accessing process cache */
  for (size_t i = 0; i < opseq.size(); i++) {
    OpInfo& op_info = opseq[i];
    if (op_info.type == OpInfo::READ || op_info.type == OpInfo::PRE_UPDATE) {
      /* Double index: row_in_cache[index0[i]] -> accessed_row[index1[i]]
       * It should be used when the number of rows accessed
       * in this batch is much smaller than the total number of rows
       * (e.g., less than half). */
      vi_create_double_index(op_info);
    }
  }
}

void ClientLib::vi_create_double_index(OpInfo& opinfo) {
  opinfo.index_type = OpInfo::DOUBLE_INDEX;
  vector<vector<DoubleIndex> > row_index_tmp;
  row_index_tmp.resize(num_channels);
  for (size_t j = 0; j < opinfo.rows.size(); j++) {
    TableRow table_row(opinfo.table, opinfo.rows[j]);
    uint channel_id = get_channel_id(table_row);
    CHECK_LT(channel_id, comm_channels.size());
    CommunicationChannel& comm_channel = comm_channels[channel_id];
    uint table_id = opinfo.table_id;
    CHECK_LT(table_id, comm_channel.cached_tables.size());
    CachedTable& cached_table = comm_channel.cached_tables[table_id];
    StaticCache& static_cache = cached_table.static_cache_cpu;
    StaticCacheIndex& static_cache_index = static_cache.static_cache_index;
    StaticCacheIndex::iterator index_it = static_cache_index.find(table_row);
    CHECK(index_it != static_cache_index.end())
        << "row = " << opinfo.rows[j];
    StaticCacheMetadata& static_cache_metadata = index_it->second;
    size_t cache_idx = static_cache_metadata.cache_idx;
    CHECK_LT(cache_idx, static_cache.num_rows);
    row_index_tmp[channel_id].push_back(DoubleIndex(j, cache_idx));
  }
  /* Allocate and create index in CPU memory */
  opinfo.row_index_size = opinfo.rows.size();
  size_t index_memsize = opinfo.rows.size() * sizeof(DoubleIndex);
  opinfo.row_index_cpu = reinterpret_cast<DoubleIndex *>(malloc(index_memsize));
  CHECK(opinfo.row_index_cpu);
  opinfo.index_infos.resize(num_channels);
  size_t current_index = 0;
  for (uint channel_id = 0; channel_id < num_channels; channel_id++) {
    PerChannelIndexInfo& index_info = opinfo.index_infos[channel_id];
    index_info.index_start = current_index;
    index_info.index_size = row_index_tmp[channel_id].size();
    size_t index_range = 0;
    if (row_index_tmp[channel_id].size()) {
      size_t index_min = row_index_tmp[channel_id][0].id1;
      size_t index_max = row_index_tmp[channel_id][0].id1;
      for (size_t index_id = 0;
           index_id < row_index_tmp[channel_id].size(); index_id++) {
        opinfo.row_index_cpu[current_index++] =
            row_index_tmp[channel_id][index_id];
        if (row_index_tmp[channel_id][index_id].id1 < index_min) {
          index_min = row_index_tmp[channel_id][index_id].id1;
        }
        if (row_index_tmp[channel_id][index_id].id1 > index_max) {
          index_max = row_index_tmp[channel_id][index_id].id1;
        }
      }
      index_range = index_max - index_min + 1;
    }
    /* We think the indexed rows should be contiguous.
     * So maybe memcpy can be used instead? */
    CHECK_EQ(index_range, row_index_tmp[channel_id].size());
  }
  CHECK_EQ(current_index, opinfo.rows.size());
}
