/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <string>
#include <vector>

#include "geeps.hpp"
#include "clientlib.hpp"

using std::string;
using std::vector;
using boost::shared_ptr;

GeePs::GeePs(
    uint process_id, const Config& config) {
  ClientLib::CreateInstance(process_id, config);
  client_lib->thread_start();
}

void GeePs::Shutdown() {
  client_lib->thread_stop();
  client_lib->shutdown();
}

string GeePs::GetStats() {
  return client_lib->json_stats();
}

void GeePs::StartIterations() {
  client_lib->start_opseq();
}

int GeePs::VirtualRead(
    size_t table_id, const vector<size_t>& row_ids) {
  size_t num_val_limit = row_ids.size() * ROW_DATA_SIZE;
  return client_lib->virtual_read_batch(
      table_id, row_ids, num_val_limit);
}

int GeePs::VirtualPostRead(int prestep_handle) {
  return client_lib->virtual_postread_batch(prestep_handle);
}

int GeePs::VirtualPreUpdate(size_t table_id, const vector<size_t>& row_ids) {
  size_t num_val_limit = row_ids.size() * ROW_DATA_SIZE;
  return client_lib->virtual_preupdate_batch(
      table_id, row_ids, num_val_limit);
}

int GeePs::VirtualUpdate(int prestep_handle) {
  return client_lib->virtual_update_batch(prestep_handle);
}

int GeePs::VirtualLocalAccess(const vector<size_t>& row_ids, bool fetch) {
  /* table_id doesn't matter for local access */
  size_t table_id = 0xdeadbeef;
  size_t num_val_limit = row_ids.size() * ROW_DATA_SIZE;
  return client_lib->virtual_localaccess_batch(
      table_id, row_ids, num_val_limit, fetch);
}

int GeePs::VirtualPostLocalAccess(int prestep_handle, bool keep) {
  return client_lib->virtual_postlocalaccess_batch(prestep_handle, keep);
}

int GeePs::VirtualClock() {
  return client_lib->virtual_clock();
}

void GeePs::Read(int handle, RowData **buffer_ptr) {
  client_lib->app_acquire_routine(buffer_ptr, handle);
}

void GeePs::PostRead(int handle) {
  client_lib->app_release_routine(handle);
}

void GeePs::PreUpdate(int handle, RowData **buffer_ptr) {
  client_lib->app_acquire_routine(buffer_ptr, handle);
}

void GeePs::Update(int handle) {
  client_lib->app_release_routine(handle);
}

void GeePs::LocalAccess(int handle, RowData **buffer_ptr) {
  client_lib->app_acquire_routine(buffer_ptr, handle);
}

void GeePs::PostLocalAccess(int handle) {
  client_lib->app_release_routine(handle);
}

void GeePs::Clock() {
  client_lib->iterate();
}

void GeePs::FinishVirtualIteration() {
  client_lib->finish_virtual_iteration();
}

void GeePs::Report(double progress) {
  client_lib->report(progress);
}

Tunable GeePs::GetTunable() {
  return client_lib->get_tunable();
}

void GeePs::GetCurrentBranchInfo(Tunable *tunable, int *flag) {
  client_lib->get_current_branch_info(tunable, flag);
}

shared_ptr<BranchManager> GeePs::GetBranchManager() {
  return client_lib->get_branch_manager();
}
