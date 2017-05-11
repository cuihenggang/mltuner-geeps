/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include "mltuner-utils.hpp"

BranchManager::BranchManager(uint max_num_branches, uint num_queues) :
    max_num_branches(max_num_branches), num_queues(num_queues) {
  for (uint i = 0; i < max_num_branches; i++) {
    avail_branch_idx.insert(i);
  }
  branch_requests_queues.resize(num_queues);
}

void BranchManager::alloc_branch(int branch_id) {
  
}

void BranchManager::free_branch(int branch_id) {
  BranchInfo *branch_info = branch_info_map[branch_id];
  CHECK(branch_info);
  uint branch_idx = branch_info->branch_idx;
  if (free_branch_idx(branch_idx)) {
    /* Branch freed */
    branch_info->active = false;
    branch_info->branch_idx = max_num_branches;
  } else {
    /* This branch still has non-zero reference count */
  }
}

uint BranchManager::alloc_branch_idx(uint ref_count) {
  std::set<uint>::iterator it = avail_branch_idx.begin();
  CHECK(it != avail_branch_idx.end());
  uint branch_idx = *it;
  avail_branch_idx.erase(branch_idx);
  CHECK_EQ(allocated_branch_idx[branch_idx], 0);
  allocated_branch_idx[branch_idx] = ref_count;
  return branch_idx;
}

bool BranchManager::free_branch_idx(uint branch_idx) {
  std::map<uint, uint>::iterator it = allocated_branch_idx.find(branch_idx);
  CHECK(it != allocated_branch_idx.end());
  uint& ref_count = it->second;
  CHECK_GT(ref_count, 0) << "branch_idx = " << branch_idx;
  ref_count--;
  if (ref_count == 0) {
    avail_branch_idx.insert(branch_idx);
    return true;
  }
  return false;
}

void BranchManager::add_clock_schedule_impl(int clock, int branch_id) {
  if (clock >= clock_schedules.size()) {
    /* New clock schedule */
    CHECK_EQ(clock, clock_schedules.size());
    BranchInfo *branch_info = branch_info_map[branch_id];
    CHECK(branch_info);
    CHECK_EQ(branch_info->internal_clock, branch_info->scheduled_clocks.size())
        << " clock = " << clock << " branch_id = " << branch_id;
    branch_info->scheduled_clocks.push_back(clock);
    clock_schedules.push_back(
        ClockSchedule(branch_id, branch_info->internal_clock));
    branch_info->internal_clock++;
  } else {
    ClockSchedule& clock_schedule = clock_schedules[clock];
    CHECK_EQ(branch_id, clock_schedule.branch_id);
  }
}

void BranchManager::add_clock_schedule(int clock, int branch_id) {
  add_clock_schedule_impl(clock, branch_id);
  cvar.notify_all();
}

void BranchManager::add_clock_schedules(
    uint batch_size, int *clocks, int *branch_ids) {
  for (uint i = 0; i < batch_size; i++) {
    add_clock_schedule_impl(clocks[i], branch_ids[i]);
  }
  cvar.notify_all();
}

ClockSchedule BranchManager::get_clock_schedule(
    ScopedLock *lock_ptr, int clock) {
  if (lock_ptr) {
    while (clock >= clock_schedules.size()) {
      if (!cvar.timed_wait(*lock_ptr,
          boost::posix_time::milliseconds(12000))) {
        cerr << "Waiting for branch decision of clock " << clock
             << " timed out\n";
        // CHECK(0);
      }
    }
  }
  CHECK_LT(clock, clock_schedules.size());
  ClockSchedule clock_schedule = clock_schedules[clock];
  return clock_schedule;
}

/* This function does not wait.
 * If the clock schedule of the next clock is unknown,
 * the clock_schedule will be filled with (-1, UNINITIALIZED_CLOCK) */
ClockSchedule BranchManager::get_clock_schedule_no_wait(int clock) {
  if (clock < clock_schedules.size()) {
    ClockSchedule clock_schedule = clock_schedules[clock];
    return clock_schedule;
  } else {
    ClockSchedule clock_schedule;
    clock_schedule.branch_id = -1;
    clock_schedule.internal_clock = UNINITIALIZED_CLOCK;
    return clock_schedule;
  }
}

uint BranchManager::get_branch_idx(int branch_id) {
  BranchInfo *branch_info = branch_info_map[branch_id];
  CHECK(branch_info) << "branch_id = " << branch_id;
  uint branch_idx = branch_info->branch_idx;
  if (branch_info->active) {
    CHECK_LT(branch_idx, max_num_branches);
  }
  /* Return branch_idx as max_num_branches when this branch is inactivated */
  return branch_idx;
}

Tunable& BranchManager::get_tunable(int branch_id) {
  BranchInfo *branch_info = branch_info_map[branch_id];
  CHECK(branch_info) << "branch_id = " << branch_id;
  Tunable& tunable = branch_info->tunable;
  return tunable;
}
