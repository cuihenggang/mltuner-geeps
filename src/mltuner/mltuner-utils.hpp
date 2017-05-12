#ifndef __mltuner_utils_hpp__
#define __mltuner_utils_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <set>
#include <string>
#include <vector>
#include <list>
#include <queue>

#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>

#include <glog/logging.h>

#include "geeps.hpp"
#include "common/common-utils.hpp"

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;

typedef std::pair<int, int> BranchOpRefCountKey;
    /* The key is <OpType, Clock> */
typedef boost::unordered_map<BranchOpRefCountKey, int> BranchOpRefCountMap;

#define INVALID_BRANCH    0
#define TRAINING_BRANCH   1
#define TESTING_BRANCH    2
#define INIT_BRANCH       3
#define EVAL_BRANCH       4

struct BranchInfo {
  int branch_id;
  bool active;
  Tunable tunable;
  int flag;
  int parent_branch_id;
  uint branch_idx;
  int internal_clock;
  int lineage_clock;
  BranchOpRefCountMap branch_op_ref_count_map;
  std::vector<int> scheduled_clocks;
  BranchInfo() {
    init();
  }
  BranchInfo(
      int branch_id, const Tunable& tunable, int flag, int parent_branch_id)
        : branch_id(branch_id),
          tunable(tunable), flag(flag), parent_branch_id(parent_branch_id) {
    init();  
  }
  void init() {
    active = true;
    internal_clock = 0;
    lineage_clock = 0;
  }
};
typedef boost::unordered_map<int, BranchInfo *> BranchInfoMap;

struct BranchRequest {
  int clock;
  enum Type {
    NONE,
    MAKE_BRANCH,
    INACTIVATE_BRANCH,
  } type;
  int branch_id;
  BranchRequest(int clock, Type type, int branch_id)
      : clock(clock), type(type), branch_id(branch_id) {}
};
typedef std::queue<BranchRequest *> BranchRequestsQueue;
typedef std::vector<BranchRequestsQueue> BranchRequestsQueues;

struct ClockSchedule {
  int branch_id;
  int internal_clock;
  ClockSchedule() : branch_id(-1) {}
  ClockSchedule(int branch_id, int internal_clock) :
      branch_id(branch_id), internal_clock(internal_clock) {}
};
typedef vector<ClockSchedule> ClockSchedules;
    /* Indexed by clock */

class BranchManager {
 public:
  uint max_num_branches;
  uint num_queues;
  BranchInfoMap branch_info_map;
  std::set<uint> avail_branch_idx;
  std::map<uint, uint> allocated_branch_idx;
      /* We need this allocated_branch_idx map to
       * keep track of the ref count */
  BranchRequestsQueues branch_requests_queues;
  std::vector<ClockSchedule> clock_schedules;
  boost::mutex mutex;
  boost::condition_variable cvar;

  BranchManager(uint max_num_branches, uint num_queues);
  void alloc_branch(int branch_id);
  void free_branch(int branch_id);
  uint alloc_branch_idx(uint ref_count);
  bool free_branch_idx(uint branch_idx);
  void add_clock_schedule_impl(int clock, int branch_id);
  void add_clock_schedule(int clock, int branch_id);
  void add_clock_schedules(uint batch_size, int *clocks, int *branch_ids);
  ClockSchedule get_clock_schedule(ScopedLock *lock_ptr, int clock);
  ClockSchedule get_clock_schedule_no_wait(int clock);
  uint get_branch_idx(int branch_id);
  Tunable& get_tunable(int branch_id);
};

#endif  // defined __mltuner_utils_hpp__
