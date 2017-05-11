#ifndef __mltuner_impl_hpp__
#define __mltuner_impl_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>
#include <iostream>

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>

#include <tbb/tick_count.h>

#include <glog/logging.h>

#include "mltuner.hpp"

#include "common/common-utils.hpp"
#include "mltuner-utils.hpp"

#define RUN_TO_TRY_RATIO              10
#define GO_UP_THRESHOLD               0.1
// #define GO_UP_THRESHOLD_RATIO        100.0
#define SAVING_BY_INTERSECT           1
#define MAX_PENDING_SPEARMINT_EXPS    2
// #define MAX_PENDING_SPEARMINT_EXPS    1
#define DIVERGENCE_THRESHOLD  10

typedef std::map<int, float> TunableChoice;
typedef vector<TunableChoice> TunableChoices;

enum ConvergenceState {
  DIVERGED,
  UNSTABLE,
  CONVERGING,
};

struct SchedulerBranchInfo : public BranchInfo {
  TunableChoice tunable_choice;
  int num_clocks_scheduled;
  double scheduled_run_time;
  double time_per_clock;
  double last_print_time;
  ConvergenceState convergence_state;
  double progress_slope;
  double first_progress_time;
  DoubleVec live_progress;
  DoubleVec timeline;
  DoubleVec progress;
  DoubleVec timeline_ds;
  DoubleVec progress_ds;
  DoubleVec timeline_adjusted_start;
  DoubleVec progress_adjusted_start;
  std::queue<int> resumed_clocks;
  int next_global_clock;

  SchedulerBranchInfo(
      int branch_id, const Tunable& tunable, int flag, int parent_branch_id)
        : BranchInfo(branch_id, tunable, flag, parent_branch_id),
          num_clocks_scheduled(0), scheduled_run_time(0.0),
          time_per_clock(std::numeric_limits<double>::quiet_NaN()),
          last_print_time(0.0),
          first_progress_time(std::numeric_limits<double>::quiet_NaN()),
          next_global_clock(UNINITIALIZED_CLOCK) {}
};
typedef boost::unordered_map<int, SchedulerBranchInfo *> SchedulerBranchInfoMap;
    /* Indexed by branch_id */

typedef vector<int> TunableIds;

class TunableSearcher;
class TunerLogic;

/* Define singleton */
class MltunerImpl;
extern MltunerImpl *mltuner_impl;

class MltunerImpl {
 public:
  shared_ptr<Communicator> communicator_;
  uint num_workers_;
  const Config& config_;

  vector<int> msg_payload_clocks_;
  vector<int> msg_payload_branch_ids_;

  tbb::tick_count global_start_tick_;

  int global_clock_;
  VecClock worker_clocks_;
      /* Indexed by worker_id */

  int next_branch_id_;

  int next_schedule_clock_;
  ClockSchedules clock_schedules_;

  SchedulerBranchInfoMap branch_map_;
  SchedulerBranchInfoMap active_branch_map_;

  shared_ptr<TunerLogic> logic_;
  int next_decision_clock_;

 public:
  static void CreateInstance(
      shared_ptr<Communicator> communicator,
      const Config& config) {
    if (mltuner_impl == NULL) {
      mltuner_impl = new MltunerImpl(communicator, config);
    }
  }
  MltunerImpl(
      shared_ptr<Communicator> communicator,
      const Config& config);
  void recv_worker_started(uint worker_id);
  void recv_worker_progress(
      uint worker_id, int clock, int branch_id, double progress);

  /* Utility functions */
  int make_branch(int parent_branch_id, const Tunable& tunable, int flag);
  int make_branch(
      int parent_branch_id, const TunableChoice& tunable_choice, int flag);
  void make_branches(
      int parent_branch_id, const vector<TunableChoice>& tunable_choices,
      int flag, vector<int>& branch_ids);
  void inactivate_branch(int branch_id);
  void schedule_run_time(int branch_id, double scheduled_run_time);
  void schedule_branch(int branch_id, int num_clocks);
  void send_branch_schedules();
  void interleave_branches(
      const vector<int>& branch_ids,
      uint num_clocks_per_branch, uint stride);
  void interleave_all_active_branches(
      uint num_clocks_per_branch, uint stride);
  Tunable make_tunable(const TunableChoice& tunable_choice);
  void run_one_epoch(int branch_id);

  /* Math utility functions */
  double get_interception(const DoubleVec& b, double bt);
  bool check_monotone(const DoubleVec& x);
  bool go_down_more_often_than_go_up(const DoubleVec& x);
  double calc_max_go_up(const DoubleVec& x);
  bool go_up_less_than_threshold(const DoubleVec& x, double threshold);
  double calc_slope(const DoubleVec& x, const DoubleVec& y);
  void downsample(const DoubleVec& input, int rate, DoubleVec& output);
  void downsample(
      const DoubleVec& input, int start, int end, int rate, DoubleVec& output);
  bool check_valid_progress(const DoubleVec& y);
  double calc_slope_with_check(const DoubleVec& x, const DoubleVec& y);
  pair<ConvergenceState, double> calc_slope_with_penalty(
      const DoubleVec& x, const DoubleVec& y, double threshold);
  pair<ConvergenceState, double> summarize_progress(
      double starting_time_offset, double starting_progress, int branch_id,
      double threshold);
  double calc_time_saving(int new_branch_id, int old_branch_id);
  double calc_time_saving(
      const DoubleVec& new_timeline, const DoubleVec& new_progress,
      const DoubleVec& old_timeline, const DoubleVec& old_progress);
  double average_progress(int branch_id);
  void summarize_runned_branch(
      int branch_id, double *starting_time_offset, double *starting_progress,
      double *time_to_try);
  bool decide_topk_converged(const DoubleVec& x, int slack, double threshold);
};

#endif  // defined __mltuner_impl_hpp__
