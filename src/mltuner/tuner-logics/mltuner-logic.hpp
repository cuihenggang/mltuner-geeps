#ifndef __mltuner_logic_hpp__
#define __mltuner_logic_hpp__

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

#include "mltuner/tuner-logic.hpp"

class MltunerLogic : public TunerLogic {
  enum State {
    GET_STARTING_PROGRESS,
    START_SEARCHING,
    SEARCHING,
    RUNNING,
    TESTING,
  } state_;

  struct ExpInfo {
    uint exp_id;
    int branch_id;
    ExpInfo() {}
    ExpInfo(uint exp_id, int branch_id)
        : exp_id(exp_id), branch_id(branch_id) {}
  };
  typedef boost::unordered_map<int, ExpInfo> ExpInfoMap;

  int best_branch_id_;
  int parent_branch_id_;
  int current_branch_id_;
  string searching_identity_;
  bool early_adjust_;
  double starting_time_offset_;
  double starting_progress_;
  double time_to_try_;
  double searcher_time_;
  shared_ptr<TunableSearcher> tunable_searcher_;
  ExpInfoMap exp_info_map_;
  ExpInfo current_exp_info_;
  tbb::tick_count search_start_tick_;
  tbb::tick_count tick_since_last_best_;
  int training_branch_id_;
  int testing_branch_id_;
  uint num_clocks_to_try_;
  uint num_clocks_to_run_;
  double best_progress_slope_;
  double last_time_saving_;
  double search_time_to_get_last_best_;
  double time_for_next_search_;
  int num_trials_;
  int num_trials_bound_;
  int num_epoches_;
  DoubleVec val_accuracies_;

  uint num_retunes_;
  TunableChoice baseline_tunable_choice_;
  TunableIds tunable_id_refine_order_;
  uint tunable_id_refine_order_idx_;
  int tunable_id_to_search_;

 public:
  MltunerLogic(
      MltunerImpl *impl, const Config& config);

  virtual void make_and_schedule_initial_branches();
  virtual void make_branch_decisions();

  void make_branch_decisions_start_initial_search();
  void state_machine__get_starting_progress();
  void start_getting_starting_progress();
  void state_machine__start_searching();
  void summarize_starting_progress();
  void start_searching();
  void state_machine__searching();
  void find_any_valid_progress();
  void finish_searching();
  void summarize_exp();
  void state_machine__running();
  void state_machine__testing();
  bool try_new_setting();
};

#endif  // defined __mltuner_logic_hpp__
