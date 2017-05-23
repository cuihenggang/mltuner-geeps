#ifndef __hyperband_infinite_logic_hpp__
#define __hyperband_infinite_logic_hpp__

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

class HyperbandInfiniteLogic : public TunerLogic {
  enum State {
    START_SEARCHING,
    RUNNING,
    TESTING,
  } state_;

  struct ExpInfo {
    uint exp_id;
    int branch_id;
    int epoch;
    DoubleVec accuracies;
    double peak_accuracy;
    ExpInfo() {}
    ExpInfo(uint exp_id, int branch_id)
        : exp_id(exp_id), branch_id(branch_id), epoch(0) {}
  };
  typedef boost::unordered_map<int, ExpInfo> ExpInfoMap;

  int best_branch_id_;
  int parent_branch_id_;
  int current_branch_id_;
  string searching_identity_;
  shared_ptr<TunableSearcher> tunable_searcher_;
  ExpInfoMap exp_info_map_;
  ExpInfo current_exp_info_;
  tbb::tick_count search_start_tick_;
  int training_branch_id_;
  int testing_branch_id_;

  /* HyperBand states */
  int K_;
  int LL_;
  int L_;
  int k_;
  int r_;

 public:
  HyperbandInfiniteLogic(
      MltunerImpl *impl, const Config& config);

  virtual void make_and_schedule_initial_branches();
  virtual void make_branch_decisions();

  void make_branch_decisions_start_initial_search();
  void state_machine__start_searching();
  void state_machine__running();
  void state_machine__testing();
  bool get_new_setting();
  void hyperband_start();
  void hyperband_update();
  void successive_halving_start();
  bool successive_halving_update();
  void run_exps();
};

#endif  // defined __hyperband_infinite_logic_hpp__
