/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>

#include <boost/format.hpp>

#include "common/internal-config.hpp"
#include "common/common-utils.hpp"

#include "mltuner/tuner-logics/hyperband-infinite-logic.hpp"
#include "mltuner/tunable-searchers/spearmint-searcher.hpp"
#include "mltuner/tunable-searchers/hyperopt-searcher.hpp"
#include "mltuner/tunable-searchers/grid-searcher.hpp"
#include "mltuner/tunable-searchers/random-searcher.hpp"
#include "mltuner/tunable-searchers/marginal-grid-searcher.hpp"
#include "mltuner/tunable-searchers/marginal-hyperopt-searcher.hpp"


HyperbandInfiniteLogic::HyperbandInfiniteLogic(MltunerImpl *impl, const Config& config)
    : TunerLogic(impl, config) {
}

void HyperbandInfiniteLogic::make_branch_decisions() {
  if (state_ == START_SEARCHING) {
    if (impl_->global_clock_ + 1 != impl_->next_decision_clock_) {
      CHECK_LT(impl_->global_clock_ + 1, impl_->next_decision_clock_);
      return;
    }
    state_machine__start_searching();
    impl_->send_branch_schedules();
    impl_->next_decision_clock_ = impl_->next_schedule_clock_;
    if (impl_->global_clock_ + 1 == impl_->next_decision_clock_) {
      make_branch_decisions();
    }
    return;
  }

  if (state_ == RUNNING) {
    state_machine__running();
    impl_->send_branch_schedules();
    impl_->next_decision_clock_ = impl_->next_schedule_clock_;
    return;
  }

  if (state_ == TESTING) {
    if (impl_->global_clock_ + 1 != impl_->next_decision_clock_) {
      CHECK_LT(impl_->global_clock_ + 1, impl_->next_decision_clock_);
      return;
    }
    state_machine__testing();
    impl_->send_branch_schedules();
    impl_->next_decision_clock_ = impl_->next_schedule_clock_;
    if (impl_->global_clock_ + 1 == impl_->next_decision_clock_) {
      make_branch_decisions();
    }
    return;
  }

  CHECK(0);
}

void HyperbandInfiniteLogic::make_and_schedule_initial_branches() {
  /* Make the initial branch */
  int parent_branch_id0 = -1;
  /* Empty tunable, just to init the param values */
  Tunable tunable;
  tunable.init_default(config_.mltuner_config.tunable_specs);
  int flag = INIT_BRANCH;
  int branch_id0 = impl_->make_branch(parent_branch_id0, tunable, flag);

  /* Schedule the initial branch to run for one clock.
   * We assume the application will set the initial parameter values
   * at this clock */
  uint num_clocks = 1;
  impl_->schedule_branch(branch_id0, num_clocks);

  parent_branch_id_ = branch_id0;
  state_ = START_SEARCHING;
  impl_->send_branch_schedules();
  impl_->next_decision_clock_ = impl_->next_schedule_clock_;
}

void HyperbandInfiniteLogic::state_machine__start_searching() {
  const string& tunable_searcher = config_.mltuner_config.tunable_searcher;
  if (tunable_searcher == "spearmint") {
    string app_name = config_.mltuner_config.app_name;
    string identity = "baseline";
    bool background_mode = false;
    bool rerun_duplicate = true;
    tunable_searcher_ = make_shared<SpearmintSearcher>(
        config_, app_name, identity, tunable_ids_,
        background_mode, rerun_duplicate);
  } else if (tunable_searcher == "grid") {
    tunable_searcher_ = make_shared<GridSearcher>(
        config_, tunable_ids_);
  } else if (tunable_searcher == "hyperopt") {
    string app_name = config_.mltuner_config.app_name;
    string identity = "baseline";
    tunable_searcher_ = make_shared<HyperoptSearcher>(
        config_, app_name, identity, tunable_ids_);
  } else if (tunable_searcher == "random") {
    tunable_searcher_ = make_shared<RandomSearcher>(
        config_, tunable_ids_);
  } else {
    CHECK(0) << "Unknown tunable searcher " << tunable_searcher;
  }
  tunable_searcher_->start();

  /* Initialize HyperBand states */
  hyperband_start();
  state_ = RUNNING;
}

void HyperbandInfiniteLogic::state_machine__running() {
  if (config_.mltuner_config.converge_type == "loss") {
    /* Check convergence with the loss */
    SchedulerBranchInfo *training_branch_info =
        impl_->branch_map_[training_branch_id_];
    CHECK(training_branch_info);
    bool converged =
        decide_loss_converged(training_branch_info->progress, config_);
    double converged_loss = *std::min_element(
        training_branch_info->progress.begin(),
        training_branch_info->progress.end());
    if (converged) {
      cout << "Stopping condition met, loss = " << converged_loss
           << endl << endl;
      if (config_.mltuner_config.stop_on_converge) {
        // CHECK(0);
        exit(0);
      }
    }
    // if (!converged) {
    if (true) {
      /* Schedule one more clock */
      int num_clocks_to_schedule = 1;
      impl_->schedule_branch(training_branch_id_, num_clocks_to_schedule);
      state_ = RUNNING;
      return;
    } else {
      /* Stop training */
    }
  }

  if (impl_->global_clock_ + 1 != impl_->next_decision_clock_) {
    CHECK_LT(impl_->global_clock_ + 1, impl_->next_decision_clock_);
    return;
  }

  if (config_.mltuner_config.converge_type == "val-accuracy") {
    /* Do testing */
    cout << "Do testing\n";
    int flag = TESTING_BRANCH;
    testing_branch_id_ = impl_->make_branch(
        training_branch_id_, config_.mltuner_config.tunable_for_test, flag);
    CHECK(config_.mltuner_config.num_clocks_to_test);
    impl_->schedule_branch(testing_branch_id_,
        config_.mltuner_config.num_clocks_to_test);
    state_ = TESTING;
    return;
  }

  CHECK(0);
}

void HyperbandInfiniteLogic::state_machine__testing() {
  /* Finish testing */
  CHECK_NE(testing_branch_id_, -1);
  ExpInfo& exp_info = exp_info_map_[training_branch_id_];
  exp_info.epoch++;
  cout << "Epoch " << exp_info.epoch << endl;
  double val_accuracy = impl_->average_progress(testing_branch_id_);
  cout << "Validation accuracy = " << val_accuracy << endl;
  exp_info.accuracies.push_back(val_accuracy);
  bool converged = decide_accuracy_plateaued(
      exp_info.accuracies,
      config_.mltuner_config.plateau_size_for_convergence);
  exp_info.peak_accuracy =
      *std::max_element(exp_info.accuracies.begin(), exp_info.accuracies.end());
  if (converged) {
    /* Don't need to keep running it if it's converged */
    exp_info.epoch = r_;
  }
  impl_->inactivate_branch(testing_branch_id_);
  testing_branch_id_ = -1;

  cout << "exp_info.epoch = " << exp_info.epoch << " r_ = " << r_ << endl;
  if (exp_info.epoch < r_) {
    /* Keep running more epoches */
    impl_->run_one_epoch(training_branch_id_);
    state_ = RUNNING;
  } else {
    /* Finished trying this setting */
    cout << "hyperband_update()\n";
    hyperband_update();
    state_ = RUNNING;
  }
}

bool HyperbandInfiniteLogic::get_new_setting() {
  search_start_tick_ = tbb::tick_count::now();
  TunableChoice tunable_choice;
  CHECK_EQ(current_exp_info_.accuracies.size(), 0);
  bool success = tunable_searcher_->get_result(
      &tunable_choice, &current_exp_info_.exp_id);
  if (!success) {
    return false;
  }
  int flag = TRAINING_BRANCH;
  current_exp_info_.branch_id =
      impl_->make_branch(parent_branch_id_, tunable_choice, flag);
  search_start_tick_ = tbb::tick_count::now();
  return true;
}

void HyperbandInfiniteLogic::hyperband_start() {
  K_ = 1;
  LL_ = 0;
  L_ = 1 << LL_;
  CHECK_GE(K_ - L_, LL_);
  successive_halving_start();
}

void HyperbandInfiniteLogic::hyperband_update() {
  bool still_running = successive_halving_update();
  if (still_running) {
    return;
  }

  LL_++;
  L_ = 1 << LL_;
  if (K_ - L_ >= LL_) {
    successive_halving_start();
    return;
  }

  K_++;
  LL_ = 0;
  L_ = 1 << LL_;
  CHECK_GE(K_ - L_, LL_);
  successive_halving_start();
}

void HyperbandInfiniteLogic::successive_halving_start() {
  k_ = 0;
  int B = 2 << K_;
  int n = 2 << L_;
  int Sk = 2 << (L_ - k_);
  r_ = B / Sk / L_;
  cout << "successive_halving_start:"
       << " K_ = " << K_ << " L_ = " << L_
       << " k_ = " << k_ << " n = " << n << " r_ = " << r_ << endl;

  for (int i = 0; i < n; i++) {
    bool success = get_new_setting();
    CHECK(success);
    exp_info_map_[current_exp_info_.branch_id] = current_exp_info_;
    exp_info_map_[current_exp_info_.branch_id].epoch = 0;
  }
  run_exps();
}

typedef pair<double, int> Pair;
struct PairCompare {
  bool operator () (const Pair& left, const Pair& right) {
    return left.first < right.first;
  }
};
typedef vector<Pair> Pairs;

bool HyperbandInfiniteLogic::successive_halving_update() {
  for (ExpInfoMap::iterator iter = exp_info_map_.begin();
       iter != exp_info_map_.end(); iter++) {
    int branch_id = iter->first;
    ExpInfo& exp_info = iter->second;
    if (exp_info.epoch < r_) {
      training_branch_id_ = branch_id;
      impl_->run_one_epoch(branch_id);
      return true;
    }
  }

  /* Kill half of the exps according to the peak accuracy */
  Pairs pairs;
  PairCompare pair_compare;
  for (ExpInfoMap::iterator iter = exp_info_map_.begin();
       iter != exp_info_map_.end(); iter++) {
    int branch_id = iter->first;
    ExpInfo& exp_info = iter->second;
    double accuracy = exp_info.peak_accuracy;
    pairs.push_back(Pair(accuracy, branch_id));
  }
  std::sort(pairs.begin(), pairs.end(), pair_compare);
  for (int i = 0; i < pairs.size() / 2; i++) {
    int branch_id = pairs[i].second;
    impl_->inactivate_branch(branch_id);
    exp_info_map_.erase(branch_id);
  }

  k_++;
  if (k_ <= L_ - 1) {
    int B = 2 << K_;
    int Sk = 2 << (L_ - k_);
    r_ = B / Sk / L_;
    cout << "successive_halving_update:"
         << " K_ = " << K_ << " L_ = " << L_
         << " k_ = " << k_ << " r_ = " << r_ << endl;
    run_exps();
    return true;
  }

  /* SucessiveHalving finished, should have only one exp left */
  for (ExpInfoMap::iterator iter = exp_info_map_.begin();
       iter != exp_info_map_.end(); iter++) {
    int branch_id = iter->first;
    impl_->inactivate_branch(branch_id);
  }
  exp_info_map_.clear();
  return false;
}

void HyperbandInfiniteLogic::run_exps() {
  for (ExpInfoMap::iterator iter = exp_info_map_.begin();
       iter != exp_info_map_.end(); iter++) {
    int branch_id = iter->first;
    ExpInfo& exp_info = iter->second;
    if (exp_info.epoch < r_) {
      training_branch_id_ = branch_id;
      impl_->run_one_epoch(branch_id);
      return;
    }
  }
  CHECK(0);
}
