/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>

#include <boost/format.hpp>

#include "common/internal-config.hpp"
#include "common/common-utils.hpp"

#include "mltuner/tuner-logics/spearmint-logic.hpp"
#include "mltuner/tunable-searchers/spearmint-searcher.hpp"
#include "mltuner/tunable-searchers/hyperopt-searcher.hpp"
#include "mltuner/tunable-searchers/grid-searcher.hpp"
#include "mltuner/tunable-searchers/random-searcher.hpp"
#include "mltuner/tunable-searchers/marginal-grid-searcher.hpp"
#include "mltuner/tunable-searchers/marginal-hyperopt-searcher.hpp"


SpearmintLogic::SpearmintLogic(MltunerImpl *impl, const Config& config)
    : TunerLogic(impl, config) {
}

void SpearmintLogic::make_branch_decisions() {
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

void SpearmintLogic::make_and_schedule_initial_branches() {
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

void SpearmintLogic::state_machine__start_searching() {
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
  } else {
    CHECK(0) << "Unknown tunable searcher " << tunable_searcher;
  }

  tunable_searcher_->start();
  bool success = try_new_setting();
  CHECK(success);
  state_ = RUNNING;
}

void SpearmintLogic::state_machine__running() {
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

void SpearmintLogic::state_machine__testing() {
  /* Finish testing */
  CHECK_NE(testing_branch_id_, -1);
  ExpInfo& exp_info = current_exp_info_;
  exp_info.epoch++;
  cout << "Epoch " << exp_info.epoch << endl;
  double val_accuracy = impl_->average_progress(testing_branch_id_);
  cout << "Validation accuracy = " << val_accuracy << endl;
  exp_info.accuracies.push_back(val_accuracy);
  bool converged = decide_accuracy_plateaued(
      exp_info.accuracies,
      config_.mltuner_config.plateau_size_for_convergence);

  double peak_accuracy =
      *std::max_element(exp_info.accuracies.begin(), exp_info.accuracies.end());
  if (converged) {
    cout << "Stopping condition met, accuracy = " << peak_accuracy
         << endl << endl;
  }
  impl_->inactivate_branch(testing_branch_id_);
  testing_branch_id_ = -1;

  /* Run to completion with chooser */
  if (!converged) {
    impl_->run_one_epoch(training_branch_id_);
    state_ = RUNNING;
  } else {
    /* Report accuracy to searcher */
    double training_time =
        (tbb::tick_count::now() - search_start_tick_).seconds();
    double converged_error = 0.0 - peak_accuracy;
    tunable_searcher_->set_result(
        current_exp_info_.exp_id, training_time, converged_error);
    cout << "Training time = " << training_time << endl;
    impl_->inactivate_branch(training_branch_id_);
    training_branch_id_ = -1;
    bool success = try_new_setting();
    if (!success) {
      cout << "Done\n";
      if (system("pdsh -R ssh -w h[1-7] \"pkill -9 geeps\"")) {
        cerr << "Non-zero syscall return value\n";
      }
      if (system("pdsh -R ssh -w h0 \"pkill -9 geeps\"")) {
        cerr << "Non-zero syscall return value\n";
      }
      exit(0);
    }
    state_ = RUNNING;
  }
}

bool SpearmintLogic::try_new_setting() {
  search_start_tick_ = tbb::tick_count::now();
  TunableChoice tunable_choice;
  current_exp_info_.epoch = 0;
  current_exp_info_.accuracies.clear();
  bool success = tunable_searcher_->get_result(
      &tunable_choice, &current_exp_info_.exp_id);
  if (!success) {
    return false;
  }
  int flag = TRAINING_BRANCH;
  current_exp_info_.branch_id =
      impl_->make_branch(parent_branch_id_, tunable_choice, flag);
  training_branch_id_ = current_exp_info_.branch_id;
  impl_->run_one_epoch(training_branch_id_);
  search_start_tick_ = tbb::tick_count::now();
  return true;
}
