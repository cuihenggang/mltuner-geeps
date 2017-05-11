/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>

#include <boost/format.hpp>

#include "common/internal-config.hpp"
#include "common/common-utils.hpp"

#include "mltuner/tuner-logics/mltuner-logic.hpp"
#include "mltuner/tunable-searchers/spearmint-searcher.hpp"
#include "mltuner/tunable-searchers/hyperopt-searcher.hpp"
#include "mltuner/tunable-searchers/grid-searcher.hpp"
#include "mltuner/tunable-searchers/random-searcher.hpp"
#include "mltuner/tunable-searchers/marginal-grid-searcher.hpp"
#include "mltuner/tunable-searchers/marginal-hyperopt-searcher.hpp"


MltunerLogic::MltunerLogic(MltunerImpl *impl, const Config& config)
    : TunerLogic(impl, config) {
}

void MltunerLogic::make_branch_decisions() {
  if (state_ == GET_STARTING_PROGRESS) {
    if (impl_->global_clock_ + 1 != impl_->next_decision_clock_) {
      CHECK_LT(impl_->global_clock_ + 1, impl_->next_decision_clock_);
      return;
    }
    state_machine__get_starting_progress();
    impl_->send_branch_schedules();
    impl_->next_decision_clock_ = impl_->next_schedule_clock_;
    if (impl_->global_clock_ + 1 == impl_->next_decision_clock_) {
      make_branch_decisions();
    }
    return;
  }

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

  if (state_ == SEARCHING) {
    if (impl_->global_clock_ + 1 != impl_->next_decision_clock_) {
      CHECK_LT(impl_->global_clock_ + 1, impl_->next_decision_clock_);
      return;
    }
    state_machine__searching();
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

void MltunerLogic::make_and_schedule_initial_branches() {
  num_retunes_ = 0;

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
  state_ = GET_STARTING_PROGRESS;
  impl_->send_branch_schedules();
  impl_->next_decision_clock_ = impl_->next_schedule_clock_;
}

void MltunerLogic::state_machine__get_starting_progress() {
  if (tunable_ids_.size() == 0) {
    /* No tunables to search, so we just run with the default */
    Tunable tunable;
    tunable.init_default(config_.mltuner_config.tunable_specs);
    int flag = TRAINING_BRANCH;
    training_branch_id_ =
        impl_->make_branch(parent_branch_id_, tunable, flag);
    impl_->run_one_epoch(training_branch_id_);
    state_ = RUNNING;
    return;
  }

  if (!config_.mltuner_config.initial_tuning) {
    /* Don't need to do the initial tuning, just use the hard coded one */
    int flag = TRAINING_BRANCH;
    training_branch_id_ =
        impl_->make_branch(parent_branch_id_,
            config_.mltuner_config.tunable_for_train, flag);
    impl_->run_one_epoch(training_branch_id_);
    state_ = RUNNING;
    return;
  }

  /* Start the initial search */
  searching_identity_ = "initial_search";
  /* We will initialize time_to_try with searcher_time
   * in try_new_setting(). */
  time_to_try_ = 0.0;
  num_trials_bound_ = -1;
  start_getting_starting_progress();
  state_ = START_SEARCHING;
}

void MltunerLogic::start_getting_starting_progress() {
  // int flag = EVAL_BRANCH;
  int flag = TRAINING_BRANCH;
  current_branch_id_ = impl_->make_branch(
      parent_branch_id_, config_.mltuner_config.tunable_for_eval, flag);
  uint num_clocks_to_run = config_.mltuner_config.num_clocks_to_eval;
  impl_->schedule_branch(current_branch_id_, num_clocks_to_run);
}

void MltunerLogic::state_machine__start_searching() {
  /* Summarize starting progress */
  summarize_starting_progress();

  start_searching();
  state_ = SEARCHING;
}

void MltunerLogic::summarize_starting_progress() {
  starting_time_offset_ = 0.0;
  starting_progress_ = impl_->average_progress(current_branch_id_);
  cout << "starting_progress = " << starting_progress_ << endl;
  impl_->inactivate_branch(current_branch_id_);
  current_branch_id_ = -1;
}

void MltunerLogic::start_searching() {
  CHECK(tunable_ids_.size());
  search_start_tick_ = tbb::tick_count::now();
  best_branch_id_ = -1;
  best_progress_slope_ = std::numeric_limits<double>::quiet_NaN();
  last_time_saving_ = std::numeric_limits<double>::quiet_NaN();
  exp_info_map_.clear();
  num_trials_ = 0;
  tick_since_last_best_ = tbb::tick_count::now();
  if (searching_identity_ == "initial_search") {
    const string& tunable_searcher = config_.mltuner_config.tunable_searcher;
    if (tunable_searcher == "spearmint") {
      string app_name = config_.mltuner_config.app_name;
      bool background_mode = true;
      bool rerun_duplicate = true;
      tunable_searcher_ = make_shared<SpearmintSearcher>(
          config_, app_name, searching_identity_, tunable_ids_,
          background_mode, rerun_duplicate);
    } else if (tunable_searcher == "grid") {
        tunable_searcher_ = make_shared<GridSearcher>(
            config_, tunable_ids_);
    } else if (tunable_searcher == "hyperopt") {
        string app_name = config_.mltuner_config.app_name;
        tunable_searcher_ = make_shared<HyperoptSearcher>(
            config_, app_name, searching_identity_, tunable_ids_);
    } else if (tunable_searcher == "random") {
      tunable_searcher_ = make_shared<RandomSearcher>(
          config_, tunable_ids_);
    } else {
      CHECK(0) << "Unknown tunable searcher " << tunable_searcher;
    }
  } else {
    /* Re-tune tunable */
    const string& tunable_searcher =
        config_.mltuner_config.tunable_searcher_for_retune != "" ?
            config_.mltuner_config.tunable_searcher_for_retune :
            config_.mltuner_config.tunable_searcher;
    if (tunable_searcher == "spearmint") {
      string app_name = config_.mltuner_config.app_name;
      bool background_mode = true;
      bool rerun_duplicate = true;
      tunable_searcher_ = make_shared<SpearmintSearcher>(
          config_, app_name, searching_identity_, tunable_ids_,
          background_mode, rerun_duplicate);
    } else if (tunable_searcher == "grid") {
      tunable_searcher_ = make_shared<GridSearcher>(
          config_, tunable_ids_);
    } else if (tunable_searcher == "hyperopt") {
      string app_name = config_.mltuner_config.app_name;
      tunable_searcher_ = make_shared<HyperoptSearcher>(
          config_, app_name, searching_identity_, tunable_ids_);
    } else if (tunable_searcher == "marginal-grid") {
      SchedulerBranchInfo *parent_branch_info =
          impl_->branch_map_[parent_branch_id_];
      CHECK(parent_branch_info);
      TunableChoice& base_tunable_choice = parent_branch_info->tunable_choice;
      tunable_searcher_ = make_shared<MarginalGridSearcher>(
          config_, tunable_ids_, base_tunable_choice);
    } else if (tunable_searcher == "marginal-hyperopt") {
      SchedulerBranchInfo *parent_branch_info =
          impl_->branch_map_[parent_branch_id_];
      CHECK(parent_branch_info);
      TunableChoice& base_tunable_choice = parent_branch_info->tunable_choice;
      string app_name = config_.mltuner_config.app_name;
      tunable_searcher_ = make_shared<MarginalHyperoptSearcher>(
          config_, app_name, searching_identity_,
          tunable_ids_, base_tunable_choice);
    } else {
      CHECK(0) << "Unknown tunable searcher " << tunable_searcher;
    }
  }
  tunable_searcher_->start();
  bool success = try_new_setting();
  CHECK(success);
  exp_info_map_[current_exp_info_.branch_id] = current_exp_info_;
}

void MltunerLogic::find_any_valid_progress() {
  CHECK_EQ(best_branch_id_, -1);
  CHECK(isnan(best_progress_slope_));
  int best_branch_id_allow_unstable = -1;
  double best_progress_slope_allow_unstable =
      std::numeric_limits<double>::quiet_NaN();
  for (ExpInfoMap::iterator it = exp_info_map_.begin();
       it != exp_info_map_.end(); it++) {
    /* Summarize experiment result */
    const ExpInfo& exp_info = it->second;
    double threshold = GO_UP_THRESHOLD;
    pair<ConvergenceState, double> ret = impl_->summarize_progress(
        starting_time_offset_, starting_progress_,
        exp_info.branch_id, threshold);
    ConvergenceState convergence_state = ret.first;
    double progress_slope = ret.second;
    cout << "convergence_state = " << convergence_state << endl;
    cout << "progress_slope = " << progress_slope << endl;
    /* Set experiment result */
    tunable_searcher_->set_result(exp_info.exp_id, 1.0, progress_slope);
    if (convergence_state == CONVERGING && !isnan(progress_slope)
          && (isnan(best_progress_slope_)
            || progress_slope < best_progress_slope_)) {
      /* Update best branch */
      best_branch_id_ = exp_info.branch_id;
      best_progress_slope_ = progress_slope;
    }
    if (!isnan(progress_slope)
          && (isnan(best_progress_slope_allow_unstable)
            || progress_slope < best_progress_slope_allow_unstable)) {
      /* Update best branch */
      best_branch_id_allow_unstable = exp_info.branch_id;
      best_progress_slope_allow_unstable = progress_slope;
    }
  }
  cout << "best_branch_id_ = " << best_branch_id_ << endl;
  cout << "best_branch_id_allow_unstable = "
       << best_branch_id_allow_unstable << endl;
  if (best_branch_id_ != -1) {
    /* First valid progress got */
    cout << "First valid progress got\n";
    search_time_to_get_last_best_ =
        (tbb::tick_count::now() - tick_since_last_best_).seconds();
    tick_since_last_best_ = tbb::tick_count::now();
    /* Inactivate the other branches being tried */
    for (ExpInfoMap::iterator it = exp_info_map_.begin();
       it != exp_info_map_.end(); it++) {
      const ExpInfo& exp_info = it->second;
      if (exp_info.branch_id != best_branch_id_) {
        impl_->inactivate_branch(exp_info.branch_id);
      }
    }
    /* Set time_to_try to the run time of the best branch */
    SchedulerBranchInfo *best_branch_info = impl_->branch_map_[best_branch_id_];
    CHECK(best_branch_info);
    double branch_run_time =
        best_branch_info->timeline.back() - best_branch_info->timeline.front();
    time_to_try_ = branch_run_time;
    /* TODO: we should also have a mechanism to decrease the number of clocks to try */
    cout << "time_to_try set to " << time_to_try_ << endl;
    /* Keep searching */
    bool success = try_new_setting();
    if (success) {
      state_ = SEARCHING;
      return;
    } else {
      finish_searching();
      return;
    }
  } else {
    bool trial_time_bound_reached = true;
    /* Kill the branches that have reached trial time bound or diverged */
    vector<int> branch_ids_to_kill;
    for (ExpInfoMap::const_iterator it = exp_info_map_.begin();
       it != exp_info_map_.end(); it++) {
      const ExpInfo& exp_info = it->second;
      SchedulerBranchInfo *branch_info = impl_->branch_map_[exp_info.branch_id];
      CHECK(branch_info) << " exp_info.branch_id";
      if (branch_info->convergence_state == DIVERGED) {
        branch_ids_to_kill.push_back(exp_info.branch_id);
        continue;
      }
      int max_trial_clocks =
          clocks_per_epoch(branch_info->tunable, config_);
      if (branch_info->internal_clock >= max_trial_clocks) {
        if (exp_info.branch_id != best_branch_id_allow_unstable) {
          branch_ids_to_kill.push_back(exp_info.branch_id);
        }
      } else {
        trial_time_bound_reached = false;
      }
    }
    for (int i = 0; i < branch_ids_to_kill.size(); i++) {
      impl_->inactivate_branch(branch_ids_to_kill[i]);
      exp_info_map_.erase(branch_ids_to_kill[i]);
    }

    /* Double time_to_try */
    double max_branch_run_time = 0.0;
    for (ExpInfoMap::const_iterator it = exp_info_map_.begin();
       it != exp_info_map_.end(); it++) {
      const ExpInfo& exp_info = it->second;
      SchedulerBranchInfo *branch_info = impl_->branch_map_[exp_info.branch_id];
      CHECK(branch_info) << " exp_info.branch_id";
      double branch_run_time =
          branch_info->timeline.back() - branch_info->timeline.front();
      max_branch_run_time = branch_run_time > max_branch_run_time ?
          branch_run_time : max_branch_run_time;
    }
    // time_to_try_ = max_branch_run_time * 2;
    if (!trial_time_bound_reached) {
      time_to_try_ *= 2;
      cout << "time_to_try doubled to " << time_to_try_ << endl;
    }

    /* Schedule more clocks */
    bool more_clocks_to_schedule = false;
    bool success = try_new_setting();
    /* Schedule all the exps being tried for longer */
    cout << "Schedule all the exps being tried for longer\n";
    for (ExpInfoMap::const_iterator it = exp_info_map_.begin();
       it != exp_info_map_.end(); it++) {
      const ExpInfo& exp_info = it->second;
      SchedulerBranchInfo *branch_info = impl_->branch_map_[exp_info.branch_id];
      CHECK(branch_info) << " exp_info.branch_id";
      CHECK_NE(branch_info->convergence_state, DIVERGED);
      double branch_run_time =
          branch_info->timeline.back() - branch_info->timeline.front();
      int max_trial_clocks =
          clocks_per_epoch(branch_info->tunable, config_);
      if (branch_run_time < time_to_try_) {
        CHECK(branch_info->time_per_clock);
        int num_clocks = round(
            (time_to_try_ - branch_run_time) / branch_info->time_per_clock);
        if (num_clocks < 0) {
          cout << "***WARNING: num_clocks overflow!\n";
          cout << "num_clocks = " << num_clocks << endl;
          cout << "time_to_try_ = " << time_to_try_ << endl;
          cout << "branch_run_time = " << branch_run_time << endl;
          cout << "branch_info->time_per_clock = " << branch_info->time_per_clock << endl;
          num_clocks = max_trial_clocks;
        }
        cout << "num_clocks = " << num_clocks << endl;
        cout << "branch_info->internal_clock = " << branch_info->internal_clock << endl;
        if (num_clocks + branch_info->internal_clock > max_trial_clocks) {
          /* Bound the number of trial clocks to be less than an epoch */
          num_clocks = max_trial_clocks - branch_info->internal_clock;
          CHECK_GE(num_clocks, 0);
        }
        if (num_clocks) {
          cout << "Schedule branch " << exp_info.branch_id
               << " for " << num_clocks << " clocks" << endl;
          impl_->schedule_branch(exp_info.branch_id, num_clocks);
          more_clocks_to_schedule = true;
        }
      }
    }
    /* Append the new branch to the list */
    if (success) {
      exp_info_map_[current_exp_info_.branch_id] = current_exp_info_;
      more_clocks_to_schedule = true;
    } else {
      cout << "WARNING: no more tunables to try\n";
    }

    if (!trial_time_bound_reached && !more_clocks_to_schedule) {
      cout << "WARNING: no more clocks to schedule, but trial time bound not reached\n";
    }
    if (!more_clocks_to_schedule) {
      cout << "Stop searching early\n";
      CHECK(trial_time_bound_reached);
      if (best_branch_id_allow_unstable == -1) {
        /* Converged */
        cout << "Converged\n";
        if (system("pdsh -R ssh -w h[1-7] \"pkill -9 geeps\"")) {
          cerr << "Non-zero syscall return value\n";
        }
        if (system("pdsh -R ssh -w h0 \"pkill -9 geeps\"")) {
          cerr << "Non-zero syscall return value\n";
        }
        exit(0);
      }
      /* Use current best tunable (even it's not stable) */
      CHECK_NE(best_branch_id_allow_unstable, -1);
      best_branch_id_ = best_branch_id_allow_unstable;
      /* Kill the other branches */
      for (ExpInfoMap::const_iterator it = exp_info_map_.begin();
          it != exp_info_map_.end(); it++) {
        const ExpInfo& exp_info = it->second;
        if (exp_info.branch_id != best_branch_id_) {
          impl_->inactivate_branch(exp_info.branch_id);
        }
      }
      finish_searching();
      return;
    }

    state_ = SEARCHING;
    return;
  }
}

void MltunerLogic::state_machine__searching() {
  if (best_branch_id_ == -1) {
    find_any_valid_progress();
    return;
  }

  summarize_exp();

  bool keep_searching = try_new_setting();
  if (keep_searching) {
    state_ = SEARCHING;
  } else {
    /* Stop the search if searcher has no more tunables to propose */
    cout << "Stop searching\n";
    finish_searching();
  }
}

void MltunerLogic::finish_searching(){
  /* Inactivate parent branch */
  impl_->inactivate_branch(parent_branch_id_);
  tunable_searcher_->stop();
  double total_search_time =
      (tbb::tick_count::now() - search_start_tick_).seconds();
  double time_planned_to_run = total_search_time * RUN_TO_TRY_RATIO;
  double time = (tbb::tick_count::now() - impl_->global_start_tick_).seconds();
  time_for_next_search_ = time + time_planned_to_run;
  /* We bound the number of trials of each search
   * to be less than the last one */
  num_trials_bound_ = num_trials_;
  training_branch_id_ = best_branch_id_;
  SchedulerBranchInfo *training_branch_info =
      impl_->branch_map_[training_branch_id_];
  CHECK(training_branch_info);
  const Tunable& training_tunable = training_branch_info->tunable;
  cout << "Run for " << time_planned_to_run << " sec"
       << " (" << time_planned_to_run << " time)"
       << " with tunable" << endl;
  training_tunable.print();
  CHECK_EQ(training_branch_info->internal_clock,
      training_branch_info->num_clocks_scheduled);
  int num_clocks_already_run = training_branch_info->internal_clock;
  if (config_.mltuner_config.converge_type == "val-accuracy") {
    int num_clocks_per_test =
        clocks_per_test(training_tunable, config_);
    int num_clocks_to_schedule =
        num_clocks_per_test - num_clocks_already_run % num_clocks_per_test;
    impl_->schedule_branch(training_branch_id_, num_clocks_to_schedule);
    state_ = RUNNING;
    return;
  }
  if (config_.mltuner_config.converge_type == "loss") {
    SchedulerBranchInfo *training_branch_info =
        impl_->branch_map_[training_branch_id_];
    CHECK(training_branch_info);
    /* TODO: this should be a different config */
    int num_clocks_to_schedule =
        config_.mltuner_config.plateau_size_for_convergence;
    impl_->schedule_branch(training_branch_id_, num_clocks_to_schedule);
    state_ = RUNNING;
    return;
  }
  CHECK(0);
}

void MltunerLogic::summarize_exp() {
  /* Summarize experiment result */
  // double threshold = 1.0;
  double threshold = GO_UP_THRESHOLD;
  pair<ConvergenceState, double> ret = impl_->summarize_progress(
      starting_time_offset_, starting_progress_,
      current_exp_info_.branch_id, threshold);
  ConvergenceState convergence_state = ret.first;
  double progress_slope = ret.second;
  cout << "convergence_state = " << convergence_state << endl;
  cout << "progress_slope = " << progress_slope << endl;
  /* Set experiment result */
  tunable_searcher_->set_result(current_exp_info_.exp_id, 1.0, progress_slope);

  /* Update best branch */
  if (!isnan(progress_slope)) {
    if (best_branch_id_ == -1) {
      /* Best branch updated */
      best_branch_id_ = current_exp_info_.branch_id;
      best_progress_slope_ = progress_slope;
      search_time_to_get_last_best_ =
          (tbb::tick_count::now() - tick_since_last_best_).seconds();
      tick_since_last_best_ = tbb::tick_count::now();
      cout << "Best tunable updated to\n";
    } else {
      /* Note that both slopes are negative numbers,
       * so "progress_slope < best_progress_slope_" means that
       * "abs(progress_slope) > abs(best_progress_slope_)" */
      bool better = progress_slope < best_progress_slope_;
      if (better) {
        /* Best branch updated */
        impl_->inactivate_branch(best_branch_id_);
        best_branch_id_ = current_exp_info_.branch_id;
        best_progress_slope_ = progress_slope;
        search_time_to_get_last_best_ =
            (tbb::tick_count::now() - tick_since_last_best_).seconds();
        tick_since_last_best_ = tbb::tick_count::now();
        cout << "Best tunable updated\n";
      }
    }
  }
  if (current_exp_info_.branch_id != best_branch_id_) {
    impl_->inactivate_branch(current_exp_info_.branch_id);
  }
}

void MltunerLogic::state_machine__running() {
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

void MltunerLogic::state_machine__testing() {
  /* Finish testing */
  CHECK_NE(testing_branch_id_, -1);
  num_epoches_++;
  cout << "Epoch " << num_epoches_ << endl;
  double val_accuracy = impl_->average_progress(testing_branch_id_);
  cout << "Validation accuracy = " << val_accuracy << endl;
  val_accuracies_.push_back(val_accuracy);
  bool converged = decide_accuracy_plateaued(
      val_accuracies_,
      config_.mltuner_config.plateau_size_for_convergence);
  bool retune_tunable = false;
  if (config_.mltuner_config.retune_tunable) {
    /* Re-tune tunables when the top K accuracies plateau */
    retune_tunable = decide_accuracy_plateaued(
        val_accuracies_,
        config_.mltuner_config.plateau_size_for_retune);
  }
  if (config_.mltuner_config.force_retune > 0) {
    if (num_epoches_ == config_.mltuner_config.force_retune) {
      retune_tunable = true;
    }
  }

  double peak_accuracy =
      *std::max_element(val_accuracies_.begin(), val_accuracies_.end());
  if (converged) {
    cout << "Stopping condition met, accuracy = " << peak_accuracy
         << endl << endl;
    if (config_.mltuner_config.stop_on_converge) {
      if (system("pdsh -R ssh -w h[1-7] \"pkill -9 geeps\"")) {
          cerr << "Non-zero syscall return value\n";
        }
      if (system("pdsh -R ssh -w h0 \"pkill -9 geeps\"")) {
        cerr << "Non-zero syscall return value\n";
      }
      exit(0);
    }
  }
  impl_->inactivate_branch(testing_branch_id_);
  testing_branch_id_ = -1;

  /* Re-tune tunable */
  if (retune_tunable) {
    /* Re-tuning the tunable */
    cout << "Try re-tuning the tunables\n";
    parent_branch_id_ = training_branch_id_;
    searching_identity_ =
        (boost::format("searching_%i") % num_retunes_++).str();
    time_to_try_ = 0.0;
    start_getting_starting_progress();
    state_ = START_SEARCHING;
    return;
  }

  /* Schedule until the next test */
  CHECK(!retune_tunable);
  impl_->run_one_epoch(training_branch_id_);
  state_ = RUNNING;
  return;
}

bool MltunerLogic::try_new_setting() {
  if (num_trials_bound_ > 0 && num_trials_ >= num_trials_bound_) {
    /* We bound the number of trials of each search
     * to be less than the last one */
    cout << "num_trials_bound_ reached\n";
    return false;
  }
  TunableChoice tunable_choice;
  bool success = tunable_searcher_->get_result(
      &tunable_choice, &current_exp_info_.exp_id);
  if (!success) {
    return false;
  }
  int flag = TRAINING_BRANCH;
  current_exp_info_.branch_id =
      impl_->make_branch(parent_branch_id_, tunable_choice, flag);
  if (time_to_try_ == 0.0) {
    /* Initialize time_to_try with searcher_time */
    searcher_time_ = tunable_searcher_->get_searcher_time();
    time_to_try_ = searcher_time_;
    // time_to_try_ = config_.mltuner_config.time_to_try;
    cout << "time_to_try initialized to " << time_to_try_ << endl;
  }
  impl_->schedule_run_time(current_exp_info_.branch_id, time_to_try_);
  impl_->schedule_branch(current_exp_info_.branch_id,
      config_.mltuner_config.num_progress_samples);
  num_trials_++;
  return true;
}
