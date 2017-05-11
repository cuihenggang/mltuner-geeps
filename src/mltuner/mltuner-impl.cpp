/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>

#include <boost/format.hpp>

#include "common/internal-config.hpp"
#include "common/common-utils.hpp"
#include "mltuner-impl.hpp"
#include "tuner-logic.hpp"
#include "tuner-logics/mltuner-logic.hpp"
#include "tuner-logics/spearmint-logic.hpp"
#include "tuner-logics/tupaq-logic.hpp"
#include "tuner-logics/hyperband-logic.hpp"

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

MltunerImpl *mltuner_impl = NULL;

MltunerImpl::MltunerImpl(
    shared_ptr<Communicator> communicator,
    const Config& config_) :
      communicator_(communicator),
      num_workers_(config_.num_processes), config_(config_) {
  global_clock_ = UNINITIALIZED_CLOCK;
  worker_clocks_.resize(num_workers_);
  for (uint i = 0; i < worker_clocks_.size(); i++) {
    worker_clocks_[i] = UNINITIALIZED_CLOCK;
  }

  next_branch_id_ = 0;
  if (config_.mltuner_config.tuner_logic == "mltuner") {
    logic_ = make_shared<MltunerLogic>(this, config_);
  } else if (config_.mltuner_config.tuner_logic == "spearmint") {
    logic_ = make_shared<SpearmintLogic>(this, config_);
  } else if (config_.mltuner_config.tuner_logic == "tupaq") {
    logic_ = make_shared<TupaqLogic>(this, config_);
  } else if (config_.mltuner_config.tuner_logic == "hyperband") {
    logic_ = make_shared<HyperbandLogic>(this, config_);
  } else {
    CHECK(0) << " unknown tuner_logic = "
        << config_.mltuner_config.tuner_logic;
  }
}

void MltunerImpl::recv_worker_started(uint worker_id) {
  cout << "Worker " << worker_id << " started" << endl;

  CHECK_LT(worker_id, worker_clocks_.size());
  /* The worker_clock might not necessarily be UNINITIALIZED_CLOCK,
   * because the client might resend the worker start message. */
  if (worker_clocks_[worker_id] == UNINITIALIZED_CLOCK) {
    worker_clocks_[worker_id] = INITIAL_CLOCK -1;
  }
  for (uint i = 0; i < worker_clocks_.size(); i++) {
    cout << "Worker " << i << " started : "
         << (worker_clocks_[i] == UNINITIALIZED_CLOCK ?
             "no" : "yes") << endl;
  }
  int new_global_clock = clock_min(worker_clocks_);
  if (global_clock_ == UNINITIALIZED_CLOCK
      && new_global_clock != global_clock_) {
    global_clock_ = new_global_clock;
    next_schedule_clock_ = global_clock_ + 1;
    logic_->make_and_schedule_initial_branches();
    global_start_tick_ = tbb::tick_count::now();
  }
  // if (global_clock_ == UNINITIALIZED_CLOCK && new_global_clock == global_clock_) {
    // for (uint i = 0; i < worker_clocks_.size(); i++) {
      // if (worker_clocks_[i] == UNINITIALIZED_CLOCK) {
        // cout << "Waiting for worker " << i << endl;
      // }
    // }
  // }
}

void MltunerImpl::recv_worker_progress(
    uint worker_id, int clock, int branch_id, double progress) {
  // cout << "worker " << worker_id
      // << " at clock " << clock
      // << " and branch " << branch_id
      // << " has progress " << progress
      // << endl;

  CHECK_LT(worker_id, worker_clocks_.size());
  CHECK_EQ(clock, worker_clocks_[worker_id] + 1);
  worker_clocks_[worker_id] = clock;
  int new_global_clock = clock_min(worker_clocks_);
  
  CHECK_LT(clock, clock_schedules_.size());
  ClockSchedule clock_schedule = clock_schedules_[clock];
  CHECK_EQ(branch_id, clock_schedule.branch_id);

  SchedulerBranchInfo *branch_info = branch_map_[branch_id];
  CHECK(branch_info);
  int branch_internal_clock = clock_schedule.internal_clock;
  while (branch_internal_clock >= branch_info->live_progress.size()) {
    CHECK_EQ(branch_internal_clock, branch_info->live_progress.size());
    branch_info->live_progress.push_back(0.0);
  }
  branch_info->live_progress[branch_internal_clock] += progress;

  if (new_global_clock != global_clock_) {
    CHECK_EQ(new_global_clock, global_clock_ + 1);
    global_clock_ = new_global_clock;

    double time = (tbb::tick_count::now() - global_start_tick_).seconds();
    CHECK_EQ(branch_info->timeline.size(), branch_internal_clock);
    if (branch_info->timeline.size() == 0) {
      branch_info->first_progress_time = time;
    } else {
      int next_resumed_clock = branch_internal_clock + 1;
      if (!branch_info->resumed_clocks.empty()) {
        next_resumed_clock = branch_info->resumed_clocks.front();
      }
      CHECK_GE(next_resumed_clock, branch_internal_clock);
      if (next_resumed_clock == branch_internal_clock) {
        /* This branch has been resumed */
        CHECK(!isnan(branch_info->time_per_clock));
        cout << "Branch " << branch_id
             << " resumed at internal clock " << branch_internal_clock << endl;
        branch_info->first_progress_time =
            time - branch_info->timeline.back() - branch_info->time_per_clock;
        branch_info->resumed_clocks.pop();
        cout << "branch_info->timeline.back() = "
             << branch_info->timeline.back() << endl;
        cout << "branch_info->time_per_clock = "
             << branch_info->time_per_clock << endl;
        cout << "first_progress_time = "
             << branch_info->first_progress_time << endl;
      }
    }
    branch_info->timeline.push_back(time - branch_info->first_progress_time);
    branch_info->progress.push_back(
        branch_info->live_progress[branch_internal_clock]);
    if (config_.mltuner_config.average_worker_progress) {
      branch_info->progress[branch_internal_clock] /= num_workers_;
    }
    if (branch_info->flag != TRAINING_BRANCH
        || (time - branch_info->last_print_time
            > config_.mltuner_config.print_interval)) {
      cout << "BRANCH SCHEDULER: branch " << branch_id
          << " at internal clock " << branch_info->internal_clock
          << " global clock " << global_clock_
          << " lineage clock " << branch_info->lineage_clock
          << " has progress " << branch_info->progress[branch_internal_clock]
          << " time = " << time
          << endl;
      branch_info->last_print_time = time;
    }
    branch_info->internal_clock++;
    branch_info->lineage_clock++;

    /* TODO: move this to the logic */
    if (branch_info->timeline.size() >= 2) {
      /* Calculate time per clock */
      branch_info->time_per_clock =
          (branch_info->timeline.back() - branch_info->timeline.front())
              / (branch_info->timeline.size() - 1);
      int estimated_num_clocks =
          floor(branch_info->scheduled_run_time / branch_info->time_per_clock);
      int max_trial_clocks =
          clocks_per_epoch(branch_info->tunable, config_);
      if (estimated_num_clocks < 0) {
        cout << "***WARNING: estimated_num_clocks overflow!\n";
        cout << "estimated_num_clocks = " << estimated_num_clocks << endl;
        cout << "branch_info->scheduled_run_time = " << branch_info->scheduled_run_time << endl;
        cout << "branch_info->time_per_clock = " << branch_info->time_per_clock << endl;
        estimated_num_clocks = max_trial_clocks;
      }
      if (estimated_num_clocks > max_trial_clocks) {
        estimated_num_clocks = max_trial_clocks;
      }
      if (estimated_num_clocks > branch_info->num_clocks_scheduled) {
        int more_clocks_to_schedule =
            estimated_num_clocks - branch_info->num_clocks_scheduled;
        cout << "Schedule " << more_clocks_to_schedule
             << " more clocks for branch " << branch_id << endl;
        schedule_branch(branch_id, more_clocks_to_schedule);
        send_branch_schedules();
        next_decision_clock_ += more_clocks_to_schedule;
      }
    }

    // if (global_clock_ + 1 != next_schedule_clock_) {
    /* Make branch decisions */
    logic_->make_branch_decisions();
    // }
  }
}

void MltunerImpl::schedule_run_time(
    int branch_id, double scheduled_run_time) {
  SchedulerBranchInfo *branch_info = branch_map_[branch_id];
  CHECK(branch_info);
  CHECK(branch_info->active);
  branch_info->scheduled_run_time = scheduled_run_time;
}

void MltunerImpl::schedule_branch(
    int branch_id, int num_clocks) {
  CHECK_GT(num_clocks, 0) << "schedule no clocks for branch " << branch_id;
  SchedulerBranchInfo *branch_info = branch_map_[branch_id];
  CHECK(branch_info);
  CHECK(branch_info->active);
  if (branch_info->num_clocks_scheduled != 0
      && next_schedule_clock_ != branch_info->next_global_clock) {
    /* This branch is scheduled to be resumed */
    cout << "going to resume branch " << branch_id
         << " for " << num_clocks << " clocks"
         << " at internal clock " << branch_info->num_clocks_scheduled << endl;
    int resumed_clock = branch_info->num_clocks_scheduled;
    if (!branch_info->resumed_clocks.empty()) {
      CHECK_GT(resumed_clock, branch_info->resumed_clocks.back());
    }
    branch_info->resumed_clocks.push(resumed_clock);
  }
  for (int i = 0; i < num_clocks; i++) {
    int clock = next_schedule_clock_++;
    int branch_id = branch_info->branch_id;
    int branch_internal_clock = branch_info->num_clocks_scheduled++;
    clock_schedules_.push_back(
        ClockSchedule(branch_id, branch_internal_clock));
    msg_payload_clocks_.push_back(clock);
    msg_payload_branch_ids_.push_back(branch_id);
  }
  branch_info->next_global_clock = next_schedule_clock_;
}

void MltunerImpl::send_branch_schedules() {
  CHECK_EQ(msg_payload_clocks_.size(), msg_payload_branch_ids_.size());
  if (!msg_payload_clocks_.size()) {
    return;
  }
  communicator_->schedule_branches(
      msg_payload_clocks_.size(),
      msg_payload_clocks_.data(), msg_payload_branch_ids_.data());
  msg_payload_clocks_.clear();
  msg_payload_branch_ids_.clear();
}

void MltunerImpl::interleave_branches(
    const vector<int>& branch_ids,
    uint num_clocks_per_branch, uint stride) {
  CHECK(stride);
  uint batch_size = num_clocks_per_branch * branch_ids.size();
  for (uint i = 0; i < batch_size; i++) {
    int clock = next_schedule_clock_++;
    uint idx = i % (branch_ids.size() * stride) / stride;
    int branch_id = branch_ids[idx];
    SchedulerBranchInfo *branch_info = branch_map_[branch_id];
    CHECK(branch_info);
    CHECK(branch_info->active);
    int branch_internal_clock = branch_info->num_clocks_scheduled++;
    clock_schedules_.push_back(
        ClockSchedule(branch_id, branch_internal_clock));
    msg_payload_clocks_.push_back(clock);
    msg_payload_branch_ids_.push_back(branch_id);
  }
}

void MltunerImpl::interleave_all_active_branches(
    uint num_clocks_per_branch, uint stride) {
  vector<int> branch_ids;
  for (SchedulerBranchInfoMap::iterator it = active_branch_map_.begin();
      it != active_branch_map_.end(); it++) {
    BranchInfo *branch_info = it->second;
    CHECK(branch_info);
    branch_ids.push_back(branch_info->branch_id);
  }
  interleave_branches(branch_ids, num_clocks_per_branch, stride);
}

int MltunerImpl::make_branch(
    int parent_branch_id, const Tunable& tunable, int flag) {
  int branch_id = next_branch_id_++;
  cout << "Make branch " << branch_id
       << " with flag " << flag
       << " from " << parent_branch_id
       << " with tunable: " << endl;
  tunable.print();
  SchedulerBranchInfo *branch_info = new SchedulerBranchInfo(
      branch_id, tunable, flag, parent_branch_id);
  branch_info->internal_clock = INITIAL_CLOCK;
  if (parent_branch_id != -1) {
    SchedulerBranchInfo *parent_branch_info = branch_map_[parent_branch_id];
    CHECK(parent_branch_info);
    CHECK_EQ(parent_branch_info->internal_clock,
        parent_branch_info->num_clocks_scheduled);
    branch_info->lineage_clock =
        branch_info->internal_clock + parent_branch_info->lineage_clock;
  } else {
    branch_info->lineage_clock = branch_info->internal_clock;
  }
  if (active_branch_map_.size() >= config_.num_branches) {
    cout << "active branches:" << endl;
    for (SchedulerBranchInfoMap::iterator it = active_branch_map_.begin();
        it != active_branch_map_.end(); it++) {
      cout << it->second->branch_id << endl;
    }
  }
  CHECK_LT(active_branch_map_.size(), config_.num_branches);
  active_branch_map_[branch_id] = branch_info;
  branch_map_[branch_id] = branch_info;
  // choice_branch_info_map[tunable_hash] = branch_info;
  int clock_to_happen = next_schedule_clock_;
  communicator_->make_branch(
      branch_info->branch_id, branch_info->tunable, branch_info->flag,
      branch_info->parent_branch_id, clock_to_happen);
  return branch_id;
}

int MltunerImpl::make_branch(
    int parent_branch_id, const TunableChoice& tunable_choice, int flag) {
  Tunable tunable = make_tunable(tunable_choice);
  int branch_id = make_branch(parent_branch_id, tunable, flag);
  SchedulerBranchInfo *branch_info = branch_map_[branch_id];
  CHECK(branch_info);
  branch_info->tunable_choice = tunable_choice;
  return branch_id;
}

void MltunerImpl::make_branches(
    int parent_branch_id, const vector<TunableChoice>& tunable_choices,
    int flag, vector<int>& branch_ids) {
  for (uint i = 0; i < tunable_choices.size(); i++) {
    branch_ids.push_back(make_branch(
        parent_branch_id, tunable_choices[i], flag));
  }
}

void MltunerImpl::inactivate_branch(int branch_id) {
  cout << "inactivate branch " << branch_id << endl;
  SchedulerBranchInfo *branch_info = branch_map_[branch_id];
  CHECK(branch_info);
  branch_info->active = false;
  active_branch_map_.erase(branch_id);
  int clock_to_happen = next_schedule_clock_;
  communicator_->inactivate_branch(branch_id, clock_to_happen);
}

Tunable MltunerImpl::make_tunable(
    const TunableChoice& tunable_choice) {
  Tunable tunable;
  const TunableSpecs& tunable_specs = config_.mltuner_config.tunable_specs;
  tunable.init_default(tunable_specs);
  for (int tunable_id = 0; tunable_id < tunable_specs.size(); tunable_id++) {
    const TunableSpec& tunable_spec = tunable_specs[tunable_id];
    float val = tunable_spec.default_val;
    if (tunable_spec.to_search) {
      TunableChoice::const_iterator it = tunable_choice.find(tunable_id);
      CHECK(it != tunable_choice.end());
      val = it->second;
    }
    tunable.set(tunable_id, val, tunable_specs);
  }
  return tunable;
}

void MltunerImpl::run_one_epoch(int branch_id) {
  cout << "run one epoch for branch " << branch_id << endl;
  SchedulerBranchInfo *training_branch_info = branch_map_[branch_id];
  CHECK(training_branch_info);
  const Tunable& training_tunable = training_branch_info->tunable;
  int num_clocks_per_test = clocks_per_test(training_tunable, config_);
  schedule_branch(branch_id, num_clocks_per_test);
}
