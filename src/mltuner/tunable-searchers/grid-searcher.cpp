/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>
#include <algorithm>    // std::random_shuffle

#include <tbb/tick_count.h>

#include "grid-searcher.hpp"

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

GridSearcher::GridSearcher(
    const Config& config,
    const TunableIds& tunable_ids)
    : config_(config),
      tunable_ids_(tunable_ids),
      tunable_specs_(config.mltuner_config.tunable_specs) {
  init();
}

void GridSearcher::init() {
  int num_choices = 1;
  for (uint tidx = 0; tidx < tunable_ids_.size(); tidx++) {
    int tunable_id = tunable_ids_[tidx];
    CHECK_GE(tunable_id, 0);
    CHECK_LT(tunable_id, tunable_specs_.size());
    TunableChoiceValues& tunable_choice_values =
        multi_dim_tunbale_choice_values_[tunable_id];
    const TunableSpec& tunable_spec = tunable_specs_[tunable_id];
    if (tunable_spec.type == "discrete") {
      for (int val = 0; val < tunable_spec.valid_vals.size(); val++) {
        tunable_choice_values.push_back(val);
      }
      num_choices *= tunable_choice_values.size();
      continue;
    }
    if (tunable_spec.type == "continuous-linear"
        || tunable_spec.type == "continuous-log") {
      float min_val = tunable_spec.min_val;
      float max_val = tunable_spec.max_val;
      float range = max_val - min_val;
      float step = range / config_.mltuner_config.grid_size;
      for (int i = 0; i <= config_.mltuner_config.grid_size; i++) {
        float val = min_val + step * i;
        tunable_choice_values.push_back(val);
      }
      num_choices *= tunable_choice_values.size();
      continue;
    }
    CHECK(0);
  }

  vector<int> choice_ids;
  for (int i = 0; i < num_choices; i++) {
    choice_ids.push_back(i);
  }
  if (config_.mltuner_config.shuffle_grid) {
    std::random_shuffle(choice_ids.begin(), choice_ids.end());
  }
  for (int i = 0; i < choice_ids.size(); i++) {
    int choice_id = choice_ids[i];
    int residual = choice_id;
    TunableChoice tunable_choice;
    for (uint tidx = 0; tidx < tunable_ids_.size(); tidx++) {
      int tunable_id = tunable_ids_[tidx];
      CHECK(multi_dim_tunbale_choice_values_.count(tunable_id));
      TunableChoiceValues& tunable_choice_values =
          multi_dim_tunbale_choice_values_[tunable_id];
      int choice_idx = residual % tunable_choice_values.size();
      tunable_choice[tunable_id] = tunable_choice_values[choice_idx];
      residual /= tunable_choice_values.size();
    }
    CHECK_EQ(residual, 0);
    tunable_choices_.push_back(tunable_choice);
  }
  current_choice_idx_ = 0;
  searcher_time_ = 0.0;
}

bool GridSearcher::get_result(
    TunableChoice *tunable_choice_ptr, uint *exp_id_ptr) {
  bool success = make_tunable_exp();
  if (!success) {
    return false;
  }
  CHECK(tunable_exp_results_.size());
  TunableExpResult& tunable_exp_result = tunable_exp_results_.back();
  *tunable_choice_ptr = tunable_exp_result.tunable_choice;
  *exp_id_ptr = tunable_exp_results_.size() - 1;
  return true;
}

void GridSearcher::set_result(uint exp_id, double cost, double loss) {
  /* Do nothing */
}

bool GridSearcher::make_tunable_exp() {
  cout << "current_choice_idx_ = " << current_choice_idx_ << endl;
  if (current_choice_idx_ >= tunable_choices_.size()) {
    return false;
  }
  tbb::tick_count tick_start = tbb::tick_count::now();
  TunableChoice tunable_choice = tunable_choices_[current_choice_idx_];
  tunable_exp_results_.push_back(TunableExpResult(tunable_choice));
  current_choice_idx_++;
  CHECK_EQ(current_choice_idx_, tunable_exp_results_.size());
  searcher_time_ =
      (tbb::tick_count::now() - tick_start).seconds();
  return true;
}
