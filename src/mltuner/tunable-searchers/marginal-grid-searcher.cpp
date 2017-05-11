/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>

#include <tbb/tick_count.h>

#include "marginal-grid-searcher.hpp"

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

MarginalGridSearcher::MarginalGridSearcher(
    const Config& config,
    const TunableIds& tunable_ids,
    const TunableChoice& base_tunable_choice)
    : config_(config),
      tunable_ids_(tunable_ids),
      tunable_specs_(config.mltuner_config.tunable_specs),
      base_tunable_choice_(base_tunable_choice) {
  init();
}

void MarginalGridSearcher::init() {
  tunable_choices_.push_back(base_tunable_choice_);
  for (uint tidx = 0; tidx < tunable_ids_.size(); tidx++) {
    int tunable_id = tunable_ids_[tidx];
    CHECK_GE(tunable_id, 0);
    CHECK_LT(tunable_id, tunable_specs_.size());
    const TunableSpec& tunable_spec = tunable_specs_[tunable_id];
    TunableChoiceValues tunable_choice_values;
    if (tunable_spec.type ==  "discrete") {
      for (int val = 0; val < tunable_spec.valid_vals.size(); val++) {
        if (val != base_tunable_choice_[tunable_id]) {
          TunableChoice tunable_choice = base_tunable_choice_;
          tunable_choice[tunable_id] = val;
          tunable_choices_.push_back(tunable_choice);
        }
      }
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
        if (val != base_tunable_choice_[tunable_id]) {
          TunableChoice tunable_choice = base_tunable_choice_;
          tunable_choice[tunable_id] = val;
          tunable_choices_.push_back(tunable_choice);
        }
      }
      continue;
    }
    CHECK(0);
  }
  current_choice_idx_ = 0;
  searcher_time_ = 0.0;
}

bool MarginalGridSearcher::get_result(
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

void MarginalGridSearcher::set_result(
    uint exp_id, double cost, double loss) {
  /* Do nothing */
}

bool MarginalGridSearcher::make_tunable_exp() {
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
