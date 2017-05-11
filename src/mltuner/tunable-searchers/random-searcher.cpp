/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>
#include <algorithm>    // std::random_shuffle

#include <tbb/tick_count.h>

#include "random-searcher.hpp"

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

RandomSearcher::RandomSearcher(
    const Config& config,
    const TunableIds& tunable_ids)
    : config_(config),
      tunable_ids_(tunable_ids),
      tunable_specs_(config.mltuner_config.tunable_specs) {
  init();
}

void RandomSearcher::init() {}

bool RandomSearcher::get_result(
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

void RandomSearcher::set_result(uint exp_id, double cost, double loss) {
  /* Do nothing */
}

bool RandomSearcher::make_tunable_exp() {
  tbb::tick_count tick_start = tbb::tick_count::now();
  TunableChoice tunable_choice;
  for (uint tidx = 0; tidx < tunable_ids_.size(); tidx++) {
    int tunable_id = tunable_ids_[tidx];
    const TunableSpec& tunable_spec = tunable_specs_[tunable_id];
    if (tunable_spec.type ==  "discrete") {
      int val_id = rand() % tunable_spec.valid_vals.size();
      tunable_choice[tunable_id] = val_id;
      continue;
    }
    if (tunable_spec.type == "continuous-linear"
        || tunable_spec.type == "continuous-log") {
      float min_val = tunable_spec.min_val;
      float max_val = tunable_spec.max_val;
      float range = max_val - min_val;
      float val = ((double)rand() / (RAND_MAX)) * range + min_val;
      tunable_choice[tunable_id] = val;
      continue;
    }
    CHECK(0);
  }
  tunable_exp_results_.push_back(TunableExpResult(tunable_choice));
  searcher_time_ =
      (tbb::tick_count::now() - tick_start).seconds();
  return true;
}
