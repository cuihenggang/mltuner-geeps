#ifndef __tuner_logic_hpp__
#define __tuner_logic_hpp__

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

#include "mltuner-impl.hpp"

class TunerLogic {
 protected:
  MltunerImpl *impl_;
  const Config& config_;

  TunableIds tunable_ids_;
  int next_decision_clock_;

 public:
  TunerLogic(MltunerImpl *impl, const Config& config)
      : impl_(impl), config_(config) {
    const TunableSpecs& tunable_specs = config_.mltuner_config.tunable_specs;
    for (int tunable_id = 0; tunable_id < tunable_specs.size(); tunable_id++) {
      const TunableSpec& tunable_spec = tunable_specs[tunable_id];
      if (tunable_spec.to_search) {
        tunable_ids_.push_back(tunable_id);
      }
    }
  }

  virtual void make_branch_decisions() = 0;
  virtual void make_and_schedule_initial_branches() = 0;
};

#endif  // defined __tuner_logic_hpp__
