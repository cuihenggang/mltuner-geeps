#ifndef __grid_searcher_hpp__
#define __grid_searcher_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include "mltuner/tunable-searcher.hpp"


class GridSearcher : public TunableSearcher {
  typedef vector<float> TunableChoiceValues;
  typedef std::map<int, TunableChoiceValues> MultiDimTunableChoiceValues;
  struct TunableExpResult {
    TunableChoice tunable_choice;
    TunableExpResult() {}
    TunableExpResult(const TunableChoice& tunable_choice)
        : tunable_choice(tunable_choice) {}
  };
  typedef vector<TunableExpResult> TunableExpResults;

 public:
  GridSearcher(
      const Config& config,
      const TunableIds& tunable_ids);
  virtual void start() {}
  virtual void stop() {}
  virtual bool get_result(TunableChoice *tunable_choice_ptr, uint *exp_id_ptr);
  virtual void set_result(uint exp_id, double cost, double loss);
  virtual double get_searcher_time() {
    return searcher_time_;
  }

 private:
  const Config& config_;
  TunableIds tunable_ids_;
  TunableSpecs tunable_specs_;
  MultiDimTunableChoiceValues multi_dim_tunbale_choice_values_;
  vector<TunableChoice> tunable_choices_;
  int current_choice_idx_;
  TunableExpResults tunable_exp_results_;
  double searcher_time_;

  void init();
  bool make_tunable_exp();
};

#endif