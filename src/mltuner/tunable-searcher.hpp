#ifndef __tunable_searcher_hpp__
#define __tunable_searcher_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include "mltuner-impl.hpp"


class TunableSearcher {
 public:
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual bool get_result(
      TunableChoice *tunable_choice_ptr, uint *exp_id_ptr) = 0;
  virtual void set_result(uint exp_id, double cost, double loss) = 0;
  virtual double get_searcher_time() = 0;
};

#endif  // defined __tunable_searcher_hpp__