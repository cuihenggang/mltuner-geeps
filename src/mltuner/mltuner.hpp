#ifndef __mltuner_hpp__
#define __mltuner_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include <glog/logging.h>

#include "geeps.hpp"

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;

using boost::shared_ptr;

class Communicator {
 public:
  virtual void make_branch(
      int branch_id, const Tunable& tunable, int flag,
      int parent_branch_id, int clock_to_happen) = 0;
  virtual void inactivate_branch(
      int branch_id, int clock_to_happen) = 0;
  virtual void schedule_branches(
      uint batch_size, const int *clocks, const int *branch_ids) = 0;
};

class Mltuner {
 public:
  Mltuner(
      shared_ptr<Communicator> communicator,
      const Config& config);
  void recv_worker_started(uint worker_id);
  void recv_worker_progress(
      uint worker_id, int clock, int branch_id, double progress);
};

#endif  // defined __mltuner_hpp__
