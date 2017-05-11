/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include <boost/format.hpp>

#include <tbb/tick_count.h>

#include "marginal-hyperopt-searcher.hpp"

#define SEARCHING_STOP_TOPK       5
#define SEARCHING_STOP_THRESHOLD  0.1
// #define SEARCHING_STOP_LASTK      10

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::cerr;
using std::cout;
using std::endl;

MarginalHyperoptSearcher::MarginalHyperoptSearcher(
    const Config& config,
    const string& app_name, const string& identity,
    const TunableIds& tunable_ids,
    const TunableChoice& base_tunable_choice)
    : config_(config),
      searcher_root_(config_.mltuner_config.hyperopt_root),
      app_name_(app_name),
      identity_(identity),
      tunable_ids_(tunable_ids),
      base_tunable_choice_(base_tunable_choice) {
  init();
}

MarginalHyperoptSearcher::~MarginalHyperoptSearcher() {
  stop();
}

void MarginalHyperoptSearcher::init() {
  string exp_dir;
  if (config_.output_dir != "") {
    exp_dir = (boost::format("%s")
        % config_.output_dir).str();
  } else {
    exp_dir = (boost::format("%s")
        % searcher_root_).str();
  }

  result_file_ =
      (boost::format("%s/%s.dat")
          % exp_dir % identity_).str();

  searcher_time_ = 0.0;
}

bool MarginalHyperoptSearcher::get_result(
    TunableChoice *tunable_choice_ptr, uint *exp_id_ptr) {
  /* Check whether we have already converged */
  bool search_converged = check_convergence();
  if (search_converged) {
    return false;
  }

  /* Get the next tunable choice to try */
  ScopedLock lock(mutex_);
  tbb::tick_count tick_start = tbb::tick_count::now();
  call_searcher();
  searcher_time_ = (tbb::tick_count::now() - tick_start).seconds();
  cout << "call_searcher_time = " << searcher_time_ << endl;
  uint last_exp_id = tunable_exp_results_.size() - 1;
  pending_exp_set_.insert(last_exp_id);
  CHECK(pending_exp_set_.size());
  *exp_id_ptr = *pending_exp_set_.begin();
  CHECK_LT(*exp_id_ptr, tunable_exp_results_.size());
  const TunableExpResult& tunable_exp_result =
      tunable_exp_results_[*exp_id_ptr];
  CHECK(!tunable_exp_result.exp_done) << " exp_id = " << *exp_id_ptr;
  *tunable_choice_ptr = tunable_exp_result.tunable_choice;
  return true;
}

void MarginalHyperoptSearcher::set_result(uint exp_id, double cost, double loss) {
  ScopedLock lock(mutex_);
  CHECK_LT(exp_id, tunable_exp_results_.size());
  TunableExpResult& tunable_exp_result = tunable_exp_results_[exp_id];
  tunable_exp_result.loss = loss;
  tunable_exp_result.exp_done = true;
  /* This exp_id might not be in the pending_exp_set_,
   * because sometimes we will update the result of a finished experiment. */
  if (pending_exp_set_.find(exp_id) != pending_exp_set_.end()) {
    pending_exp_set_.erase(exp_id);
  }
  cvar_.notify_all();
}

void MarginalHyperoptSearcher::start() {
  /* Start searcher process */
  background_thread_ = make_shared<boost::thread>(
      boost::bind(&MarginalHyperoptSearcher::background_thread_entry, this));
}

void MarginalHyperoptSearcher::stop() {
  if (system("pkill -9 python")) {
    cerr << "Non-zero syscall return value\n";
  }
  background_thread_->join();
}

void MarginalHyperoptSearcher::operator()() {
  background_thread_entry();
}

void MarginalHyperoptSearcher::background_thread_entry() {
  searcher_cmd_ =
      (boost::format("python %s/call_hyperopt_retune_%s.py %s")
          % searcher_root_ % app_name_ % result_file_).str();
  cout << "searcher_cmd_ = " << searcher_cmd_ << endl;
  if (system(searcher_cmd_.c_str())) {
    cerr << "Non-zero syscall return value\n";
  }
}

void MarginalHyperoptSearcher::call_searcher() {
  /* Since the main thread will not add or remove tunable_exp_results_ entries,
   * it's safe to do that without lock */
  write_result_file();
  read_result_file();
}

void MarginalHyperoptSearcher::write_result_file() {
  if (!tunable_exp_results_.size()) {
    return;
  }
  ofstream fout(result_file_.c_str());
  CHECK(fout) << " result_file = " << result_file_;
  for (uint i = 0; i < tunable_exp_results_.size(); i++) {
    TunableExpResult& tunable_exp_result = tunable_exp_results_[i];
    fout << tunable_exp_result.tunable_id << ' '
         << tunable_exp_result.val << ' ';
    if (tunable_exp_result.exp_done) {
      if (!isnan(tunable_exp_result.loss)) {
        fout << ' ' << tunable_exp_result.loss;
      } else {
        fout << "0";
      }
    } else {
      fout << "P";
    }
    fout << endl;
  }
}

void MarginalHyperoptSearcher::read_result_file() {
  while (true) {
    ifstream fin(result_file_.c_str());
    if (!fin) {
      cout << "Missing result file " << result_file_ << endl;
      sleep(1);
      continue;
    }
    CHECK(fin) << " result_file = " << result_file_;
    string line;
    while (!fin.eof()) {
      std::getline(fin, line);
      if (line.size() && line[line.size() - 1] == 'P') {
        /* Got a proposal line */
        cout << "got " << line << endl;
        stringstream sin(line);
        TunableChoice tunable_choice = base_tunable_choice_;
        CHECK(!sin.eof());
        float tunable_id_float;
        sin >> tunable_id_float;
        int tunable_id = round(tunable_id_float);
        CHECK(!sin.eof());
        float val;
        sin >> val;
        tunable_choice[tunable_id] = val;
        cout << "tunable " << tunable_id << " set to " << val << endl;
        tunable_exp_results_.push_back(
            TunableExpResult(tunable_choice, tunable_id, val));
        return;
      }
    }
    cout << "Proposal not ready yet" << endl;
    sleep(1);
  }
}

bool MarginalHyperoptSearcher::check_convergence() {
  DoubleVec perfs;
  for (uint i = 0; i < tunable_exp_results_.size(); i++) {
    if (!isnan(tunable_exp_results_[i].loss)) {
      perfs.push_back(tunable_exp_results_[i].loss);
    }
  }

  /* We stop the searching if the top SEARCHING_STOP_TOPK best progress
   * has a difference less than SEARCHING_STOP_THRESHOLD. */
  bool search_converged = is_topk_close_enough(
      perfs, SEARCHING_STOP_TOPK, SEARCHING_STOP_THRESHOLD);
  if (search_converged) {
    return true;
  }

  // /* We also stop the searching if we haven't seen a better progress
   // * in the last SEARCHING_STOP_LASTK trials */
  // search_converged = is_no_improvement_over_lastk(
      // perfs, SEARCHING_STOP_LASTK);
  // if (search_converged) {
    // return true;
  // }
  return false;
}

bool MarginalHyperoptSearcher::is_topk_close_enough(
    const DoubleVec& x, int k, double threshold) {
  DoubleVec x_filtered;
  for (uint i = 0; i < x.size(); i++) {
    if (!isnan(x[i])) {
      x_filtered.push_back(x[i]);
    }
  }
  std::sort(x_filtered.begin(), x_filtered.end());
  cout << "Sorted search results:\n";
  for (uint i = 0; i < x_filtered.size(); i++) {
    cout << x_filtered[i] << endl;
  }
  CHECK_GT(k, 0);
  if (x_filtered.size() < k) {
    return false;
  }
  double x_min = x_filtered[0];
  double x_topk = x_filtered[k - 1];
  CHECK_LE(x_min, x_topk);
  /* Both x_min and x_topk should be negative numbers */
  CHECK_LE(x_min, 0.0);
  CHECK_LE(x_topk, 0.0);
  double diff_ratio = 1 - x_topk / x_min;
  return diff_ratio < threshold;
}

bool MarginalHyperoptSearcher::is_no_improvement_over_lastk(
    const DoubleVec& x, int k) {
  DoubleVec x_filtered;
  for (uint i = 0; i < x.size(); i++) {
    if (!isnan(x[i])) {
      x_filtered.push_back(x[i]);
    }
  }
  CHECK_GT(k, 0);
  if (x_filtered.size() < k + 1) {
    return false;
  }
  double best = 0.0;
  for (int i = x_filtered.size() - 1; i >= 0; i--) {
    if (x_filtered[i] < best) {
      best = x_filtered[i];
      if (i < x_filtered.size() - k) {
        return true;
      }
    }
  }
  return false;
}
