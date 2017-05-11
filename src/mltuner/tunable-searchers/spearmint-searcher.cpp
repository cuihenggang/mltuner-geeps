/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>
#include <fstream>

#include <boost/format.hpp>

#include <tbb/tick_count.h>

#include "spearmint-searcher.hpp"

#define SEARCHING_STOP_TOPK       5
#define SEARCHING_STOP_THRESHOLD  0.1
// #define SEARCHING_STOP_LASTK      10

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::cerr;
using std::cout;
using std::endl;

const int CONSTRAINED_CHOOSER = 0;
// const int CONSTRAINED_CHOOSER = 1;

SpearmintSearcher::SpearmintSearcher(
    const Config& config,
    const string& app_name, const string& identity,
    const TunableIds& tunable_ids,
    bool background_mode, bool rerun_duplicate)
    : config_(config),
      searcher_root_(config_.mltuner_config.spearmint_root),
      app_name_(app_name),
      identity_(identity),
      tunable_ids_(tunable_ids),
      background_mode_(background_mode),
      rerun_duplicate_(rerun_duplicate),
      stop_signal_(0) {
  init();
}

SpearmintSearcher::~SpearmintSearcher() {
  stop();
}

void SpearmintSearcher::init() {
  string exp_dir;
  if (config_.output_dir != "") {
    exp_dir = (boost::format("%s/%s-%s")
        % config_.output_dir % app_name_ % identity_).str();
  } else {
    exp_dir = (boost::format("%s/myexp/%s-%s")
        % searcher_root_ % app_name_ % identity_).str();
  }
  string clear_exp_dir_cmd =
      (boost::format("rm -r %s")
          % exp_dir).str();
  if (system(clear_exp_dir_cmd.c_str())) {
    cerr << "Non-zero syscall return value\n";
  }
  string copy_exp_config_cmd =
      (boost::format("cp -r %s/myexp/%s %s")
          % searcher_root_ % app_name_ % exp_dir).str();
  if (system(copy_exp_config_cmd.c_str())) {
    cerr << "Non-zero syscall return value\n";
  }
  string python_cmd = "python";
  searcher_cmd_ =
      (boost::format("python %s/spearmint/mymain.py %s")
          % searcher_root_ % exp_dir).str();
  cout << "searcher_cmd_ = " << searcher_cmd_ << endl;
  result_file_ =
      (boost::format("%s/results.dat")
          % exp_dir).str();

  searcher_time_ = 0.0;
}

bool SpearmintSearcher::get_result(
    TunableChoice *tunable_choice_ptr, uint *exp_id_ptr) {
  /* Check whether we have already converged */
  bool search_converged = check_convergence();
  if (search_converged) {
    return false;
  }

  /* Get the next tunable choice to try */
  ScopedLock lock(mutex_);
  if (background_mode_) {
    while (!pending_exp_set_.size()) {
      cout << "pending_exp_set_ empty!\n";
      cvar_.wait(lock);
    }
  } else {
    call_searcher();
    uint last_exp_id = tunable_exp_results_.size() - 1;
    pending_exp_set_.insert(last_exp_id);
  }
  CHECK(pending_exp_set_.size());
  *exp_id_ptr = *pending_exp_set_.begin();
  CHECK_LT(*exp_id_ptr, tunable_exp_results_.size());
  const TunableExpResult& tunable_exp_result =
      tunable_exp_results_[*exp_id_ptr];
  CHECK(!tunable_exp_result.exp_done) << " exp_id = " << *exp_id_ptr;
  *tunable_choice_ptr = tunable_exp_result.tunable_choice;
  return true;
}

void SpearmintSearcher::set_result(uint exp_id, double cost, double loss) {
  ScopedLock lock(mutex_);
  CHECK_LT(exp_id, tunable_exp_results_.size());
  TunableExpResult& tunable_exp_result = tunable_exp_results_[exp_id];
  tunable_exp_result.cost = cost;
  tunable_exp_result.loss = loss;
  tunable_exp_result.exp_done = true;
  /* This exp_id might not be in the pending_exp_set_,
   * because sometimes we will update the result of a finished experiment. */
  if (pending_exp_set_.find(exp_id) != pending_exp_set_.end()) {
    pending_exp_set_.erase(exp_id);
  }
  cvar_.notify_all();
}

void SpearmintSearcher::start() {
  /* Start searcher thread */
  CHECK(background_mode_);
  background_thread_ = make_shared<boost::thread>(
      boost::bind(&SpearmintSearcher::background_thread_entry, this));
}

void SpearmintSearcher::stop() {
  ScopedLock lock(mutex_);
  CHECK(background_mode_);
  stop_signal_ = 1;
  cvar_.notify_all();
  lock.unlock();
  background_thread_->join();
}

void SpearmintSearcher::operator()() {
  background_thread_entry();
}

void SpearmintSearcher::background_thread_entry() {
  ScopedLock lock(mutex_);
  while (true) {
    /* We allow at most two pending experiments */
    while (pending_exp_set_.size() >= MAX_PENDING_SPEARMINT_EXPS
        && !stop_signal_) {
      cout << "pending_exp_set_.size() = " << pending_exp_set_.size() << endl;
      cvar_.wait(lock);
    }
    if (stop_signal_) {
      write_result_file();
      return;
    }
    lock.unlock();
    tbb::tick_count tick_start = tbb::tick_count::now();
    while (true) {
      call_searcher();
      if (rerun_duplicate_) {
        uint last_exp_id = tunable_exp_results_.size() - 1;
        bool duplicate = check_duplicate(last_exp_id);
        if (!duplicate) {
          break;
        }
      } else {
        break;
      }
    }
    searcher_time_ =
        (tbb::tick_count::now() - tick_start).seconds();
    cout << "call_searcher_time = " << searcher_time_ << endl;
    lock.lock();
    uint last_exp_id = tunable_exp_results_.size() - 1;
    pending_exp_set_.insert(last_exp_id);
    cvar_.notify_all();
  }
}

void SpearmintSearcher::call_searcher() {
  /* Since the main thread will not add or remove tunable_exp_results_ entries,
   * it's safe to do that without lock */
  write_result_file();
  if (system(searcher_cmd_.c_str())) {
    CHECK(0);
  }
  read_result_file();
}

void SpearmintSearcher::write_result_file() {
  if (!tunable_exp_results_.size()) {
    return;
  }
  ofstream fout(result_file_.c_str());
  CHECK(fout) << " result_file = " << result_file_;
  for (uint i = 0; i < tunable_exp_results_.size(); i++) {
    TunableExpResult& tunable_exp_result = tunable_exp_results_[i];
    if (tunable_exp_result.exp_done) {
      if (!isnan(tunable_exp_result.loss)) {
        fout << tunable_exp_result.loss;
      } else {
        if (CONSTRAINED_CHOOSER) {
          fout << "inf";
        } else {
          fout << "0";
        }
      }
      fout << ' ' << tunable_exp_result.cost;
    } else {
      fout << "P P";
    }
    TunableChoice& tunable_choice = tunable_exp_result.tunable_choice;
    for (uint tidx = 0; tidx < tunable_ids_.size(); tidx++) {
      int tunable_id = tunable_ids_[tidx];
      float val = tunable_choice[tunable_id];
      fout << ' ' << val;
    }
    fout << endl;
  }
}

void SpearmintSearcher::read_result_file() {
  ifstream fin(result_file_.c_str());
  CHECK(fin) << " result_file = " << result_file_;
  string line;
  for (uint i = 0; i < tunable_exp_results_.size(); i++) {
    CHECK(!fin.eof());
    std::getline(fin, line);
  }
  /* Only the last line should be different */
  CHECK(!fin.eof());
  char c1;
  char c2;
  fin >> c1 >> c2;
  CHECK_EQ(c1, 'P')
      << " result_file = " << result_file_
      << " tunable_exp_results_.size() = " << tunable_exp_results_.size();
  CHECK_EQ(c2, 'P')
      << " result_file = " << result_file_
      << " tunable_exp_results_.size() = " << tunable_exp_results_.size();
  TunableChoice tunable_choice;
  for (uint tidx = 0; tidx < tunable_ids_.size(); tidx++) {
    int tunable_id = tunable_ids_[tidx];
    CHECK(!fin.eof());
    float val;
    fin >> val;
    tunable_choice[tunable_id] = val;
  }
  tunable_exp_results_.push_back(TunableExpResult(tunable_choice));
}

bool SpearmintSearcher::check_duplicate(uint exp_id) {
  CHECK_LT(exp_id, tunable_exp_results_.size());
  const TunableChoice& tunable_choice =
      tunable_exp_results_[exp_id].tunable_choice;
  for (uint i = 0; i < tunable_exp_results_.size(); i++) {
    if (!tunable_exp_results_[i].exp_done) {
      continue;
    }
    TunableChoice& prev_tunable_choice = tunable_exp_results_[i].tunable_choice;
    bool duplicate = true;
    for (TunableChoice::const_iterator it = tunable_choice.begin();
        it != tunable_choice.end(); it++) {
      int tunable_id = it->first;
      if (it->second != prev_tunable_choice[tunable_id]) {
        duplicate = false;
        break;
      }
    }
    if (duplicate) {
      tunable_exp_results_[exp_id].loss = tunable_exp_results_[i].loss;
      tunable_exp_results_[exp_id].cost = tunable_exp_results_[i].cost;
      tunable_exp_results_[exp_id].exp_done = true;
      tunable_exp_results_[exp_id].duplicate = true;
      return true;
    }
  }
  return false;
}

bool SpearmintSearcher::check_convergence() {
  DoubleVec perfs;
  for (uint i = 0; i < tunable_exp_results_.size(); i++) {
    if (!tunable_exp_results_[i].duplicate
        && !isnan(tunable_exp_results_[i].loss)) {
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

bool SpearmintSearcher::is_topk_close_enough(
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

bool SpearmintSearcher::is_no_improvement_over_lastk(
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
