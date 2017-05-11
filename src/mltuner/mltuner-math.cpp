/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>

#include "common/internal-config.hpp"
#include "common/common-utils.hpp"
#include "mltuner-impl.hpp"

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

double MltunerImpl::get_interception(
    const DoubleVec& b, double bt) {
  CHECK_GE(b.size(), 2);
  double b_front = b.front();
  double b_back = b.back();
  uint i;
  if (b_front > b_back) {
    if (b_front <= bt || b_back >= bt) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    for (i = 0; i < b.size(); i++) {
      if (b[i] < bt) {
        break;
      }
    }
  } else if (b_front < b_back) {
    if (b_front >= bt || b_back <= bt) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    for (i = 0; i < b.size(); i++) {
      if (b[i] > bt) {
        break;
      }
    }
  } else {
    return std::numeric_limits<double>::quiet_NaN();
  }
  CHECK_GE(i, 1);
  double a1 = static_cast<double>(i - 1);
  double a2 = static_cast<double>(i);
  double b1 = b[i - 1];
  double b2 = b[i];
  double at = a1 + (a2 - a1) / (b2 - b1) * (bt - b1);
  return at;
}

bool MltunerImpl::check_monotone(const DoubleVec& x) {
  CHECK_GE(x.size(), 2);
  if (x.front() <= x.back()) {
    for (int i = 1; i < x.size(); i++) {
      if (x[i - 1] > x[i]) {
        return false;
      }
    }
    return true;
  } else {
    for (int i = 1; i < x.size(); i++) {
      if (x[i - 1] < x[i]) {
        return false;
      }
    }
    return true;
  }
}

bool MltunerImpl::go_down_more_often_than_go_up(const DoubleVec& x) {
  CHECK_GE(x.size(), 4);
  int num_go_down = 0;
  int num_go_up = 0;
  for (int i = 1; i < x.size(); i++) {
    if (x[i] < x[i - 1]) {
      num_go_down++;
    } else {
      num_go_up++;
    }
  }
  return num_go_down > num_go_up;
}

double MltunerImpl::calc_max_go_up(const DoubleVec& x) {
  double max_go_up = 0.0;
  for (int i = 1; i < x.size(); i++) {
    if (x[i] - x[i - 1] > max_go_up) {
      max_go_up = x[i] - x[i - 1];
    }
  }
  return max_go_up;
}

bool MltunerImpl::go_up_less_than_threshold(
    const DoubleVec& x, double threshold) {
  double max_go_up = calc_max_go_up(x);
  return (max_go_up < threshold);
}

double MltunerImpl::calc_slope(
    const DoubleVec& x, const DoubleVec& y) {
  double slope = (y.back() - y.front()) / (x.back() - x.front());
  return slope;
}

void MltunerImpl::downsample(
    const DoubleVec& input, int rate, DoubleVec& output) {
  output.clear();
  double sum = 0.0;
  int count = 0;
  for (int i = 0; i < input.size(); i++) {
    sum += input[i];
    count++;
    if (count == rate) {
      output.push_back(sum / count);
      sum = 0.0;
      count = 0;
    }
  }
}

void MltunerImpl::downsample(
    const DoubleVec& input, int start, int end, int rate, DoubleVec& output) {
  output.clear();
  double sum = 0.0;
  int count = 0;
  for (int i = start; i < end; i++) {
    CHECK_GE(i, 0);
    CHECK_LT(i, input.size());
    sum += input[i];
    count++;
    if (count == rate) {
      output.push_back(sum / count);
      sum = 0.0;
      count = 0;
    }
  }
}

bool MltunerImpl::check_valid_progress(const DoubleVec& y) {
  if (y.back() >= y.front()) {
    return false;
  }
  double front_back_diff = y.front() - y.back();
  double threshold = front_back_diff * GO_UP_THRESHOLD;
  if (!go_up_less_than_threshold(y, threshold)) {
    return false;
  }
  return true;
}

double MltunerImpl::calc_slope_with_check(
    const DoubleVec& x, const DoubleVec& y) {
  if (!check_valid_progress(y)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return calc_slope(x, y);
}

pair<ConvergenceState, double> MltunerImpl::calc_slope_with_penalty(
    const DoubleVec& x, const DoubleVec& y, double threshold) {
  if (y.back() > y.front() * DIVERGENCE_THRESHOLD) {
    ConvergenceState convergence_state = DIVERGED;
    double slope = std::numeric_limits<double>::quiet_NaN();
    return pair<ConvergenceState, double>(convergence_state, slope);
  }
  double y_range = y.back() - y.front();
  double x_range = x.back() - x.front();
  double max_y_go_up = calc_max_go_up(y);
  CHECK_GE(max_y_go_up, 0.0);
  cout << "max_y_go_up = " << max_y_go_up << endl;
  cout << "y_range = " << y_range << endl;
  cout << "threshold = " << threshold << endl;
  double y_range_with_penalty = y_range + max_y_go_up;
  double slope = y_range / x_range;
  double slope_with_penalty = y_range_with_penalty / x_range;
  cout << "slope = " << slope << endl;
  cout << "slope_with_penalty = " << slope_with_penalty << endl;
  if (slope > 0.0) {
    slope = std::numeric_limits<double>::quiet_NaN();
  }
  if (slope_with_penalty > 0.0) {
    slope_with_penalty = std::numeric_limits<double>::quiet_NaN();
  }
  ConvergenceState convergence_state;
  if (max_y_go_up > fabs(y_range) * threshold) {
    convergence_state = UNSTABLE;
  } else {
    convergence_state = CONVERGING;
  }
  return pair<ConvergenceState, double>(convergence_state, slope_with_penalty);
}

pair<ConvergenceState, double> MltunerImpl::summarize_progress(
    double starting_time_offset, double starting_progress, int branch_id,
    double threshold) {
  SchedulerBranchInfo *branch_info = branch_map_[branch_id];
  CHECK(branch_info);
  cout << "Summarize progress for branch " << branch_id
       << " with tunable:\n";
  branch_info->tunable.print();
  CHECK_EQ(branch_info->timeline.size(), branch_info->progress.size());
  int num_samples = branch_info->timeline.size();
  int num_ds_samples = config_.mltuner_config.num_progress_samples;
  int rate = num_samples / num_ds_samples;
  cout << "time_per_clock = " << branch_info->time_per_clock << endl;
  cout << "num_clocks = " << num_samples << endl;
  cout << "rate = " << rate << endl;
  if (rate > 1) {
    downsample(branch_info->timeline, rate, branch_info->timeline_ds);
    downsample(branch_info->progress, rate, branch_info->progress_ds);
  } else {
    branch_info->timeline_ds = branch_info->timeline;
    branch_info->progress_ds = branch_info->progress;
  }
  /* Replace the first entry of the progress vector
   * with the starting progress */
  CHECK(!isnan(starting_progress));
  branch_info->timeline_adjusted_start = branch_info->timeline_ds;
  double starting_time = branch_info->timeline[0] + starting_time_offset;
  branch_info->timeline_adjusted_start.insert(
      branch_info->timeline_adjusted_start.begin(), starting_time);
  branch_info->progress_adjusted_start = branch_info->progress_ds;
  branch_info->progress_adjusted_start.insert(
      branch_info->progress_adjusted_start.begin(), starting_progress);
  if (branch_info->flag != TESTING_BRANCH
      && config_.mltuner_config.progress_in_logscale) {
    for (uint i = 0; i < branch_info->progress_adjusted_start.size(); i++) {
      branch_info->progress_adjusted_start[i] =
          log10(branch_info->progress[i]);
    }
  }
  cout << "timeline_adjusted_start: " << endl;
  for (uint i = 0; i < branch_info->timeline_adjusted_start.size(); i++) {
    cout << branch_info->timeline_adjusted_start[i] << endl;
  }
  cout << "time span = " <<
        (branch_info->timeline_adjusted_start.back()
            - branch_info->timeline_adjusted_start.front())
      << endl;
  cout << "progress_adjusted_start: " << endl;
  for (uint i = 0; i < branch_info->progress_adjusted_start.size(); i++) {
    cout << branch_info->progress_adjusted_start[i] << endl;
  }
  pair<ConvergenceState, double> ret =
      calc_slope_with_penalty(
          branch_info->timeline_adjusted_start,
          branch_info->progress_adjusted_start,
          threshold);
  branch_info->convergence_state = ret.first;
  branch_info->progress_slope = ret.second;
  return ret;
}

double MltunerImpl::calc_time_saving(
    int new_branch_id, int old_branch_id) {
  SchedulerBranchInfo *new_branch_info = branch_map_[new_branch_id];
  CHECK(new_branch_info);
  SchedulerBranchInfo *old_branch_info = branch_map_[old_branch_id];
  CHECK(old_branch_info);
  double saving_fraction;
  if (!SAVING_BY_INTERSECT
      || old_branch_info->progress_adjusted_start.back()
          < new_branch_info->progress_adjusted_start.back()) {
    cout << "WARNING: fall back to the fraction of slopes\n";
    saving_fraction =
        new_branch_info->progress_slope / old_branch_info->progress_slope - 1.0;
    CHECK_GE(saving_fraction, 0.0);
    return saving_fraction;
  } else {
    saving_fraction = calc_time_saving(
        new_branch_info->timeline_adjusted_start,
        new_branch_info->progress_adjusted_start,
        old_branch_info->timeline_adjusted_start,
        old_branch_info->progress_adjusted_start);
    if (saving_fraction < 0.0) {
      cout << "WARNING: fall back to the fraction of slopes\n";
      saving_fraction =
          new_branch_info->progress_slope / old_branch_info->progress_slope - 1.0;
    }
  }
  CHECK_GE(saving_fraction, 0.0);
  return saving_fraction;
}

double MltunerImpl::calc_time_saving(
    const DoubleVec& new_timeline, const DoubleVec& new_progress,
    const DoubleVec& old_timeline, const DoubleVec& old_progress) {
  CHECK_GE(old_progress.back(), new_progress.back());
  double last_old_progress = old_progress.back();
  double old_time_cost = old_timeline.back() - old_timeline.front();
  /* Calculate the time for the new branch to reach last_old_progress */
  double new_time_idx =
      get_interception(new_progress, last_old_progress);
  int new_time_int_idx = floor(new_time_idx);
  CHECK_GE(new_time_int_idx, 0);
  CHECK_LT(new_time_int_idx, new_timeline.size());
  double new_finish_time;
  double diff = new_time_idx - new_time_int_idx;
  if (diff > 0) {
    CHECK_LT(new_time_int_idx + 1, new_timeline.size());
    new_finish_time = new_timeline[new_time_int_idx]
        + (new_timeline[new_time_int_idx + 1] - new_timeline[new_time_int_idx])
            * diff;
  } else {
    new_finish_time = new_timeline[new_time_int_idx];
  }
  double new_time_cost = new_finish_time - new_timeline.front();
  double saving_fraction = old_time_cost / new_time_cost - 1.0;
  return saving_fraction;
}

double MltunerImpl::average_progress(int branch_id) {
  SchedulerBranchInfo *branch_info = branch_map_[branch_id];
  CHECK(branch_info);
  double sum = 0.0;
  for (uint i = 0; i < branch_info->progress.size(); i++) {
    sum += branch_info->progress[i];
  }
  double average = sum / branch_info->progress.size();
  return average;
}

void MltunerImpl::summarize_runned_branch(
    int branch_id, double *starting_time_offset, double *starting_progress,
    double *time_to_try) {
  SchedulerBranchInfo *branch_info = branch_map_[branch_id];
  CHECK(branch_info);
  int rate = 1;
  int num_samples = config_.mltuner_config.num_progress_samples;
  while (true) {
    int start = branch_info->progress.size() - rate * num_samples;
    CHECK_GE(start, 0);
    int end = branch_info->progress.size();
    downsample(branch_info->progress, start, end, rate,
        branch_info->progress_ds);
    if (check_valid_progress(branch_info->progress_ds)) {
      downsample(branch_info->timeline, start, end, rate,
          branch_info->timeline_ds);
      *starting_time_offset =
          branch_info->timeline_ds.back() - branch_info->timeline.back();
      *starting_progress = branch_info->progress_ds.back();
      /* We do "num_samples + 1" for robustness */
      *time_to_try = rate * (num_samples + 1) * branch_info->time_per_clock;
      return;
    }
    rate++;
  }
}
