#ifndef __lazytable_hpp__
#define __lazytable_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "lazytable-user-defined-types.hpp"

using std::string;
using std::vector;

struct TunableSpec {
  string name;
  string type;
  bool to_search;
  float min_val;
  float max_val;
  float default_val;
  vector<float> valid_vals;
  TunableSpec() : to_search(true) {}
};
typedef vector<TunableSpec> TunableSpecs;

struct GeePsConfig;

struct Tunable {
  float lr;
  float slack;
  float batch_size;
  float momentum;
  float lr_decay;
  Tunable() {}
  Tunable(const Tunable& other) :
      lr(other.lr),
      slack(other.slack),
      batch_size(other.batch_size),
      momentum(other.momentum),
      lr_decay(other.lr_decay) {}
  void init_default(const TunableSpecs& specs) {
    for (int idx = 0; idx < specs.size(); idx++) {
      set(idx, specs[idx].default_val, specs);
    }
  }
  void set(uint idx, float val, const TunableSpecs& specs) {
    CHECK_LT(idx, specs.size());
    const TunableSpec& spec = specs[idx];
    float tunable_val;
    if (spec.type == "continuous-log") {
      tunable_val = pow(10, val);
    } else if (spec.type == "continuous-linear") {
      tunable_val = val;
    } else if (spec.type == "discrete") {
      uint val_idx = round(val);
      CHECK_LT(val_idx, spec.valid_vals.size()) << "idx = " << idx;
      tunable_val = spec.valid_vals[val_idx];
    } else {
      CHECK(0) << "Unknown tunable type " << spec.type;
    }
    const string& tunable_name = spec.name;
    if (tunable_name == "lr") {
      lr = tunable_val;
    } else if (tunable_name == "slack") {
      slack = tunable_val;
    } else if (tunable_name == "batch_size") {
      batch_size = tunable_val;
    } else if (tunable_name == "momentum") {
      momentum = tunable_val;
    } else if (tunable_name == "lr_decay") {
      lr_decay = tunable_val;
    } else {
      // CHECK(0);
    }
  }
  void print() const {
    std::cout
        << "lr = " << lr
        << ", slack = " << slack
        << ", batch_size = " << batch_size
        << ", momentum = " << momentum
        << ", lr_decay = " << lr_decay
        << std:: endl;
  }
};

struct MltunerConfig {
  string tuner_logic;
  string tunable_searcher;
  string tunable_searcher_for_retune;
  int initial_tuning;
  int retune_tunable;
  int force_retune;
  int stop_on_converge;
  string converge_type;
  int plateau_size_for_convergence;
  int plateau_size_for_retune;
  uint num_clocks_to_eval;
  int num_samples_per_epoch;
  int num_epoches_per_test;
  uint num_clocks_to_test;
  int num_progress_samples;
  int average_worker_progress;
  int progress_in_logscale;
  double time_to_try;
  string app_name;
  string spearmint_root;
  string hyperopt_root;
  int searching_stop_slack;
  double searching_stop_threshold;
  int grid_size;
  int shuffle_grid;
  double print_interval;

  TunableSpecs tunable_specs;
  Tunable tunable_for_train;
  Tunable tunable_for_eval;
  Tunable tunable_for_test;

  MltunerConfig() :
      tuner_logic("mltuner"),
      tunable_searcher("hyperopt"),
      tunable_searcher_for_retune(""),
      initial_tuning(1),
      retune_tunable(1),
      force_retune(-1),
      stop_on_converge(1),
      converge_type("val-accuracy"),
      plateau_size_for_convergence(5),
      plateau_size_for_retune(4),
      num_clocks_to_eval(0),
      num_samples_per_epoch(0),
      num_epoches_per_test(1),
      num_clocks_to_test(0),
      num_progress_samples(10),
      average_worker_progress(1),
      progress_in_logscale(0),
      searching_stop_slack(5),
      searching_stop_threshold(0.1),
      grid_size(10),
      shuffle_grid(1),
      print_interval(0.0) {}
};

struct GeePsConfig {
  uint num_branches;
  uint num_tables;
  uint num_processes;
  uint num_threads;
  std::vector<std::string> host_list;
  std::vector<uint> port_list;
  uint tcp_base_port;
  uint num_channels;
  std::string output_dir;
  iter_t start_clock;
  iter_t snapshot_interval;
  iter_t log_interval;
  uint prefetch;
  uint pp_policy;
  uint local_opt;
  uint affinity;
  uint num_cores;
  uint num_zones;
  int max_slack;
  size_t gpu_memory_capacity;
  int mm_warning_level;
  int read_my_writes;
  int pinned_cpu_memory;
  string update_func;
  float lr_decay;
  int lr_decay_every;

  MltunerConfig mltuner_config;

  GeePsConfig() :
    num_branches(1),
    num_tables(1),
    num_threads(1),
    tcp_base_port(9090),
    num_channels(1),
    output_dir(""),
    start_clock(0), snapshot_interval(0), log_interval(0),
    prefetch(1),
    pp_policy(3), local_opt(1), affinity(0),
    max_slack(10),
    gpu_memory_capacity(std::numeric_limits<size_t>::max()),
    mm_warning_level(1),
    read_my_writes(0), pinned_cpu_memory(0),
    update_func("blank"),
    lr_decay(0.0), lr_decay_every(0) {}
};
typedef GeePsConfig Config;

inline static int clocks_per_epoch(
    const Tunable& tunable, const GeePsConfig& config) {
  CHECK_LT(tunable.batch_size, config.mltuner_config.num_samples_per_epoch);
  int num_clocks =
      config.mltuner_config.num_samples_per_epoch / tunable.batch_size;
  return num_clocks;
}

inline static int clocks_per_test(
    const Tunable& tunable, const GeePsConfig& config) {
  CHECK_LT(tunable.batch_size, config.mltuner_config.num_samples_per_epoch);
  int num_clocks =
      config.mltuner_config.num_samples_per_epoch / tunable.batch_size;
  num_clocks *= config.mltuner_config.num_epoches_per_test;
  cout << "clocks_per_test = " << num_clocks << endl;
  return num_clocks;
}

inline static bool decide_accuracy_plateaued(
    const std::vector<double>& accuracies, int plateau_size) {
  if (accuracies.size() < plateau_size + 1) {
    return false;
  }
  double max = 0.0;
  for (int i = accuracies.size() - 1; i >= 0; i--) {
    if (accuracies[i] >= max) {
      max = accuracies[i];
      if (i < accuracies.size() - plateau_size) {
        return true;
      }
    }
  }
  return false;
}

inline static bool decide_loss_converged(
    const std::vector<double>& losses, const GeePsConfig& config) {
  CHECK(0);
  return false;
}

struct BranchManager;

class GeePs {
 public:
  GeePs(uint process_id, const GeePsConfig& config);

  void Shutdown();
  std::string GetStats();
  void StartIterations();

  /* Interfaces for virtual iteration */
  int VirtualRead(size_t table_id, const vector<size_t>& row_ids);
  int VirtualPostRead(int prestep_handle);
  int VirtualPreUpdate(size_t table_id, const vector<size_t>& row_ids);
  int VirtualUpdate(int prestep_handle);
  int VirtualLocalAccess(const vector<size_t>& row_ids, bool fetch);
  int VirtualPostLocalAccess(int prestep_handle, bool keep);
  int VirtualClock();
  void FinishVirtualIteration();

  /* Interfaces for real access */
  void Read(int handle, RowData **buffer_ptr);
  void PostRead(int handle);
  void PreUpdate(int handle, RowOpVal **buffer_ptr);
  void Update(int handle);
  void LocalAccess(int handle, RowData **buffer_ptr);
  void PostLocalAccess(int handle);
  void Clock();

  /* Interfaces for ModelSitter */
  void Report(double progress);
  Tunable GetTunable();
  void GetCurrentBranchInfo(Tunable *tunable, int *flag);
  boost::shared_ptr<BranchManager> GetBranchManager();
};

#endif  // defined __lazy_table_hpp__
