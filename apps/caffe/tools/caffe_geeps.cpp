#include <boost/program_options.hpp>
#include <boost/format.hpp>

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "geeps.hpp"
#include "caffe/caffe.hpp"

namespace po = boost::program_options;

using std::vector;
using std::string;

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

DEFINE_int32(worker_id, 0,
    "");
DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(machinefile, "",
    "Machine file path.");
DEFINE_string(ps_config, "",
    "Configuration file path.");
DEFINE_string(output_dir, "",
    "Output directory.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  CHECK_GT(FLAGS_gpu, -1) << "Need a device ID to query.";
  LOG(INFO) << "Querying device ID = " << FLAGS_gpu;
  caffe::Caffe::SetDevice(FLAGS_gpu);
  caffe::Caffe::DeviceQuery();
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

void parse_hostfile(const string& hostfile, vector<string>& hostlist) {
  std::ifstream is(hostfile.c_str());
  CHECK(is);
  std::string line;
  hostlist.clear();
  while (!!getline(is, line)) {
    hostlist.push_back(line);
  }
  is.close();
}

void parse_config_file(caffe::PsConfig& ps_config) {
  po::options_description desc("Allowed options");
  desc.add_options()
    /* Worker configs */
    // ("slack",
     // po::value<int>(&ps_config.slack),
     // "")
    ("batches_per_clock",
     po::value<int>(&(ps_config.batches_per_clock))
     ->default_value(1),
     "")
    ("multi_table",
     po::value<int>(&(ps_config.multi_table))
     ->default_value(1),
     "")
    ("layers_per_table",
     po::value<int>(&(ps_config.layers_per_table))
     ->default_value(1),
     "")
    ("restore_snapshot",
     po::value<string>(&(ps_config.snapshot_name))
     ->default_value(""),
     "")
    ("keep_momentum",
     po::value<int>(&(ps_config.keep_momentum))
     ->default_value(1),
     "")
    ("debug",
     po::value<int>(&(ps_config.debug))
     ->default_value(0),
     "")
    ("log_interval",
     po::value<int>(&(ps_config.log_interval))
     ->default_value(-1),
     "")
 
    /* GeePS configs */
    ("num_channels",
     po::value<uint32_t>(&(ps_config.geeps_config.num_channels)),
     "")
    ("mm_warning_level",
     po::value<int>(&(ps_config.geeps_config.mm_warning_level))
     ->default_value(0),
     "")
    ("gpu_memory_capacity",
     po::value<size_t>(&(ps_config.geeps_config.gpu_memory_capacity))
     ->default_value(std::numeric_limits<size_t>::max()),
     "")
    ("max_slack",
     po::value<int>(&(ps_config.geeps_config.max_slack))
     ->default_value(100),
     "")
    ("read_my_writes",
     po::value<int>(&(ps_config.geeps_config.read_my_writes))
     ->default_value(0),
     "")
    ("pinned_cpu_memory",
     po::value<int>(&(ps_config.geeps_config.pinned_cpu_memory))
     ->default_value(1),
     "")
    ("update_func",
     po::value<string>(&(ps_config.geeps_config.update_func))
     ->default_value("momentum"),
     "")
    ("lr_decay",
     po::value<float>(&(ps_config.geeps_config.lr_decay))
     ->default_value(0.0),
     "")
    ("lr_decay_every",
     po::value<int>(&(ps_config.geeps_config.lr_decay_every))
     ->default_value(0),
     "")
    ("num_branches",
     po::value<uint32_t>(&(ps_config.geeps_config.num_branches))
     ->default_value(1),
     "")

    /* MLtuner configs */
    ("tuner_logic",
     po::value<string>(&(ps_config.geeps_config.mltuner_config.tuner_logic))
     ->default_value("mltuner"),
     "")
    ("tunable_searcher",
     po::value<string>(&(ps_config.geeps_config.mltuner_config.tunable_searcher))
     ->default_value("hyperopt"),
     "")
    ("tunable_searcher_for_retune",
     po::value<string>(&(ps_config.geeps_config.mltuner_config.tunable_searcher_for_retune))
     ->default_value(""),
     "")
    ("initial_tuning",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.initial_tuning))
     ->default_value(1),
     "")
    ("retune_tunable",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.retune_tunable))
     ->default_value(1),
     "")
    ("force_retune",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.force_retune))
     ->default_value(-1),
     "")
    ("stop_on_converge",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.stop_on_converge))
     ->default_value(1),
     "")
    ("plateau_size_for_convergence",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.plateau_size_for_convergence))
     ->default_value(5),
     "")
    ("plateau_size_for_retune",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.plateau_size_for_retune))
     ->default_value(4),
     "")
    ("num_samples_per_epoch",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.num_samples_per_epoch)),
     "")
    ("num_epoches_per_test",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.num_epoches_per_test))
     ->default_value(1),
     "")
    ("num_clocks_to_eval",
     po::value<uint32_t>(&(ps_config.geeps_config.mltuner_config.num_clocks_to_eval))
     ->default_value(100),
     "")
    ("test_batch_size",
     po::value<float>(&(ps_config.geeps_config.mltuner_config.tunable_for_test.batch_size)),
     "")
    ("num_clocks_to_test",
     po::value<uint>(&(ps_config.geeps_config.mltuner_config.num_clocks_to_test)),
     "")
    ("time_to_try",
     po::value<double>(&(ps_config.geeps_config.mltuner_config.time_to_try)),
     "")
    ("num_progress_samples",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.num_progress_samples))
     ->default_value(10),
     "")
    ("average_worker_progress",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.average_worker_progress))
     ->default_value(1),
     "")
    ("progress_in_logscale",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.progress_in_logscale))
     ->default_value(1),
     "")
    ("mltuner_print_interval",
     po::value<double>(&(ps_config.geeps_config.mltuner_config.print_interval))
     ->default_value(-1),
     "")
    ("searching_stop_slack",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.searching_stop_slack))
     ->default_value(5),
     "")
    ("searching_stop_threshold",
     po::value<double>(&(ps_config.geeps_config.mltuner_config.searching_stop_threshold))
     ->default_value(0.1),
     "")
    ("app_name",
     po::value<string>(&(ps_config.geeps_config.mltuner_config.app_name)),
     "")
    ("spearmint_root",
     po::value<string>(&(ps_config.geeps_config.mltuner_config.spearmint_root)),
     "")
    ("hyperopt_root",
     po::value<string>(&(ps_config.geeps_config.mltuner_config.hyperopt_root)),
     "")
    ("grid_size",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.grid_size))
     ->default_value(10),
     "")
    ("shuffle_grid",
     po::value<int>(&(ps_config.geeps_config.mltuner_config.shuffle_grid))
     ->default_value(1),
     "")
    ("tune_lr",
     po::value<int>(&(ps_config.tune_lr))
     ->default_value(1),
     "")
    ("tune_momentum",
     po::value<int>(&(ps_config.tune_momentum))
     ->default_value(1),
     "")
    ("tune_slack",
     po::value<int>(&(ps_config.tune_slack))
     ->default_value(1),
     "")
    ("tune_batch_size",
     po::value<int>(&(ps_config.tune_batch_size))
     ->default_value(1),
     "")
    ("tune_lr_decay",
     po::value<int>(&(ps_config.tune_lr_decay))
     ->default_value(0),
     "")
    ("hardcoded_lr",
     po::value<float>(&(ps_config.geeps_config.mltuner_config.tunable_for_train.lr))
     ->default_value(std::numeric_limits<float>::quiet_NaN()),
     "")
    ("hardcoded_slack",
     po::value<float>(&(ps_config.geeps_config.mltuner_config.tunable_for_train.slack))
     ->default_value(std::numeric_limits<float>::quiet_NaN()),
     "")
    ("hardcoded_batch_size",
     po::value<float>(&(ps_config.geeps_config.mltuner_config.tunable_for_train.batch_size))
     ->default_value(std::numeric_limits<float>::quiet_NaN()),
     "")
    ("hardcoded_momentum",
     po::value<float>(&(ps_config.geeps_config.mltuner_config.tunable_for_train.momentum))
     ->default_value(std::numeric_limits<float>::quiet_NaN()),
     "")
    ("hardcoded_lr_decay",
     po::value<float>(&(ps_config.geeps_config.mltuner_config.tunable_for_train.lr_decay))
     ->default_value(std::numeric_limits<float>::quiet_NaN()),
     "");
  std::ifstream config_in(FLAGS_ps_config.c_str());
  CHECK(config_in);
  po::variables_map vm;
  po::store(po::parse_config_file(config_in, desc), vm);
  po::notify(vm);
}

// Train / Finetune a model.
int train() {
  // Cui: change solver file name according to the worker_id
  int worker_id = FLAGS_worker_id;
  FLAGS_solver = (boost::format("%s.%i") % FLAGS_solver % worker_id).str();
  LOG(INFO) << "Use solver " << FLAGS_solver;

  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu < 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    FLAGS_gpu = solver_param.device_id();
  }

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  /* Cui: prepare PS config */
  caffe::PsConfig ps_config;
  parse_config_file(ps_config);
  parse_hostfile(FLAGS_machinefile, ps_config.geeps_config.host_list);
  ps_config.gradient_only = ps_config.geeps_config.update_func != "blank";
  ps_config.geeps_config.num_processes =
      ps_config.geeps_config.host_list.size();
  ps_config.num_workers = ps_config.geeps_config.num_processes;
  CHECK_LT(worker_id, ps_config.num_workers);
  ps_config.worker_id = worker_id;
  ps_config.geeps_config.output_dir = FLAGS_output_dir;

  MltunerConfig& mltuner_config = ps_config.geeps_config.mltuner_config;
  TunableSpecs& tunable_specs = mltuner_config.tunable_specs;

  TunableSpec lr_spec;
  lr_spec.name = "lr";
  lr_spec.type = "continuous-log";
  lr_spec.to_search = ps_config.tune_lr;
  lr_spec.min_val = -5;
  lr_spec.max_val = 0;
  lr_spec.default_val = -2;
  tunable_specs.push_back(lr_spec);

  TunableSpec slack_spec;
  slack_spec.name = "slack";
  slack_spec.type = "discrete";
  slack_spec.to_search = ps_config.tune_slack;
  slack_spec.valid_vals.push_back(0);
  slack_spec.valid_vals.push_back(1);
  slack_spec.valid_vals.push_back(3);
  slack_spec.valid_vals.push_back(7);
  slack_spec.default_val = 0;
  tunable_specs.push_back(slack_spec);

  TunableSpec batch_size_spec;
  batch_size_spec.name = "batch_size";
  batch_size_spec.type = "discrete";
  batch_size_spec.to_search = ps_config.tune_batch_size;
  if (mltuner_config.app_name.find("alexnet") == 0) {
    batch_size_spec.valid_vals.push_back(4);
    batch_size_spec.valid_vals.push_back(16);
    batch_size_spec.valid_vals.push_back(64);
    batch_size_spec.valid_vals.push_back(256);
    batch_size_spec.default_val = 3;
  } else if (mltuner_config.app_name.find("googlenet") == 0) {
    batch_size_spec.valid_vals.push_back(2);
    batch_size_spec.valid_vals.push_back(4);
    batch_size_spec.valid_vals.push_back(8);
    batch_size_spec.valid_vals.push_back(16);
    batch_size_spec.valid_vals.push_back(32);
    batch_size_spec.default_val = 4;
  } else if (mltuner_config.app_name.find("inceptionv3") == 0) {
    batch_size_spec.valid_vals.push_back(2);
    batch_size_spec.valid_vals.push_back(4);
    batch_size_spec.valid_vals.push_back(8);
    batch_size_spec.valid_vals.push_back(16);
    batch_size_spec.default_val = 3;
  } else if (mltuner_config.app_name.find("rnn") == 0) {
    batch_size_spec.valid_vals.push_back(1);
    batch_size_spec.default_val = 0;
  } else {
    CHECK(0) << "Unknown app name " << mltuner_config.app_name;
  }
  tunable_specs.push_back(batch_size_spec);

  TunableSpec momentum_spec;
  momentum_spec.name = "momentum";
  momentum_spec.type = "continuous-linear";
  momentum_spec.to_search = ps_config.tune_momentum;
  momentum_spec.min_val = 0.0;
  momentum_spec.max_val = 1.0;
  momentum_spec.default_val = 0.9;
  tunable_specs.push_back(momentum_spec);

  TunableSpec lr_decay_spec;
  lr_decay_spec.name = "lr_decay";
  lr_decay_spec.type = "continuous-linear";
  lr_decay_spec.to_search = ps_config.tune_lr_decay;
  lr_decay_spec.min_val = 0.0;
  lr_decay_spec.max_val = 1.0;
  lr_decay_spec.default_val = 0.0;
  tunable_specs.push_back(lr_decay_spec);

  mltuner_config.tunable_for_eval.init_default(tunable_specs);
  mltuner_config.tunable_for_eval.lr = 0.0;
  mltuner_config.tunable_for_eval.slack = 0;
  mltuner_config.tunable_for_eval.batch_size =
      mltuner_config.tunable_for_test.batch_size;
  mltuner_config.tunable_for_test.init_default(tunable_specs);
  mltuner_config.tunable_for_test.lr = 0.0;
  mltuner_config.tunable_for_test.slack = 10;

  LOG(INFO) << "Starting Optimization";
  shared_ptr<caffe::Solver<float> >
    solver(caffe::SolverRegistry<float>::CreateSolver(
        solver_param, &ps_config));

  if (FLAGS_snapshot.size()) {
    ps_config.snapshot_name = FLAGS_snapshot;
  }
  if (ps_config.snapshot_name.size()) {
    LOG(INFO) << "Resuming from " << ps_config.snapshot_name;
    solver->Solve(ps_config.snapshot_name);
  } else if (FLAGS_weights.size()) {
    CopyLayers(&*solver, FLAGS_weights);
    solver->Solve();
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";

  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(bottom_vec, &iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // FLAGS_alsologtostderr = 0;

  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
    return GetBrewFunction(caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
