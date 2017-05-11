#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <string>
#include <vector>
#include <set>

#include "caffe/net.hpp"

#include "geeps.hpp"

#include "branch-scheduler/branch-scheduler-utils.hpp"
  /* FIXME: it's a dirty hack for using the BranchManager */

namespace caffe {

struct PsConfig {
  int worker_id;
  int num_workers;
  int batches_per_clock;
  int multi_table;
  int layers_per_table;
  bool gradient_only;
  string snapshot_name;
  int keep_momentum;
  int tune_lr;
  int tune_slack;
  int tune_batch_size;
  int tune_momentum;
  int tune_lr_decay;
  int debug;
  int log_interval;
  GeePsConfig geeps_config;
  PsConfig() : batches_per_clock(1),
      multi_table(1), layers_per_table(1), gradient_only(false),
      snapshot_name(""), keep_momentum(1),
      tune_lr(1), tune_slack(1), tune_batch_size(1),
      tune_momentum(0), tune_lr_decay(0),
      debug(0), log_interval(0) {}
};

struct RowAccessInfo {
  vector<size_t> row_ids;
  int num_vals;
  bool data_in_mem;  /* Volatile field only used at virtual iteration */
  int data_handle;  /* Volatile field only used at virtual iteration */
};

struct ParamInfo {
  int global_param_id;
  int val_offset;
};

struct ImbInfo {
  int global_imb_id;
  bool fetch;
  bool keep;
  ImbInfo(int g = -1, bool f = false, bool k = false) :
      global_imb_id(g), fetch(f), keep(k) {}
};

struct FetchKeep {
  bool fetch;
  bool keep;
  FetchKeep(bool f = false, bool k = false) : fetch(f), keep(k) {}
};

struct LayerHandles {
  int read_handle;
  int postread_handle;
  int bw_read_handle;
  int bw_postread_handle;
  int prewrite_handle;
  int write_handle;
  int history_access_handle;
  int history_postaccess_handle;
  vector<int> imbs_to_access_fw;
  vector<int> imbs_to_release_fw;
  vector<int> imb_diffs_to_access_fw;
  vector<int> imb_diffs_to_release_fw;
  vector<int> imbs_to_access_bw;
  vector<int> imbs_to_release_bw;
  vector<int> imb_diffs_to_access_bw;
  vector<int> imb_diffs_to_release_bw;
};

typedef std::map<int, FetchKeep> IntSet;
struct LayerInfo {
  bool layer_need_backward;
  vector<bool> bottom_need_backward;
  bool local_param;
  size_t table_id;
  vector<size_t> row_ids;
  vector<size_t> history_data_row_ids;
  size_t num_vals;
  vector<ParamInfo> param_infos;
  IntSet imbs_used_fw;
  IntSet imb_diffs_used_fw;
  IntSet imbs_used_bw;
  IntSet imb_diffs_used_bw;
  vector<ImbInfo> imbs_to_access_fw;
  vector<ImbInfo> imbs_to_release_fw;
  vector<ImbInfo> imb_diffs_to_access_fw;
  vector<ImbInfo> imb_diffs_to_release_fw;
  vector<ImbInfo> imbs_to_access_bw;
  vector<ImbInfo> imbs_to_release_bw;
  vector<ImbInfo> imb_diffs_to_access_bw;
  vector<ImbInfo> imb_diffs_to_release_bw;
  size_t param_size;
  size_t imb_size;
  vector<LayerHandles> layer_handles;
  double fw_read_time;
  double fw_compute_time;
  double fw_write_time;
  double bw_read_time;
  double bw_compute_time;
  double bw_write_time;
  double test_time;
  double snapshot_model_time;
  double snapshot_solverstate_time;
};

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param, const PsConfig& ps_config);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();
  void InitPs();
  void PrepareAccessInfo();
  void InitSnapshot();
  void InitNetParameterSnapshot();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual ~Solver() {}
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() { return iter_; }

 protected:
  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;
  virtual Dtype ForwardBackwardUsingPs(
      const vector<Blob<Dtype>* > & bottom,
      const shared_ptr<Net<Dtype> >& net,
      const Tunable& tunable, int branch_flag, bool do_snapshot) = 0;
  virtual void InitSolverStateSnapshot() = 0;
  virtual void InitPsValues() = 0;
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(SolverState* state) = 0;
  virtual void RestoreSolverState(const SolverState& state) = 0;
  void DisplayOutputBlobs(const int net_id);

  SolverParameter param_;

  PsConfig ps_config_;
  shared_ptr<GeePs> ps_;

  vector<RowAccessInfo> imb_data_infos_;
  vector<RowAccessInfo> imb_diff_infos_;
  int num_tables_;
  vector<LayerInfo> layer_infos_;
  vector<Blob<Dtype>*> test_net_output_blobs_;

  NetParameter snapshot_net_param_protobuf_;
  SolverState snapshot_solver_state_protobuf_;

  int iter_;
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};


/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param, const PsConfig& ps_config)
      : Solver<Dtype>(param, ps_config) { PreSolve(); }
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
  void PreSolve();
  Dtype GetLearningRate();
  virtual void ApplyUpdate();
  virtual Dtype ForwardBackwardUsingPs(
      const vector<Blob<Dtype>* > & bottom,
      const shared_ptr<Net<Dtype> >& net,
      const Tunable& tunable, int branch_flag, bool do_snapshot);
  virtual void InitPsValues();
  virtual void InitSolverStateSnapshot();
  virtual void Normalize(int param_id);
  virtual void Regularize(int param_id);
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  virtual void ClipGradients();
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(
      const SolverParameter& param, const PsConfig& ps_config)
      : SGDSolver<Dtype>(param, ps_config) {}
  explicit NesterovSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {}

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(
      const SolverParameter& param, const PsConfig& ps_config)
      : SGDSolver<Dtype>(param, ps_config) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};

template <typename Dtype>
Solver<Dtype>* GetSolver(
    const SolverParameter& param, const PsConfig& ps_config) {
  SolverParameter_SolverType type = param.solver_type();

  switch (type) {
  case SolverParameter_SolverType_SGD:
      return new SGDSolver<Dtype>(param, ps_config);
  case SolverParameter_SolverType_NESTEROV:
      return new NesterovSolver<Dtype>(param, ps_config);
  case SolverParameter_SolverType_ADAGRAD:
      return new AdaGradSolver<Dtype>(param, ps_config);
  default:
      LOG(FATAL) << "Unknown SolverType: " << type;
  }
  return (Solver<Dtype>*) NULL;
}

template <typename Dtype>
Solver<Dtype>* GetSolver(
    const SolverParameter& param) {
  SolverParameter_SolverType type = param.solver_type();
  PsConfig ps_config;

  switch (type) {
  case SolverParameter_SolverType_SGD:
      return new SGDSolver<Dtype>(param, ps_config);
  case SolverParameter_SolverType_NESTEROV:
      return new NesterovSolver<Dtype>(param, ps_config);
  case SolverParameter_SolverType_ADAGRAD:
      return new AdaGradSolver<Dtype>(param, ps_config);
  default:
      LOG(FATAL) << "Unknown SolverType: " << type;
  }
  return (Solver<Dtype>*) NULL;
}

}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
