/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <string>
#include <vector>

#include "tablet-server.hpp"

using std::string;
using std::cerr;
using std::cout;
using std::endl;
using std::vector;

void TabletStorage::init_user_defined_update_states(
    int branch_id, uint branch_idx, uint table_id, size_t batch_size) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];

  uint temp_size = 0;
  uint history_size = 0;
  if (config.update_func == "blank") {
    /* Blank */
    temp_size = 0;
    history_size = 0;
  } else if (config.update_func == "momentum") {
    /* Momentum with learning rate decay */
    temp_size = 0;
    history_size = 1;
  } else if (config.update_func == "adadelta") {
    /* AdaDelta */
    temp_size = 2;
    history_size = 2;
  } else if (config.update_func == "adagrad") {
    /* AdaGrad */
    temp_size = 1;
    history_size = 1;
  } else if (config.update_func == "adam") {
    /* Adam */
    temp_size = 1;
    history_size = 2;
  } else if (config.update_func == "nesterov") {
    /* Nesterov */
    temp_size = 1;
    history_size = 1;
  } else if (config.update_func == "rmsprop") {
    /* RMSProp */
    temp_size = 1;
    history_size = 1;
  } else if (config.update_func == "adarevision+momentum") {
    /* AdaRevision with momentum */
    temp_size = 0;
    history_size = 5;
  } else if (config.update_func == "adadelta+momentum") {
    /* AdaDelta with momentum */
    temp_size = 0;
    history_size = 3;
  } else if (config.update_func == "adagrad+momentum") {
    /* AdaGrad with momentum */
    temp_size = 0;
    history_size = 2;
  } else if (config.update_func == "rmsprop+momentum") {
    /* RMSProp with momentum */
    temp_size = 0;
    history_size = 2;
  } else {
    CHECK(0) << " Unknown update function: " << config.update_func;
  }
  CHECK(!model_branch.history_store.size());
  model_branch.history_store.init(batch_size * history_size, DataStorage::CPU);
  model_branch.history_store.zerofy_data_cpu();
  CHECK(!model_branch.temp_store.size());
  model_branch.temp_store.init(batch_size * temp_size, DataStorage::CPU);
  model_branch.temp_store.zerofy_data_cpu();
}

void TabletStorage::apply_user_defined_updates(
    int branch_id, uint branch_idx, uint table_id, RowOpVal *updates, size_t batch_size,
    const Tunable& tunable) {
  if (config.update_func == "momentum") {
    /* Momentum with learning rate decay */
    apply_updates_momentum(
        branch_id, branch_idx, table_id, updates, batch_size, tunable);
  } else if (config.update_func == "adadelta") {
    /* AdaDelta */
    apply_updates_adadelta(
        branch_id, branch_idx, table_id, updates, batch_size, tunable);
  } else if (config.update_func == "adagrad") {
    /* AdaGrad */
    apply_updates_adagrad(
        branch_id, branch_idx, table_id, updates, batch_size, tunable);
  } else if (config.update_func == "adam") {
    /* Adam */
    apply_updates_adam(
        branch_id, branch_idx, table_id, updates, batch_size, tunable);
  } else if (config.update_func == "nesterov") {
    /* Nesterov */
    apply_updates_nesterov(
        branch_id, branch_idx, table_id, updates, batch_size, tunable);
  } else if (config.update_func == "rmsprop") {
    /* RMSProp */
    apply_updates_rmsprop(
        branch_id, branch_idx, table_id, updates, batch_size, tunable);
  } else if (config.update_func == "adarevision+momentum") {
    /* AdaRevision with momentum */
    apply_updates_adarevision_with_momentum(
        branch_id, branch_idx, table_id, updates, batch_size, tunable);
  } else if (config.update_func == "adadelta+momentum") {
    /* AdaDelta with momentum */
    apply_updates_adadelta_with_momentum(
        branch_id, branch_idx, table_id, updates, batch_size, tunable);
  } else if (config.update_func == "adagrad+momentum") {
    /* AdaGrad with momentum */
    apply_updates_adagrad_with_momentum(
        branch_id, branch_idx, table_id, updates, batch_size, tunable);
  } else if (config.update_func == "rmsprop+momentum") {
    /* RMSProp with momentum */
    apply_updates_rmsprop_with_momentum(
        branch_id, branch_idx, table_id, updates, batch_size, tunable);
  } else {
    CHECK(0) << " Unknown update function: " << config.update_func;
  }
}

void TabletStorage::apply_updates_momentum(
    int branch_id, uint branch_idx,
    uint table_id, RowOpVal *update_rows, size_t batch_size,
    const Tunable& tunable) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];
  val_t *update = reinterpret_cast<val_t *>(update_rows);
  CHECK_EQ(model_branch.store.size(), batch_size);
  CHECK_GE(model_branch.history_store.size(), batch_size);
  val_t *master_data = reinterpret_cast<val_t *>(model_branch.store.data());
  val_t *history = reinterpret_cast<val_t *>(model_branch.history_store.data());
  size_t num_vals = batch_size * ROW_DATA_SIZE;
  val_t momentum = tunable.momentum;
  val_t neg_lr = tunable.lr * -1;
  if (config.lr_decay > 0.0f) {
    val_t lr_decay = config.lr_decay;
    int lr_decay_every = config.lr_decay_every;
    CHECK_LE(lr_decay, 1.0f);
    /* Currently, our lr_decay does not support
     * forking from another training branch.
     * We assume that the training branch runs from the beginning. */
    BranchInfo *branch_info = branch_manager->branch_info_map[branch_id];
    CHECK(branch_info);
    int num_clocks = branch_info->internal_clock;
    int num_epoches = num_clocks / clocks_per_epoch(tunable, config);
    CHECK_GT(lr_decay_every, 0);
    int num_decays = num_epoches / lr_decay_every;
    neg_lr *= pow(1.0f - lr_decay, num_decays);
  }

  cpu_axpby(num_vals, (1.0f - momentum),
            update, momentum,
            history);

  cpu_axpy(num_vals,
      neg_lr,
      history,
      master_data);
}

void TabletStorage::apply_updates_adadelta(
    int branch_id, uint branch_idx,
    uint table_id, RowOpVal *update_rows, size_t batch_size,
    const Tunable& tunable) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];
  val_t *update = reinterpret_cast<val_t *>(update_rows);
  CHECK_EQ(model_branch.store.size(), batch_size);
  CHECK_EQ(model_branch.history_store.size(), batch_size * 2);
  CHECK_EQ(model_branch.temp_store.size(), batch_size * 2);
  val_t *master_data =
      reinterpret_cast<val_t *>(model_branch.store.data());
  val_t *history1 =
      reinterpret_cast<val_t *>(model_branch.history_store.data());
  val_t *history2 =
      reinterpret_cast<val_t *>(&model_branch.history_store.data()[batch_size]);
  val_t *temp1 =
      reinterpret_cast<val_t *>(model_branch.temp_store.data());
  val_t *temp2 =
      reinterpret_cast<val_t *>(&model_branch.temp_store.data()[batch_size]);
  size_t num_vals = batch_size * ROW_DATA_SIZE;
  val_t delta = 1e-8;
  val_t momentum = 0.9;
  val_t neg_lr = tunable.lr * -1;

  // compute square of gradient in update
  cpu_powx(num_vals,
      update, val_t(2.0),
      temp1);

  // update history of gradients
  cpu_axpby(num_vals, val_t(1.0) - momentum,
      temp1, momentum,
      history1);

  // add delta to history to guard against dividing by zero later
  cpu_set(num_vals, delta,
      temp2);

  cpu_add(num_vals,
      temp2,
      history2,
      temp1);

  cpu_add(num_vals,
      temp2,
      history1,
      temp2);

  // divide history of updates by history of gradients
  cpu_div(num_vals,
      temp1,
      temp2,
      temp1);

  // jointly compute the RMS of both for update and gradient history
  cpu_powx(num_vals,
      temp1, val_t(0.5),
      temp1);

  // compute the update
  cpu_mul(num_vals,
      update,
      temp1,
      update);

  // compute square of update
  cpu_powx(num_vals,
      update, val_t(2.0),
      temp1);

  // update history of updates
  cpu_axpby(num_vals, val_t(1.0) - momentum,
      temp1, momentum,
      history2);

  // apply learning rate and add to master data
  // cpu_cpu_scale(num_vals, neg_lr,
      // update,
      // update);
  cpu_axpy(num_vals, neg_lr,
      update,
      master_data);
}

void TabletStorage::apply_updates_adagrad(
    int branch_id, uint branch_idx,
    uint table_id, RowOpVal *update_rows, size_t batch_size,
    const Tunable& tunable) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];
  val_t *update = reinterpret_cast<val_t *>(update_rows);
  CHECK_EQ(model_branch.store.size(), batch_size);
  CHECK_EQ(model_branch.history_store.size(), batch_size);
  CHECK_EQ(model_branch.temp_store.size(), batch_size);
  val_t *master_data = reinterpret_cast<val_t *>(model_branch.store.data());
  val_t *history = reinterpret_cast<val_t *>(model_branch.history_store.data());
  val_t *temp = reinterpret_cast<val_t *>(model_branch.temp_store.data());
  size_t num_vals = batch_size * ROW_DATA_SIZE;
  val_t delta = 1e-8;
  val_t neg_lr = tunable.lr * -1;

  // compute square of gradient in update
  cpu_powx(num_vals,
      update, val_t(2.0),
      temp);

  // update history
  cpu_add(num_vals,
      temp,
      history,
      history);

  // prepare update
  cpu_powx(num_vals,
            history, val_t(0.5),
            temp);

  cpu_add_scalar(num_vals,
            delta, temp);

  cpu_div(num_vals,
            update,
            temp,
            temp);

  // scale and apply
  cpu_axpy(num_vals, neg_lr,
      temp,
      master_data);
}

void TabletStorage::apply_updates_adam(
    int branch_id, uint branch_idx,
    uint table_id, RowOpVal *update_rows, size_t batch_size,
    const Tunable& tunable) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];
  val_t *update = reinterpret_cast<val_t *>(update_rows);
  CHECK_EQ(model_branch.store.size(), batch_size);
  CHECK_EQ(model_branch.history_store.size(), batch_size * 2);
  CHECK_EQ(model_branch.temp_store.size(), batch_size);
  val_t *master_data = reinterpret_cast<val_t *>(model_branch.store.data());
  val_t *history1 = reinterpret_cast<val_t *>(model_branch.history_store.data());
  val_t *history2 =
      reinterpret_cast<val_t *>(&model_branch.history_store.data()[batch_size]);
  val_t *temp = reinterpret_cast<val_t *>(model_branch.temp_store.data());
  size_t num_vals = batch_size * ROW_DATA_SIZE;
  val_t eps_hat = 1e-8;
  const val_t beta1 = 0.9;
  const val_t beta2 = 0.999;
  val_t neg_lr = tunable.lr * -1;
  const int t = clock_min(data_table.vec_clock);
  const val_t correction = std::sqrt(val_t(1) - pow(beta2, t)) /
      (val_t(1.0) - pow(beta1, t));

  // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
  cpu_axpby(num_vals, val_t(1)-beta1,
      update, beta1,
      history1);

  // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
  cpu_mul(num_vals,
      update,
      update,
      temp);
  cpu_axpby(num_vals, val_t(1)-beta2,
      temp, beta2,
      history2);

  // set update
  cpu_powx(num_vals,
      history2, val_t(0.5),
      temp);
  cpu_add_scalar(num_vals, eps_hat, temp);
  cpu_div(num_vals,
      history1,
      temp,
      temp);

  // cpu_scale(num_vals, neg_lr*correction,
      // temp,
      // update);
  // scale and apply
  cpu_axpy(num_vals, neg_lr*correction,
      temp,
      master_data);
}

void TabletStorage::apply_updates_nesterov(
    int branch_id, uint branch_idx,
    uint table_id, RowOpVal *update_rows, size_t batch_size,
    const Tunable& tunable) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];
  val_t *update = reinterpret_cast<val_t *>(update_rows);
  CHECK_EQ(model_branch.store.size(), batch_size);
  CHECK_EQ(model_branch.history_store.size(), batch_size);
  CHECK_EQ(model_branch.temp_store.size(), batch_size);
  val_t *master_data = reinterpret_cast<val_t *>(model_branch.store.data());
  val_t *history = reinterpret_cast<val_t *>(model_branch.history_store.data());
  val_t *temp = reinterpret_cast<val_t *>(model_branch.temp_store.data());
  size_t num_vals = batch_size * ROW_DATA_SIZE;
  val_t momentum = 0.9;
  val_t neg_lr = tunable.lr * -1;

  // save history momentum for stepping back
  cpu_copy(num_vals,
      history,
      temp);

  // update history
  cpu_axpby(num_vals, neg_lr,
            update, momentum,
            history);

  // compute update: step back then over step
  cpu_axpby(num_vals, val_t(1) + momentum,
      history, -momentum,
      temp);

  // cpu_copy(num_vals,
      // temp,
      // update);
  // apply
  cpu_add(num_vals,
      temp,
      master_data,
      master_data);
}

void TabletStorage::apply_updates_rmsprop(
    int branch_id, uint branch_idx,
    uint table_id, RowOpVal *update_rows, size_t batch_size,
    const Tunable& tunable) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];
  val_t *update = reinterpret_cast<val_t *>(update_rows);
  CHECK_EQ(model_branch.store.size(), batch_size);
  CHECK_EQ(model_branch.history_store.size(), batch_size);
  CHECK_EQ(model_branch.temp_store.size(), batch_size);
  val_t *master_data = reinterpret_cast<val_t *>(model_branch.store.data());
  val_t *history = reinterpret_cast<val_t *>(model_branch.history_store.data());
  val_t *temp = reinterpret_cast<val_t *>(model_branch.temp_store.data());
  size_t num_vals = batch_size * ROW_DATA_SIZE;
  val_t delta = 1e-8;
  val_t rms_decay = 0.99;
  val_t neg_lr = tunable.lr * -1;

  // compute square of gradient in update
  cpu_powx(num_vals,
      update, val_t(2),
      temp);

  // update history
  cpu_axpby(num_vals,
      val_t(1-rms_decay), temp,
      rms_decay, history);

  // prepare update
  cpu_powx(num_vals,
      history, val_t(0.5),
      temp);

  cpu_add_scalar(num_vals,
      delta, temp);

  cpu_div(num_vals,
      update, temp,
      temp);

  // scale and copy
  // cpu_cpu_axpby(num_vals, neg_lr,
      // temp, val_t(0),
      // update);
  cpu_axpy(num_vals, neg_lr,
      temp,
      master_data);
}

void TabletStorage::apply_updates_adarevision_with_momentum(
    int branch_id, uint branch_idx,
    uint table_id, RowOpVal *update_rows, size_t batch_size,
    const Tunable& tunable) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];
  val_t *gradient = reinterpret_cast<val_t *>(update_rows);
  CHECK_EQ(model_branch.store.size(), batch_size);
  CHECK_EQ(model_branch.history_store.size(), batch_size * 5);
  val_t *master_data = reinterpret_cast<val_t *>(model_branch.store.data());
  val_t *accum_gradients = reinterpret_cast<val_t *>(
      &model_branch.history_store.data()[0]);
  val_t *old_accum_gradients = reinterpret_cast<val_t *>(
      &model_branch.history_store.data()[batch_size * 1]);
  val_t *z = reinterpret_cast<val_t *>(
      &model_branch.history_store.data()[batch_size * 2]);
  val_t *z_max = reinterpret_cast<val_t *>(
      &model_branch.history_store.data()[batch_size * 3]);
  val_t *smoothed_gradient = reinterpret_cast<val_t *>(
      &model_branch.history_store.data()[batch_size * 4]);
  size_t num_vals = batch_size * ROW_DATA_SIZE;
  val_t momentum = 0.9;
  val_t delta = 1;
  // val_t delta = 1e-8;
  val_t neg_lr = tunable.lr * -1;

  for (uint i = 0; i < num_vals; i++) {
    smoothed_gradient[i] =
        smoothed_gradient[i] * momentum + gradient[i] * (1 - momentum);
    val_t g_bck = accum_gradients[i] - old_accum_gradients[i];
    val_t eta_old = neg_lr / sqrt(z_max[i] + delta);
    z[i] += smoothed_gradient[i] * smoothed_gradient[i]
            + 2 * smoothed_gradient[i] * g_bck;
    z_max[i] = std::max(z[i], z_max[i]);
    val_t eta = neg_lr / sqrt(z_max[i] + delta);
    val_t adjusted_update =
        eta * smoothed_gradient[i] + (eta - eta_old) * g_bck;
    master_data[i] += adjusted_update;
    accum_gradients[i] += smoothed_gradient[i];
  }
}

void TabletStorage::apply_updates_adadelta_with_momentum(
    int branch_id, uint branch_idx,
    uint table_id, RowOpVal *update_rows, size_t batch_size,
    const Tunable& tunable) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];
  val_t *gradient = reinterpret_cast<val_t *>(update_rows);
  CHECK_EQ(model_branch.store.size(), batch_size);
  CHECK_EQ(model_branch.history_store.size(), batch_size * 3);
  val_t *master_data = reinterpret_cast<val_t *>(model_branch.store.data());
  val_t *gradient_square_history = reinterpret_cast<val_t *>(
      &model_branch.history_store.data()[0]);
  val_t *update_square_history = reinterpret_cast<val_t *>(
      &model_branch.history_store.data()[batch_size * 1]);
  val_t *smoothed_gradient = reinterpret_cast<val_t *>(
      &model_branch.history_store.data()[batch_size * 2]);
  size_t num_vals = batch_size * ROW_DATA_SIZE;
  val_t momentum = 0.9;
  val_t delta = 1e-8;
  val_t adadelta_momentum = 0.9;
  val_t neg_lr = tunable.lr * -1;

  for (uint i = 0; i < num_vals; i++) {
    smoothed_gradient[i] =
        smoothed_gradient[i] * momentum + gradient[i] * (1 - momentum);
    gradient_square_history[i] =
        gradient_square_history[i] * adadelta_momentum
        + smoothed_gradient[i] * smoothed_gradient[i] * (1 - adadelta_momentum);
    val_t eta = neg_lr * sqrt(update_square_history[i] + delta)
        / sqrt(gradient_square_history[i] + delta);
    val_t adjusted_update = eta * smoothed_gradient[i];
    update_square_history[i] =
        update_square_history[i] * adadelta_momentum
        + adjusted_update * adjusted_update * (1 - adadelta_momentum);
    master_data[i] += adjusted_update;
  }
}

void TabletStorage::apply_updates_adagrad_with_momentum(
    int branch_id, uint branch_idx,
    uint table_id, RowOpVal *update_rows, size_t batch_size,
    const Tunable& tunable) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];
  val_t *gradient = reinterpret_cast<val_t *>(update_rows);
  CHECK_EQ(model_branch.store.size(), batch_size);
  CHECK_EQ(model_branch.history_store.size(), batch_size * 2);
  val_t *master_data = reinterpret_cast<val_t *>(model_branch.store.data());
  val_t *gradient_square_history = reinterpret_cast<val_t *>(
      &model_branch.history_store.data()[0]);
  val_t *smoothed_gradient = reinterpret_cast<val_t *>(
      &model_branch.history_store.data()[batch_size * 1]);
  size_t num_vals = batch_size * ROW_DATA_SIZE;
  val_t momentum = 0.9;
  val_t delta = 1e-8;
  val_t neg_lr = tunable.lr * -1;

  for (uint i = 0; i < num_vals; i++) {
    smoothed_gradient[i] =
        smoothed_gradient[i] * momentum + gradient[i] * (1 - momentum);
    gradient_square_history[i] =
        gradient_square_history[i]
        + smoothed_gradient[i] * smoothed_gradient[i];
    val_t eta = neg_lr / sqrt(gradient_square_history[i] + delta);
    val_t adjusted_update = eta * smoothed_gradient[i];
    master_data[i] += adjusted_update;
  }
}

void TabletStorage::apply_updates_rmsprop_with_momentum(
    int branch_id, uint branch_idx,
    uint table_id, RowOpVal *update_rows, size_t batch_size,
    const Tunable& tunable) {
  CHECK_LT(table_id, data_tables.size());
  DataTable& data_table = data_tables[table_id];
  ModelBranches& model_branches = data_table.model_branches;
  CHECK_LT(branch_idx, model_branches.size());
  ModelBranch& model_branch = model_branches[branch_idx];
  val_t *gradient = reinterpret_cast<val_t *>(update_rows);
  CHECK_EQ(model_branch.store.size(), batch_size);
  CHECK_EQ(model_branch.history_store.size(), batch_size * 2);
  val_t *master_data = reinterpret_cast<val_t *>(model_branch.store.data());
  val_t *gradient_square_history = reinterpret_cast<val_t *>(
      &model_branch.history_store.data()[0]);
  val_t *smoothed_gradient = reinterpret_cast<val_t *>(
      &model_branch.history_store.data()[batch_size * 1]);
  size_t num_vals = batch_size * ROW_DATA_SIZE;
  /* Caffe's default setting */
  val_t delta = 1e-8;
  val_t rms_decay = 0.99;
  // /* Setting used in the Inception-v3 paper */
  // val_t delta = 1;
  // val_t rms_decay = 0.9;

  val_t momentum = tunable.momentum;
  val_t neg_lr = tunable.lr * -1;

  for (uint i = 0; i < num_vals; i++) {
    smoothed_gradient[i] =
        smoothed_gradient[i] * momentum + gradient[i] * (1 - momentum);
    gradient_square_history[i] =
        gradient_square_history[i] * rms_decay
        + smoothed_gradient[i] * smoothed_gradient[i] * (1 - rms_decay);
    val_t eta = neg_lr / sqrt(gradient_square_history[i] + delta);
    val_t adjusted_update = eta * smoothed_gradient[i];
    master_data[i] += adjusted_update;
  }
}
