/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <vector>
#include <string>

#include "mltuner-impl.hpp"

Mltuner::Mltuner(
    shared_ptr<Communicator> communicator,
    const Config& config) {
  MltunerImpl::CreateInstance(communicator, config);
}

void Mltuner::recv_worker_started(uint worker_id) {
  mltuner_impl->recv_worker_started(worker_id);
}

void Mltuner::recv_worker_progress(
    uint worker_id, int clock, int branch_id, double progress) {
  mltuner_impl->recv_worker_progress(worker_id, clock, branch_id, progress);
}
