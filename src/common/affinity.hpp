#ifndef __AFFINITY_HPP__
#define __AFFINITY_HPP__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

// Helper functions for setting affinities

#include <iostream>

#include <glog/logging.h>

#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <sys/syscall.h>
#define gettid() syscall(__NR_gettid)
#include <numa.h>
#include <numaif.h>

using std::cout;
using std::cerr;
using std::endl;

inline void set_cpu_affinity(int node_id, int num_cores, int num_zones) {
  int rc = 0;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  int node_size = num_cores / num_zones;
  for (int i = node_id * node_size; i < (node_id + 1) * node_size; i++) {
    CPU_SET(i, &mask);
  }
  rc = sched_setaffinity(gettid(), sizeof(mask), &mask);
  assert(rc == 0);
}

inline void *set_mem_affinity(int node_id) {
  bitmask *old_mask = numa_get_membind();
  if (old_mask == NULL) {
    cerr << "*** warning: getting NULL return from numa_get_membind()\n";
  }
  bitmask *mask = numa_allocate_nodemask();
  CHECK(mask);
  mask = numa_bitmask_setbit(mask, node_id);
  numa_set_bind_policy(1); /* set NUMA zone binding to be "strict" */
  numa_set_membind(mask);
  numa_free_nodemask(mask);
  return reinterpret_cast<void *>(old_mask);
}

inline void restore_mem_affinity(void *mask_opaque) {
  if (mask_opaque == NULL) {
    return;
  }
  bitmask *mask = reinterpret_cast<bitmask *>(mask_opaque);
  numa_set_bind_policy(0); /* set NUMA zone binding to be "preferred" */
  numa_set_membind(mask);
}

inline int get_numa_node_id(int entity_id, int num_entity, int num_nodes) {
  int entity_per_node;
  if (num_entity < num_nodes) {
    entity_per_node = 1;
  } else {
    assert(num_entity % num_nodes == 0);
    entity_per_node = num_entity / num_nodes;
  }
  int node_id = entity_id / entity_per_node;
  return node_id;
}

inline void set_affinity(int node_id, int num_cores, int num_nodes) {
  set_cpu_affinity(node_id, num_cores, num_nodes);
  set_mem_affinity(node_id);
}

#endif  // __AFFINITY_HPP__