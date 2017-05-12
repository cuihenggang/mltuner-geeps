#ifndef __common_utils_hpp__
#define __common_utils_hpp__

#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include <iostream>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <list>

#include <glog/logging.h>

#include "geeps.hpp"
#include "geeps-user-defined-types.hpp"
#include "common/internal-config.hpp"

using std::string;
using std::vector;
using std::pair;
using std::cout;
using std::cerr;
using std::endl;

using boost::unordered_map;
using boost::shared_ptr;
using boost::make_shared;

typedef unsigned int uint;

typedef std::vector<double> DoubleVec;

typedef std::vector<int> VecClock;

typedef boost::unique_lock<boost::mutex> ScopedLock;

inline int clock_min(const VecClock& clocks) {
  CHECK(clocks.size());
  int cmin = clocks[0];
  for (uint i = 1; i < clocks.size(); i++) {
    cmin = clocks[i] < cmin ? clocks[i] : cmin;
  }
  return cmin;
}

inline int clock_max(const VecClock& clocks) {
  CHECK(clocks.size());
  int cmax = clocks[0];
  for (uint i = 1; i < clocks.size(); i++) {
    cmax = clocks[i] > cmax ? clocks[i] : cmax;
  }
  return cmax;
}

inline uint get_nearest_power2(uint n) {
  uint power2 = 1;
  while (power2 < n) {
    power2 <<= 1;
  }
  return power2;
}

#endif  // defined __common_utils_hpp__
