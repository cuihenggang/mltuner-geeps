#ifndef __mltuner_entry_hpp__
#define __mltuner_entry_hpp__

#include <boost/shared_ptr.hpp>

#include "common/router-handler.hpp"

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using boost::shared_ptr;

class MltunerEntry {
  uint num_processes;
  boost::shared_ptr<zmq::context_t> zmq_ctx;
  Config config;

 public:
  MltunerEntry(
      uint num_processes,
      boost::shared_ptr<zmq::context_t> zmq_ctx,
      const Config& config) :
    num_processes(num_processes),
    zmq_ctx(zmq_ctx),
    config(config) {
  }

  void operator()() {
    mltuner_entry(num_processes, zmq_ctx, config);
  }

 private:
  void mltuner_entry(
      uint num_processes,
      shared_ptr<zmq::context_t> zmq_ctx,
      const Config& config);
};

#endif  // defined __mltuner_entry_hpp__
