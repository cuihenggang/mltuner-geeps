#ifndef __geeps_user_defined_types_hpp__
#define __geeps_user_defined_types_hpp__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <iostream>
#include <string>
#include <vector>
#include <utility>

#include <boost/unordered_map.hpp>

#include <glog/logging.h>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;

typedef uint8_t command_t;
typedef size_t row_idx_t;
typedef size_t col_idx_t;
typedef float val_t;
typedef size_t table_id_t;
typedef int iter_t;

typedef std::pair<table_id_t, row_idx_t> TableRow;
typedef struct {
  table_id_t table;
  row_idx_t row;
} table_row_t;

typedef boost::unordered_map<col_idx_t, val_t> UMapData;
typedef std::vector<val_t> VectorData;
// #define ROW_DATA_SIZE 1000
// #define ROW_DATA_SIZE 21504
// #define ROW_DATA_SIZE 16
#define ROW_DATA_SIZE 128
// #define ROW_DATA_SIZE 512
// #define ROW_DATA_SIZE 10
struct ArrayData {
  val_t data[ROW_DATA_SIZE];
  void init() {
    for (size_t i = 0; i < ROW_DATA_SIZE; i++) {
      data[i] = 0;
    }
  }
  ArrayData() {
    init();
  }
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & data;
  }
};

typedef ArrayData RowData;
typedef ArrayData RowOpVal;

#endif  // defined __geeps_user_defined_types_hpp__
