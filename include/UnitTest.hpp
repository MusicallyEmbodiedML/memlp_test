#ifndef __UNIT_TEST_HPP__
#define __UNIT_TEST_HPP__


#include <vector>
#include "microunit.h"
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP


using num_t = float;
using iris_num_t = double;
using d_vector = std::vector<num_t>;
using nd_vector = std::vector< std::vector<num_t> >;

#define ASSERT_EQ(a, b)    ASSERT_TRUE(a == b)

#endif  // __UNIT_TEST_HPP__