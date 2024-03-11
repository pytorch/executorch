#pragma once

#include <cpuinfo.h>

namespace torch {
namespace executorch {
namespace cpuinfo {

uint32_t get_num_performant_cores();

} // namespace cpuinfo
} // namespace executorch
} // namespace torch
