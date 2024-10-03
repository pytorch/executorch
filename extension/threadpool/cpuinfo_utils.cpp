/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/threadpool/cpuinfo_utils.h>

#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include <executorch/runtime/platform/assert.h>

namespace executorch::extension::cpuinfo {

// Ignore revisions (last digit (4 LSBs))
#define CPUINFO_ARM_MIDR_CORTEX_A520 UINT32_C(0x410FD800)
#define CPUINFO_ARM_MIDR_CORTEX_A53 UINT32_C(0x410FD030)
#define CPUINFO_ARM_MIDR_CORTEX_A55 UINT32_C(0x410FD050)
#define CPUINFO_ARM_MIDR_CORTEX_A57 UINT32_C(0x410FD070)

#define RIVISION_MASK UINT32_C(0xFFFFFFF0)

namespace {
bool is_non_performant_core(const struct cpuinfo_uarch_info* uarch_info) {
  switch (uarch_info->uarch) {
    case cpuinfo_uarch_cortex_a55:
    case cpuinfo_uarch_cortex_a53:
    case cpuinfo_uarch_cortex_a510:
    case cpuinfo_uarch_icestorm:
      return true;
    // This can be so many other cores.
    // Need to update this to better account for slow cores
    // Also does not account Apple's A/M series cores
    // And not yet qcomm's
    default:
      break;
  }
    // A520 is not yet updated in cpuinfo
    // Hence decode it separately.
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  if ((uarch_info->midr & RIVISION_MASK) == CPUINFO_ARM_MIDR_CORTEX_A520) {
    return true;
  }
#endif
  return false;
}

std::vector<uint32_t>* get_static_cpu_midr_vector() {
  static std::vector<uint32_t> cpu_midrs;
  return &cpu_midrs;
}

uint32_t _get_model_specific_num_cores() {
  // Not sure how reliable this is but going with it for now.
  const std::string kImageVersionPath = "/sys/devices/soc0/image_version";
  ET_LOG(Info, "Reading file %s", kImageVersionPath.c_str());
  std::fstream image_version_file(kImageVersionPath, std::ios_base::in);
  if (image_version_file.is_open()) {
    std::string x;
    std::getline(image_version_file, x);
    // Hardcoding some rules for now
    if (x.find("S911") != std::string::npos) {
      // Samsung S23 has:
      // 1x3.36 GHz Cortex-X3
      // 2x2.8 GHz Cortex-A715
      // 2x2.8 GHz Cortex-A710
      // 3x2.0 GHz Cortex-A510
      // And we have balanced execution with 4 cores.
      return 4;
    }
  }
  ET_LOG(Info, "Failed to open midr file %s", kImageVersionPath.c_str());
  return 0;
}

bool populate_available_cpu_mids() {
  std::vector<uint32_t>* cpu_midrs = get_static_cpu_midr_vector();
  uint32_t num_possible_cores = cpuinfo_get_processors_count();
  cpu_midrs->resize(num_possible_cores);
  const std::string kMidrFilePathPrefix = "/sys/devices/system/cpu/cpu";
  const std::string kMidrFilePathSuffix = "/regs/identification/midr_el1";
  for (int32_t i = 0; i < num_possible_cores; ++i) {
    std::string midr_file_path =
        kMidrFilePathPrefix + std::to_string(i) + kMidrFilePathSuffix;
    ET_LOG(Info, "Reading file %s", midr_file_path.c_str());
    std::fstream midr_file(midr_file_path, std::ios_base::in);
    uint32_t tmp{0};
    if (midr_file.is_open()) {
      std::string x;
      std::getline(midr_file, x);
      tmp = std::stoi(x, nullptr, 16);
      (*cpu_midrs)[i] = tmp;
    } else {
      ET_LOG(Info, "Failed to open midr file %s", midr_file_path.c_str());
      cpu_midrs->clear();
      return false;
    }
  }
  return true;
}

uint32_t _get_num_performant_cores() {
  // @lint-ignore CLANGTIDY facebook-hte-std::once_flag
  static std::once_flag flag;
  // @lint-ignore CLANGTIDY facebook-hte-std::call_once
  std::call_once(flag, []() { populate_available_cpu_mids(); });
  std::vector<uint32_t>* cpu_midrs = get_static_cpu_midr_vector();
  uint32_t num_possible_cores = cpuinfo_get_processors_count();
  if (num_possible_cores != cpu_midrs->size()) {
    ET_LOG(Info, "CPU info and manual query on # of cpus dont match.");
    return 0;
  }
  for (int32_t i = 0; i < cpu_midrs->size(); ++i) {
    uint32_t masked_midr = (*cpu_midrs)[i] & RIVISION_MASK;
    switch (masked_midr) {
      case CPUINFO_ARM_MIDR_CORTEX_A520:
      case CPUINFO_ARM_MIDR_CORTEX_A53:
      case CPUINFO_ARM_MIDR_CORTEX_A55:
      case CPUINFO_ARM_MIDR_CORTEX_A57:
        num_possible_cores--;
        break;
      default:
        break;
    }
  }
  return num_possible_cores;
}

} // namespace

uint32_t get_num_performant_cores() {
  ET_CHECK_MSG(cpuinfo_initialize(), "cpuinfo cannot be initialized.");
  // First try and see if we have number of cores profiled for this specific
  // device
  uint32_t model_specific_num_cores = _get_model_specific_num_cores();
  if (model_specific_num_cores > 0) {
    return model_specific_num_cores;
  }

  // Else looks at either the # of litte cores if found
  // Or parse the midr in "Something seems wrong" section.
  const uint32_t uarch_count = cpuinfo_get_uarchs_count();
  uint32_t num_possible_cores = cpuinfo_get_processors_count();
  uint32_t num_non_performant_core = 0;
  if (uarch_count > 1) {
    for (int32_t i = 0; i < uarch_count; ++i) {
      const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
      if (is_non_performant_core(uarch_info)) {
        num_non_performant_core += uarch_info->processor_count;
      }
    }
    ET_LOG(Info, "Number of efficient cores %d", num_non_performant_core);
    if (num_possible_cores <= num_non_performant_core) {
      ET_LOG(
          Info, "Total number of cores must be larger than efficient cores.");
      return 0;
    }
    return (num_possible_cores - num_non_performant_core);
  } else {
    // Something seems wrong. Lets check each processor's midr
    // In one plua 12 while it has 2 little cores, the topology
    // reported in /sys/devices/system/cpu/cpu* /topology/core_siblings_list
    // report wrong topology which results in wront configratuon
    return _get_num_performant_cores();
  }
}

} // namespace executorch::extension::cpuinfo
