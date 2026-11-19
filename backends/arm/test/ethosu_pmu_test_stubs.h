/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ethosu_driver.h>
#include <pmu_ethosu.h>

#include <atomic>

struct MockPmu {
  ethosu_pmu_event_type events[8]{};
  uint32_t counts[8]{};
  uint64_t cycles = 0;
  uint32_t overflow = 0;
  uint32_t enabled_mask = 0;
  bool enabled = false;
};

MockPmu& pmu(ethosu_driver* driver);
extern std::atomic<uint32_t> observer_begins;
extern std::atomic<uint32_t> observer_ends;
