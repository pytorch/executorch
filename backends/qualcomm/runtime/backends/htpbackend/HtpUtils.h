/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

namespace torch {
namespace executor {
namespace qnn {

// constexpr config values
constexpr const int kSleepMinLatency = 40;
constexpr const int kSleepLowLatency = 100;
constexpr const int kSleepMediumLatency = 1000;
constexpr const int kSleepHighLatency = 2000;
constexpr const int kDcvsDisable = 0;
constexpr const int kDcvsEnable = 1;

// default rpc control latency - 100 us
constexpr const int kRpcControlLatency = 100;
// default rpc polling time for high power modes - 9999 us
constexpr const int kRpcPollingTimeHighPower = 9999;
// default rpc polling time for low power modes - 0 us
constexpr const int kRpcPollingTimeLowPower = 0;
} // namespace qnn
} // namespace executor
} // namespace torch
