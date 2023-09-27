/*
 * Copyright (c) 2021-2022 Arm Limited.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef COMMAND_STREAM_HPP
#define COMMAND_STREAM_HPP

/****************************************************************************
 * Includes
 ****************************************************************************/

#include <array>
#include <ethosu_driver.h>
#include <pmu_ethosu.h>
#include <stddef.h>

/****************************************************************************
 * Defines
 ****************************************************************************/

#ifndef ETHOSU_BASEP_INDEXES
#define ETHOSU_BASEP_INDEXES 8
#endif

/****************************************************************************
 * Types
 ****************************************************************************/

namespace EthosU {
namespace CommandStream {

/****************************************************************************
 * DataPointer
 ****************************************************************************/

struct DataPointer {
    DataPointer();
    DataPointer(const char *_data, size_t _size);

    bool operator!=(const DataPointer &other);

    const char *data;
    size_t size;
};

/****************************************************************************
 * Pmu
 ****************************************************************************/

using PmuEvents = std::array<ethosu_pmu_event_type, ETHOSU_PMU_NCOUNTERS>;

class Pmu {
public:
    Pmu(ethosu_driver *_drv, const PmuEvents &_config = {});

    void clear();
    void print();

    uint64_t getCycleCount() const;
    uint32_t getEventCount(size_t index) const;

private:
    ethosu_driver *drv;
    PmuEvents config;
};

/****************************************************************************
 * CommandStream
 ****************************************************************************/

using BasePointers = std::array<DataPointer, ETHOSU_BASEP_INDEXES>;

class CommandStream {
public:
    CommandStream(const DataPointer &_commandStream,
                  const BasePointers &_pointers = {},
                  const PmuEvents &_pmuEvents   = {});
    virtual ~CommandStream();

    int run(size_t repeat = 1);
    int run_async();
    int wait_async(bool block = true);

    DataPointer &getCommandStream();
    BasePointers &getBasePointers();
    Pmu &getPmu();

private:
    ethosu_driver *drv;
    DataPointer commandStream;
    BasePointers basePointers;
    Pmu pmu;
};

#define DRIVER_ACTION_MAGIC() 'C', 'O', 'P', '1',

#define DRIVER_ACTION_COMMAND_STREAM(length) 0x02, (length >> 16) & 0xff, length & 0xff, (length >> 8) & 0xff,

#define DRIVER_ACTION_NOP() 0x05, 0x00, 0x00, 0x00,

#define NPU_OP_STOP(mask) (mask >> 8) && 0xff, mask & 0xff, 0x08, 0x00,

}; // namespace CommandStream
}; // namespace EthosU

#endif /* COMMAND_STREAM_HPP */
