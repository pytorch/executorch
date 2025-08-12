/*
 * Copyright 2022-2024 NXP
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Definition of the NXP Neutron NPU driver errors.
 */

#ifndef NEUTRON_ERRORS_H
#define NEUTRON_ERRORS_H

#include <stdint.h>

typedef int32_t NeutronError;

/*
    Generate error code.
    A code is composed of (from least to most significant bit):
        3 bits = component id
        5 bits = category id
        23 bits = code
        1 bit = sign
*/
#define GEN_NEUTRON_ERROR(component, category, code)                   \
  ((NeutronError)(((component & 0xF) << 0) | ((category & 0xF) << 3) | \
                  ((code & 0x7FFFFF) << 8)))

#define ENONE 0

#define GET_ERROR_COMPONENT(e) ((e >> 0) & 0x00000007)
#define GET_ERROR_CATEGORY(e) ((e >> 3) & 0x0000001F)
#define GET_ERROR_CODE(e) ((e >> 8) & 0x007FFFFF)

/* Components ids*/
// DO NOT USE 0x0 as component magic number!
typedef enum ERROR_COMPONENT_ID {
  ERROR_COMPONENT_LIBRARY = 0x1,
  ERROR_COMPONENT_FIRMWARE = 0x2,
  ERROR_COMPONENT_DRIVER = 0x3
} ERROR_COMPONENT_ID;

/// Retrieve component name as string from NeutronError code.
char* getNeutronErrorComponent(NeutronError ne);

/// Retrieve catefory as string from NeutronError code.
char* getNeutronErrorCategory(NeutronError ne);

#endif // NEUTRON_ERRORS_H
