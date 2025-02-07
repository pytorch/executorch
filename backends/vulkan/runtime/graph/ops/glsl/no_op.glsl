/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

${define_required_extensions(DTYPE)}

#include "broadcasting_utils.h"
#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(0, "r", "t_out", DTYPE, STORAGE)}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {}
