/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define ACCUM_T ${accum_scalar_type(DTYPE)}
#define T ${texel_load_component_type(DTYPE, "buffer")}

#define NUM_OUTPUTS_PER_WG 1
#define NUM_WORKERS_PER_OUTPUT 64

${define_active_storage_type("buffer")}
${define_required_extensions(DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "indexing.glslh"
#include "convert.glslh"
#include "reduce_op_defs.glslh"

$if OUTPUT_IS_INDICES:
  ${layout_declare_tensor(B, "w", "t_out", "int", "buffer")}
$else:
  ${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}

${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Shared memory for cooperative reduction
shared Accum shared_values[NUM_OUTPUTS_PER_WG][NUM_WORKERS_PER_OUTPUT];

#define init_fn ${INIT_ACCUM_FN}
#define update_fn ${UPDATE_ACCUM_FN}
#define merge_fn ${MERGE_ACCUM_FN}

$if POSTPROCESS_ACCUM_FN != "none":
  #define postprocess_fn ${POSTPROCESS_ACCUM_FN}

$if OOB_INIT_MODE == "zero":
  #define OOB_INIT_MODE 0
$else:
  #define OOB_INIT_MODE 1

$if OUTPUT_IS_INDICES:
  #define OUTPUT_IS_INDICES

void main() {
  const uint out_bufi = gl_GlobalInvocationID.y;

  if (out_of_bounds(out_bufi, outp)) {
    return;
  }

  // Local indices
  const uint worker_id = gl_LocalInvocationID.x;
  const uint output_id = gl_LocalInvocationID.y;

  const uint in_bufi_base = out_bufi * width(inp);

  Accum local_accum;
  // Initialize accumulator with the first element being processed
  if (worker_id < width(inp)) {
    const uint in_bufi = in_bufi_base + worker_id;
    init_fn(local_accum, t_in[in_bufi], worker_id);
  }
  // For out of bounds case, initialization depends on reduction op
  else {
#if OOB_INIT_MODE == 0
    // Init with a zero value
    init_accum_zero(local_accum);
#else
    // Init with the first value (i.e. amin, amax)
    init_fn(local_accum, t_in[in_bufi_base], 0);
#endif
  }

  for (uint x = worker_id + NUM_WORKERS_PER_OUTPUT; x < width(inp);
       x += NUM_WORKERS_PER_OUTPUT) {
    update_fn(local_accum, t_in[in_bufi_base + x], x);
  }

  shared_values[output_id][worker_id] = local_accum;

  memoryBarrierShared();
  barrier();

  for (int i = NUM_WORKERS_PER_OUTPUT / 2; i > 0; i >>= 1) {
    if (worker_id < i) {
      merge_fn(
        shared_values[output_id][worker_id],
        shared_values[output_id][worker_id + i]);
    }
    memoryBarrierShared();
    barrier();
  }

  if (worker_id == 0) {
    local_accum = shared_values[output_id][0];
#ifdef postprocess_fn
    postprocess_fn(local_accum);
#endif

#ifdef OUTPUT_IS_INDICES
    t_out[out_bufi] = int(0); // int(local_accum.idx);
#else
    t_out[out_bufi] = convert_to_T(local_accum.val);
#endif
  }
}
