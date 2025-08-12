/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define IN_T ${buffer_scalar_type(IN_DTYPE)}
#define FVEC4_T ${texel_load_type(IN_DTYPE, "texture3d")}
#define SCALE_OUT_T ${buffer_scalar_type(SCALE_OUT_DTYPE)}
#define ZP_OUT_T ${buffer_scalar_type(ZP_OUT_DTYPE)}

#define ${MODE}

${define_active_storage_type("texture3d")}
${define_required_extensions(IN_DTYPE)}
${define_required_extensions(SCALE_OUT_DTYPE)}
${define_required_extensions(ZP_OUT_DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

$if MODE != "block_wise":
  ${layout_declare_tensor(B, "w", "t_scale", SCALE_OUT_DTYPE, "texture3d")}
  ${layout_declare_tensor(B, "w", "t_zero_point", ZP_OUT_DTYPE, "texture3d")}
$else:
  ${layout_declare_tensor(B, "w", "t_scale", SCALE_OUT_DTYPE, "buffer")}
  ${layout_declare_tensor(B, "w", "t_zero_point", ZP_OUT_DTYPE, "buffer")}

${layout_declare_tensor(B, "r", "t_in", IN_DTYPE, "texture3d")}

$if MODE == "per_tensor":
  layout(push_constant) uniform restrict Block {
    int quant_min;
    int quant_max;
    float eps;
  };
$if MODE == "per_token":
  layout(push_constant) uniform restrict Block {
    int num_tokens;
    int quant_min;
    int quant_max;
  };
$if MODE == "block_wise":
  layout(push_constant) uniform BlockPC {
    ivec4 blockSize; // WHCN (>=1)
    ivec4 numBlocks; // #blocks along W,H,C,N
    ivec4 blockStride; // {1, #W, #W * #H, #W * #H * #C}
    int mapping_type; // 0=ASYM, 1=SYM, 2=SYM_NO_CLIP
    int quant_min;
    int quant_max;
    float eps;
  };

${layout_declare_ubo(B, "ivec3", "t_in_limits")}
$if MODE != "block_wise":
  ${layout_declare_ubo(B, "ivec3", "t_scale_limits")}
  ${layout_declare_ubo(B, "ivec3", "t_zero_point_limits")}
$else:
  ${layout_declare_ubo(B, "ivec4", "t_scale_sizes")}
  ${layout_declare_ubo(B, "ivec4", "t_scale_strides")}
  ${layout_declare_ubo(B, "ivec4", "t_zero_point_sizes")}
  ${layout_declare_ubo(B, "ivec4", "t_zero_point_strides")}


#include "indexing_utils.h"
#include "choose_qparams.glslh"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define NWORKERS 64

// Shared memory for reduction - must match local work group size
shared float shared_min[NWORKERS];
shared float shared_max[NWORKERS];

/*/*
  Quantization Parameter Computation Shader (Buffer Storage)
    This shader computes quantization parameters (scale and zero_point) for converting
    floating-point tensors to n-bit integer representations while preserving the
    original data range as much as possible. The computed parameters enable efficient
    quantization by mapping the continuous floating-point range to discrete integer values.

  Important Considerations:
    (+) The input tensor is assumed to be WIDTH_PACKED (i.e., contiguous in the last dimension)

  Workgroup Configuration:
  - choose_qparams_per_tensor
      This mode computes a single set of quantization parameters for the entire tensor.
      Uses parallel reduction across all threads to find global min/max values.

    (*) global_wg_size: default
    (*) local_wg_size: default

  - choose_qparams_per_token
      This mode computes separate quantization parameters for each token in the tensor.
      Each workgroup processes one token independently to find token-specific min/max.

    (*) global_wg_size: default
    (*) local_wg_size: {1, 1, 1}

  - choose_qparams_block_wise
      This mode computes quantization parameters for each block of elements, allowing
      fine-grained control over quantization granularity within the tensor. Each block
      is processed independently to find its own min/max values and compute corresponding
      scale and zero_point parameters.

      NOTE: This mode currently only supports buffer storage for the output.

    (*) global_wg_size: {nBlocks, 1u, 1u} (one workgroup per block)
    (*) local_wg_size: {1, 1, 1} (single thread per block)

  Tree Reduction Algorithm for Min/Max Finding:
    The shader uses a parallel tree reduction algorithm to efficiently find minimum and
    maximum values across multiple threads. This approach reduces the number of memory
    accesses and synchronization points compared to sequential scanning.

    Example with 8 threads processing values [10, 1, 8, 1, 0, 2, 3, 5]:

    Step 1 - Initial Population:
    Each thread loads its assigned value into shared memory arrays.
    shared_min:  | 10 | 1 | 8 | 1 | 0 | 2 | 3 | 5 |
    shared_max:  | 10 | 1 | 8 | 1 | 0 | 2 | 3 | 5 |
    Thread ID:   |  0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

    Step 2 - Stride 1 (Compare Adjacent Pairs):
    Threads 0,2,4,6 compare with threads 1,3,5,7 respectively.
    shared_min:  |  1 |   | 1 |   | 0 |   | 3 |   |  (min(10,1), min(8,1), min(0,2), min(3,5))
    shared_max:  | 10 |   | 8 |   | 2 |   | 5 |   |  (max(10,1), max(8,1), max(0,2), max(3,5))
    Active:      |  0 |   | 2 |   | 4 |   | 6 |   |

    Step 3 - Stride 2 (Compare Pairs of Pairs):
    Threads 0,4 compare with threads 2,6 respectively.
    shared_min:  |  1 |   |   |   | 0 |   |   |   |  (min(1,1), min(0,3))
    shared_max:  | 10 |   |   |   | 5 |   |   |   |  (max(10,8), max(2,5))
    Active:      |  0 |   |   |   | 4 |   |   |   |

    Step 4 - Stride 4 (Final Comparison):
    Thread 0 compares with thread 4 to get final result.
    shared_min:  |  0 |   |   |   |   |   |   |   |  (min(1,0) = 0)
    shared_max:  | 10 |   |   |   |   |   |   |   |  (max(10,5) = 10)
    Active:      |  0 |   |   |   |   |   |   |   |

    Final Result: global_min = 0, global_max = 10 (stored in shared_min[0], shared_max[0])

    The tree reduction completes in log_2(N) steps where N is the number of threads,
    providing O(log N) time complexity instead of O(N) for sequential reduction.

  Quantization Parameter Calculation:
    Once min/max values are determined, the shader computes:
    - scale = (max - min) / (quant_max - quant_min)
    - zero_point = quantization offset to map floating-point zero to integer range

  Mode-Specific Behavior:
  - Per-Tensor: Single workgroup with strided access across entire tensor
  - Per-Token: Multiple workgroups, each processing one token independently
*/

#ifdef per_tensor

void choose_qparams_per_tensor() {
  uint global_id = gl_GlobalInvocationID.x;
  uint local_id = gl_LocalInvocationID.x;
  uint group_id = gl_WorkGroupID.x;
  uint total_threads = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

  uint total_texels = uint(t_in_limits.x * t_in_limits.y * t_in_limits.z);

  // Each thread processes multiple texels with stride
  float thread_min = 1.0/0.0;  // +infinity
  float thread_max = -1.0/0.0; // -infinity
  bool found_valid = false;

  // Process texels with stride across all threads
  for (uint texel_idx = global_id; texel_idx < total_texels; texel_idx += total_threads) {
    // Convert linear texel index to 3D coordinates
    uint z = texel_idx / uint(t_in_limits.x * t_in_limits.y);
    uint remainder = texel_idx % uint(t_in_limits.x * t_in_limits.y);
    uint y = remainder / uint(t_in_limits.x);
    uint x = remainder % uint(t_in_limits.x);
    ivec3 texel_pos = ivec3(int(x), int(y), int(z));

    FVEC4_T texel_data = load_texel(t_in, texel_pos);

    // For texture storage, we assume width-packed (packed_dim = 0)
    // Calculate number of valid elements in this texel (handle padding)
    int packed_dim = 0; // Width dimension is packed
    ivec4 sizes = ivec4(t_in_limits, 1); // Convert limits to sizes format
    ivec4 tensor_coord = to_tensor_idx(texel_pos, sizes, packed_dim);

    // Calculate total tensor elements to determine padding
    int total_elements = t_in_limits.x * t_in_limits.y * t_in_limits.z * 4;
    int linear_tensor_idx = tensor_coord.x + tensor_coord.y * sizes.x +
                            tensor_coord.z * sizes.x * sizes.y;
    int remaining_elements = total_elements - (linear_tensor_idx);
    int valid_elements = min(4, remaining_elements);

    // Find min/max within this texel, considering only valid elements
    if (valid_elements >= 1 && !isnan(texel_data.x) && !isinf(texel_data.x)) {
      if (!found_valid) {
        thread_min = texel_data.x;
        thread_max = texel_data.x;
        found_valid = true;
      } else {
        thread_min = min(thread_min, texel_data.x);
        thread_max = max(thread_max, texel_data.x);
      }
    }

    if (valid_elements >= 2 && !isnan(texel_data.y) && !isinf(texel_data.y)) {
      if (!found_valid) {
        thread_min = texel_data.y;
        thread_max = texel_data.y;
        found_valid = true;
      } else {
        thread_min = min(thread_min, texel_data.y);
        thread_max = max(thread_max, texel_data.y);
      }
    }

    if (valid_elements >= 3 && !isnan(texel_data.z) && !isinf(texel_data.z)) {
      if (!found_valid) {
        thread_min = texel_data.z;
        thread_max = texel_data.z;
        found_valid = true;
      } else {
        thread_min = min(thread_min, texel_data.z);
        thread_max = max(thread_max, texel_data.z);
      }
    }

    if (valid_elements >= 4 && !isnan(texel_data.w) && !isinf(texel_data.w)) {
      if (!found_valid) {
        thread_min = texel_data.w;
        thread_max = texel_data.w;
        found_valid = true;
      } else {
        thread_min = min(thread_min, texel_data.w);
        thread_max = max(thread_max, texel_data.w);
      }
    }
  }

  // Intra-workgroup reduction using shared memory
  shared_min[local_id] = thread_min;
  shared_max[local_id] = thread_max;
  barrier();

  // Tree reduction within work group
  for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      float other_min = shared_min[local_id + stride];
      float other_max = shared_max[local_id + stride];

      if (!isinf(other_min) && (isinf(shared_min[local_id]) || other_min < shared_min[local_id])) {
        shared_min[local_id] = other_min;
      }
      if (!isinf(other_max) && (isinf(shared_max[local_id]) || other_max > shared_max[local_id])) {
        shared_max[local_id] = other_max;
      }
    }
    barrier();
  }

  // Final result calculation (single workgroup only for reliability)
  if (local_id == 0 && group_id == 0) {
    float global_min = shared_min[0];
    float global_max = shared_max[0];

    float scale_val;
    int zero_point_val;
    calc_scale_zp(global_min, global_max, quant_min, quant_max, 0, eps, scale_val, zero_point_val);

    write_texel(t_scale, ivec3(0, 0, 0), vec4(SCALE_OUT_T(scale_val), 0.0, 0.0, 0.0));
    write_texel(t_zero_point, ivec3(0, 0, 0), ivec4(ZP_OUT_T(zero_point_val), 0, 0, 0));
  }
}

#elif defined(per_token)

void choose_qparams_per_token() {
  // Each token is processed by multiple workgroups for parallel reduction
  uint local_id = gl_LocalInvocationID.x;
  uint group_id = gl_WorkGroupID.x;
  uint total_workgroups = gl_NumWorkGroups.x;

  uint total_texels = uint(t_in_limits.x * t_in_limits.y * t_in_limits.z);

  // Calculate texels per token (assuming last dimension contains the token data)
  // For per-token quantization, we assume tokens are along the last dimension
  uint texels_per_token = total_texels / uint(num_tokens);

  // Calculate how many tokens each workgroup should process
  uint tokens_per_workgroup = (uint(num_tokens) + total_workgroups - 1) / total_workgroups;

  // Calculate which tokens this workgroup is responsible for
  uint start_token = group_id * tokens_per_workgroup;
  uint end_token = min(start_token + tokens_per_workgroup, uint(num_tokens));

  // Process each token assigned to this workgroup
  for (uint token_id = start_token; token_id < end_token; token_id++) {
    // Calculate the texel range for this token
    uint token_start_texel = token_id * texels_per_token;
    uint token_end_texel = token_start_texel + texels_per_token;

    // Each thread processes multiple texels within the token
    float thread_min = 1.0/0.0;  // +infinity
    float thread_max = -1.0/0.0; // -infinity
    bool found_valid = false;

    // Process texels within this token only
    for (uint texel_idx = token_start_texel + local_id; texel_idx < token_end_texel; texel_idx += gl_WorkGroupSize.x) {
      // Convert linear texel index to 3D coordinates
      uint z = texel_idx / uint(t_in_limits.x * t_in_limits.y);
      uint remainder = texel_idx % uint(t_in_limits.x * t_in_limits.y);
      uint y = remainder / uint(t_in_limits.x);
      uint x = remainder % uint(t_in_limits.x);
      ivec3 texel_pos = ivec3(int(x), int(y), int(z));

      FVEC4_T texel_data = load_texel(t_in, texel_pos);

      // For texture storage, we assume width-packed (packed_dim = 0)
      // Calculate number of valid elements in this texel (handle padding)
      int packed_dim = 0; // Width dimension is packed
      ivec4 sizes = ivec4(t_in_limits, 1); // Convert limits to sizes format
      ivec4 tensor_coord = to_tensor_idx(texel_pos, sizes, packed_dim);

      // Calculate total tensor elements to determine padding
      int total_elements = t_in_limits.x * t_in_limits.y * t_in_limits.z * 4;
      int linear_tensor_idx = tensor_coord.x + tensor_coord.y * sizes.x +
                              tensor_coord.z * sizes.x * sizes.y;
      int remaining_elements = total_elements - (linear_tensor_idx);
      int valid_elements = min(4, remaining_elements);

      // Find min/max within this texel, considering only valid elements
      if (valid_elements >= 1 && !isnan(texel_data.x) && !isinf(texel_data.x)) {
        if (!found_valid) {
          thread_min = texel_data.x;
          thread_max = texel_data.x;
          found_valid = true;
        } else {
          thread_min = min(thread_min, texel_data.x);
          thread_max = max(thread_max, texel_data.x);
        }
      }

      if (valid_elements >= 2 && !isnan(texel_data.y) && !isinf(texel_data.y)) {
        if (!found_valid) {
          thread_min = texel_data.y;
          thread_max = texel_data.y;
          found_valid = true;
        } else {
          thread_min = min(thread_min, texel_data.y);
          thread_max = max(thread_max, texel_data.y);
        }
      }

      if (valid_elements >= 3 && !isnan(texel_data.z) && !isinf(texel_data.z)) {
        if (!found_valid) {
          thread_min = texel_data.z;
          thread_max = texel_data.z;
          found_valid = true;
        } else {
          thread_min = min(thread_min, texel_data.z);
          thread_max = max(thread_max, texel_data.z);
        }
      }

      if (valid_elements >= 4 && !isnan(texel_data.w) && !isinf(texel_data.w)) {
        if (!found_valid) {
          thread_min = texel_data.w;
          thread_max = texel_data.w;
          found_valid = true;
        } else {
          thread_min = min(thread_min, texel_data.w);
          thread_max = max(thread_max, texel_data.w);
        }
      }
    }

    // Intra-workgroup reduction using shared memory
    shared_min[local_id] = thread_min;
    shared_max[local_id] = thread_max;
    barrier();

    // Tree reduction within work group
    for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
      if (local_id < stride) {
        float other_min = shared_min[local_id + stride];
        float other_max = shared_max[local_id + stride];

        // Handle infinity values properly
        if (!isinf(other_min) && (isinf(shared_min[local_id]) || other_min < shared_min[local_id])) {
          shared_min[local_id] = other_min;
        }
        if (!isinf(other_max) && (isinf(shared_max[local_id]) || other_max > shared_max[local_id])) {
          shared_max[local_id] = other_max;
        }
      }
      barrier();
    }

    // Final calculation for this token
    if (local_id == 0) {
      float token_min = shared_min[0];
      float token_max = shared_max[0];

      float scale_val;
      int zero_point_val;
      calc_scale_zp(token_min, token_max, quant_min, quant_max, 0, 1e-5, scale_val, zero_point_val);

      // Convert token_id to 3D coordinates for output texture
      // Assuming output tensors have the same layout as input but with different dimensions
      uint out_z = token_id / uint(t_scale_limits.x * t_scale_limits.y);
      uint out_remainder = token_id % uint(t_scale_limits.x * t_scale_limits.y);
      uint out_y = out_remainder / uint(t_scale_limits.x);
      uint out_x = out_remainder % uint(t_scale_limits.x);
      ivec3 out_pos = ivec3(int(out_x), int(out_y), int(out_z));

      write_texel(t_scale, out_pos, vec4(SCALE_OUT_T(scale_val), 0.0, 0.0, 0.0));
      write_texel(t_zero_point, out_pos, ivec4(ZP_OUT_T(zero_point_val), 0, 0, 0));
    }

    // Synchronize before processing next token
    barrier();
  }
}

#elif defined(block_wise)

ivec4 block_id_to_coord(uint bid) {
  ivec4 bc;
  bc.w = int(bid) / blockStride.w;

  int r = int(bid) - bc.w * blockStride.w;
  bc.z = r / blockStride.z;

  r -= bc.z * blockStride.z;
  bc.y = r / blockStride.y;

  r -= bc.y * blockStride.y;
  bc.x = r;
  return bc;
}

void choose_qparams_block_wise() {
  const uint T = uint(numBlocks.x * numBlocks.y * numBlocks.z * numBlocks.w);
  const uint STRIDE = gl_WorkGroupSize.x * gl_NumWorkGroups.x;

  // tensor full size in WHCN order
  const ivec4 tensorSz = blockSize * numBlocks;

  // Process blocks with stride for better parallelization
  for (uint blkIdx = gl_GlobalInvocationID.x; blkIdx < T; blkIdx += STRIDE) {
    // block index in WHCN
    const ivec4 b4d = block_id_to_coord(blkIdx);
    const ivec4 blockStart = b4d * blockSize;
    const ivec4 blockEnd = blockStart + blockSize;

    // scan all elements inside the block
    float vmin = 3.402823e38;  // +FLT_MAX
    float vmax = -3.402823e38; // -FLT_MAX
    bool found_valid = false;

    // Calculate total elements in block for linear iteration
    const int blockElements = blockSize.x * blockSize.y * blockSize.z * blockSize.w;

    // Linear iteration over block elements (more cache-friendly)
    for (int elemIdx = 0; elemIdx < blockElements; ++elemIdx) {
      // Convert linear index to 4D coordinates within block
      int remaining = elemIdx;
      int dn = remaining / (blockSize.x * blockSize.y * blockSize.z);
      remaining -= dn * (blockSize.x * blockSize.y * blockSize.z);
      int dc = remaining / (blockSize.x * blockSize.y);
      remaining -= dc * (blockSize.x * blockSize.y);
      int dh = remaining / blockSize.x;
      int dw = remaining - dh * blockSize.x;

      ivec4 tidx = blockStart + ivec4(dw, dh, dc, dn);

      // skip padding when tensor size is not an exact multiple of block
      if (any(greaterThanEqual(tidx, tensorSz))) { continue; }

      // tensor index -> (x,y,z,component) inside input texture
      ivec4 posi = to_texture_elem_pos(tidx, tensorSz, 0); // 0 = W_DIM (width packed)

      // fetch texel and pick the element inside it
      FVEC4_T texl = load_texel(t_in, posi.xyz);
      float v;
      if (posi.w == 0) v = texl.x;
      else if (posi.w == 1) v = texl.y;
      else if (posi.w == 2) v = texl.z;
      else v = texl.w;

      if (!isnan(v) && !isinf(v)) {
        if (!found_valid) {
          vmin = vmax = v;
          found_valid = true;
        } else {
          vmin = min(vmin, v);
          vmax = max(vmax, v);
        }
      }
    }

    // Handle case where no valid values were found
    if (!found_valid) {
      vmin = 0.0;
      vmax = 0.0;
    }

    // compute scale / zero‑point (same maths as buffer kernel)
    float scale;
    int zp;
    calc_scale_zp(vmin, vmax, quant_min, quant_max, mapping_type, eps, scale, zp);

    // Write the scalar values directly to buffer using linear index
    t_scale[blkIdx] = SCALE_OUT_T(scale);
    t_zero_point[blkIdx] = ZP_OUT_T(zp);
  }
}

#endif

void main() {
  choose_qparams_${MODE}();
}
