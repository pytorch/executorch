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
#define SCALE_OUT_T ${buffer_scalar_type(SCALE_OUT_DTYPE)}
#define ZP_OUT_T ${buffer_scalar_type(ZP_OUT_DTYPE)}

#define ${MODE}

${define_active_storage_type("buffer")}
${define_required_extensions(IN_DTYPE)}
${define_required_extensions(SCALE_OUT_DTYPE)}
${define_required_extensions(ZP_OUT_DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_scale", SCALE_OUT_DTYPE, "buffer")}
${layout_declare_tensor(B, "w", "t_zero_point", ZP_OUT_DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_in", IN_DTYPE, "buffer")}

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

${layout_declare_ubo(B, "ivec4", "t_in_sizes")}
${layout_declare_ubo(B, "ivec4", "t_in_strides")}
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

/*
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

    (*) global_wg_size: {1, 1, 1} (single workgroup processes entire tensor)
    (*) local_wg_size: {64, 1, 1} (matches NWORKERS for shared memory)

  - choose_qparams_per_token
      This mode computes separate quantization parameters for each token in the tensor.
      Each workgroup processes one token independently to find token-specific min/max.

    (*) global_wg_size: {num_tokens, 1, 1} (one workgroup per token)
    (*) local_wg_size: {1, 1, 1} (single thread per token)

  - choose_qparams_block_wise
      This mode computes quantization parameters for each block of elements, allowing
      fine-grained control over quantization granularity within the tensor. Each block
      is processed independently to find its own min/max values and compute corresponding
      scale and zero_point parameters.

    (*) global_wg_size: {nBlocks, 1u, 1u} (one workgroup per block)
    (*) local_wg_size: {1, 1, 1} (single thread per block)

    Block-wise quantization supports multiple mapping types for scale/zero_point calculation:

    - mapping_type = 0 (ASYMMETRIC):
        Uses asymmetric quantization where the full floating-point range [min, max] is
        mapped to the quantized range [quant_min, quant_max]. This preserves the original
        data distribution but may not center zero optimally.

        Calculation:
        scale = (max - min) / (quant_max - quant_min)
        zero_point = quant_min - round(min / scale)

        Example: For range [-3.5, 10.2] mapping to int4 [-8, 7]:
        scale = (10.2 - (-3.5)) / (7 - (-8)) = 13.7 / 15 = 0.913
        zero_point = -8 - round(-3.5 / 0.913) = -8 - (-4) = -4

    - mapping_type = 1 (SYMMETRIC):
        Uses symmetric quantization where the range is centered around zero. The scale
        is computed based on the maximum absolute value, ensuring zero is exactly
        representable in the quantized domain.

        Calculation:
        max_abs = max(abs(min), abs(max))
        scale = max_abs / ((quant_max - quant_min) / 2)
        zero_point = (quant_max + quant_min + 1) / 2  // midpoint

        Example: For range [-3.5, 10.2] mapping to int4 [-8, 7]:
        max_abs = max(3.5, 10.2) = 10.2
        scale = 10.2 / ((7 - (-8)) / 2) = 10.2 / 7.5 = 1.36
        zero_point = (-8 + 7 + 1) / 2 = 0

    - mapping_type = 2 (SYMMETRIC_NO_CLIPPING_ERR):
        A variant of symmetric quantization that minimizes clipping errors by computing
        separate scales for positive and negative ranges, then using the maximum. This
        reduces quantization error on the dominant range while ensuring no values are
        clipped.

        Calculation:
        smin = abs(min) / abs(quant_min)  // scale for negative range
        smax = max / quant_max            // scale for positive range
        scale = max(smin, smax)           // use larger scale to avoid clipping
        zero_point = (quant_max + quant_min + 1) / 2  // midpoint

        Example: For range [-3.5, 10.2] mapping to int4 [-8, 7]:
        smin = 3.5 / 8 = 0.4375
        smax = 10.2 / 7 = 1.457
        scale = max(0.4375, 1.457) = 1.457  // use smax to avoid clipping positives
        zero_point = (-8 + 7 + 1) / 2 = 0

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
  - Block-Wise: Each thread processes assigned blocks using nested loops over block dimensions
*/

#ifdef per_tensor

void choose_qparams_per_tensor() {
  uint global_id = gl_GlobalInvocationID.x;
  uint local_id = gl_LocalInvocationID.x;
  uint total_threads = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

  uint total_elements = uint(t_in_sizes.x * t_in_sizes.y * t_in_sizes.z * t_in_sizes.w);

  // Each thread processes multiple elements with stride
  float thread_min = 1.0/0.0;  // +infinity
  float thread_max = -1.0/0.0; // -infinity
  bool found_valid = false;

  for (uint i = global_id; i < total_elements; i += total_threads) {
    float val = t_in[i];
    if (!isnan(val) && !isinf(val)) {
      if (!found_valid) {
        thread_min = val;
        thread_max = val;
        found_valid = true;
      } else {
        thread_min = min(thread_min, val);
        thread_max = max(thread_max, val);
      }
    }
  }

  // Intra-group reduction using shared memory
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

  // Final result calculation (single workgroup only)
  if (local_id == 0) {
    float global_min = shared_min[0];
    float global_max = shared_max[0];

    float scale_val;
    int zero_point_val;
    // Use default values: mapping_type=0 (ASYMMETRIC), eps from push constant
    calc_scale_zp(global_min, global_max, quant_min, quant_max, 0, eps, scale_val, zero_point_val);

    t_scale[0] = SCALE_OUT_T(scale_val);
    t_zero_point[0] = ZP_OUT_T(zero_point_val);
  }
}

#elif defined(per_token)

void choose_qparams_per_token() {
  uint total_elements = uint(t_in_sizes.x * t_in_sizes.y * t_in_sizes.z * t_in_sizes.w);
  uint token_size = total_elements / uint(num_tokens);

  const uint TOTAL_TOKENS = uint(num_tokens);

  /* each invocation handles token-ids: id, id+STRIDE, id+2·STRIDE … */
  const uint STRIDE = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
  for (uint token_id = gl_GlobalInvocationID.x; token_id < TOTAL_TOKENS; token_id += STRIDE) {
    // Calculate the start and end indices for this token
    uint token_start = token_id * token_size;
    uint token_end = token_start + token_size;

    // Each thread processes the entire token
    float lo = 1.0/0.0;   // +INF
    float hi = -1.0/0.0;  // -INF
    bool found_valid = false;

    // Process all elements in this token
    for (uint i = token_start; i < token_end; i++) {
      float val = t_in[i];
      if (!isnan(val) && !isinf(val)) {
        if (!found_valid) {
          lo = hi = val;
          found_valid = true;
        } else {
          lo = min(lo, val);
          hi = max(hi, val);
        }
      }
    }

    if (!found_valid) {
      // If no valid values were found, use default values
      lo = 0.0;
      hi = 0.0;
    }

    // Calculate scale and zero point directly
    float scale_val;
    int zero_point_val;
    // Use default values: mapping_type=0 (ASYMMETRIC), eps=1e-5
    calc_scale_zp(lo, hi, quant_min, quant_max, 0, 1e-5, scale_val, zero_point_val);

    // Write results
    t_scale[token_id] = SCALE_OUT_T(scale_val);
    t_zero_point[token_id] = ZP_OUT_T(zero_point_val);
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
  bc.x =  r;
  return bc;
}

void choose_qparams_block_wise() {
  const uint TOTAL_BLOCKS = uint(numBlocks.x * numBlocks.y * numBlocks.z * numBlocks.w);

  // each invocation handles block-ids: id, id+STRIDE, id+2·STRIDE
  const uint STRIDE = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
  for (uint block_id = gl_GlobalInvocationID.x; block_id < TOTAL_BLOCKS; block_id += STRIDE) {
    // block -> WHCN coordinate
    ivec4 bc = block_id_to_coord(block_id);
    ivec4 blockStart = bc * blockSize; // first element (inclusive)
    ivec4 blockEnd = blockStart + blockSize; // last element (exclusive)

    // min / max scan over the block
    float lo =  1.0/0.0; // +INF
    float hi = -1.0/0.0; // -INF
    bool found_valid = false;

    // Calculate actual block dimensions
    ivec4 actualBlockSize = blockEnd - blockStart;
    int blockElements = actualBlockSize.x * actualBlockSize.y * actualBlockSize.z * actualBlockSize.w;

    // Linear iteration over block elements
    for (int elemIdx = 0; elemIdx < blockElements; ++elemIdx) {
      // Convert linear index to 4D coordinates within block
      int remaining = elemIdx;
      int dn = remaining / (actualBlockSize.x * actualBlockSize.y * actualBlockSize.z);
      remaining -= dn * (actualBlockSize.x * actualBlockSize.y * actualBlockSize.z);
      int dc = remaining / (actualBlockSize.x * actualBlockSize.y);
      remaining -= dc * (actualBlockSize.x * actualBlockSize.y);
      int dh = remaining / actualBlockSize.x;
      int dw = remaining - dh * actualBlockSize.x;

      ivec4 tidx = blockStart + ivec4(dw, dh, dc, dn);
      uint idx = tidx_to_bufi(tidx, t_in_strides);
      float v = t_in[idx];

      if (!isnan(v) && !isinf(v)) {
        if (!found_valid) {
          lo = hi = v;
          found_valid = true;
        } else {
          lo = min(lo, v);
          hi = max(hi, v);
        }
      }
    }

    // Handle the case where no valid values were found in the block
    if (!found_valid) {
      lo = 0.0;
      hi = 0.0;
    }

    float scale_val;
    int zero_point_val;
    calc_scale_zp(lo, hi, quant_min, quant_max, mapping_type, eps, scale_val, zero_point_val);

    t_scale[block_id] = SCALE_OUT_T(scale_val);
    t_zero_point[block_id] = ZP_OUT_T(zero_point_val);
  }
}

#endif

void main() {
  choose_qparams_${MODE}();
}
