/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("texture3d", DTYPE)}

#extension GL_EXT_debug_printf : enable

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}

${define_active_storage_type("texture3d")}

#define DEBUG_MODE

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "r", "t_inp", DTYPE, "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(push_constant) uniform restrict Block {
  int value_ref;
};

void main() {
  const int W = min(5, inp.sizes.x);
  const int H = min(5, inp.sizes.y);

  debugPrintfEXT(
      "\\n[print_texture] value_ref=%d, sizes=(%d, %d, %d, %d), printing %dx%d plane at c=0, n=0",
      value_ref,
      inp.sizes.x, inp.sizes.y, inp.sizes.z, inp.sizes.w,
      W, H);

  for (int y = 0; y < H; y++) {
    if (y == 0) {
      debugPrintfEXT("\\n[[");
    } else {
      debugPrintfEXT("\\n [");
    }

    for (int x = 0; x < W; x++) {
      TensorIndex4D t;
      t.data = ivec4(x, y, 0, 0);
      TextureElementIndex elem =
          tensor4d_idx_to_texture_element_idx_simple(inp, t);
      VEC4_T texel = texelFetch(t_inp, elem.pos, 0);
      float v = float(texel[elem.comp]);
      if (x < W - 1) {
        debugPrintfEXT("%8.4f, ", v);
      } else {
        debugPrintfEXT("%8.4f", v);
      }
    }

    if (y == H - 1) {
      debugPrintfEXT("]]\\n");
    } else {
      debugPrintfEXT("]");
    }
  }
}
