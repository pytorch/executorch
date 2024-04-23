/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Width Dim Index, assuming (W, H, C, N) order
#define W_DIM 0
// Height, assuming (W, H, C, N) order
#define H_DIM 1
// Channels, assuming (W, H, C, N) order
#define C_DIM 2

/*
 * Describes which texture axis the "batches" dimension runs along in a 4D
 * texture.
 *
 * Currently it is set to 2 since we represent batches by concatenating along
 * the channels dim, which has index 2 in (W, H, C, N) order and maps to the
 * depth dimension of a texture, which also corresponds to index 2 in (x, y, z)
 * order.
 */
#define BATCH_AXIS 2

//
// Basic Indexing Utility Macros and Functions
//

/*
 * Divides input and rounds up to 4
 */
#define divup4(x) ((x + 3) / 4)

/*
 * Aligns input to the next multiple of 4
 */
#define alignup4(x) ((x + 3) & -4)

//
// (w, h, c, n) Tensor Index <-> Contiguous Buffer Index Conversion
//

/*
 * Input: (w, h, c, n) tensor index, (W, H, C, N) sizes of a tensor, which dim
 *        is packed along a texel
 * Output: A ivec4 containing the buffer indices corresponding to each texel
 *         element.
 */
ivec4 get_texel_nchw_buffer_ixs(ivec4 idx, ivec4 sizes, int packed_dim) {
  ivec4 strides =
      ivec4(1, sizes.x, sizes.x * sizes.y, sizes.x * sizes.y * sizes.z);

  int base_i = idx.x * strides.x + idx.y * strides.y + idx.z * strides.z +
      idx.w * strides.w;

  return base_i + ivec4(0, 1, 2, 3) * strides[packed_dim];
}

/*
 * Input: Index into a tensor's data buffer, (W, H, C, N) sizes of a tensor
 * Returns: The WCHN index of the tensor that corresponds to the specified
 *          buffer index, assuming the buffer has contiguous memory layout
 */
ivec4 from_nchw_buffer_i(int buf_i, ivec4 sizes) {
  return ivec4(
      buf_i % sizes.x,
      (buf_i / (sizes.x)) % sizes.y,
      (buf_i / (sizes.x * sizes.y)) % sizes.z,
      (buf_i / (sizes.x * sizes.y * sizes.z)));
}

//
// (w, h, c, n) Tensor Index <-> (x, y, z) Texture Position Conversion
//

/*
 * Input: (x, y, z) texel position, (W, H, C, N) sizes of the tensor, which dim
 *        is packed along a texel
 * Output: Whether the texel position is outside the bounds of the image texture
 *         given the size and packed dimension of the tensor.
 */
bool pos_out_of_bounds(ivec3 pos, ivec4 sizes, int packed_dim) {
  // Align packed dim to next multiple of 4 to account for texel padding
  sizes[packed_dim] = alignup4(sizes[packed_dim]);

  ivec3 max_pos = sizes.xyz;
  max_pos[BATCH_AXIS] += sizes.w * sizes[BATCH_AXIS];
  max_pos[packed_dim] /= 4;
  return (any(greaterThanEqual(pos, max_pos)));
}

/*
 * Input: (x, y, z) texel position, (W, H, C, N) sizes of the tensor,
 *        which dim is packed along a texel
 * Returns: the (w, h, c, n) tensor index cooresponding to the first element of
 *          the texel at the specified position
 */
ivec4 to_tensor_idx(ivec3 pos, ivec4 sizes, int packed_dim) {
  // Align packed dim to next multiple of 4 to account for texel padding
  sizes[packed_dim] = alignup4(sizes[packed_dim]);

  // Packed dim contains 4 elements per texel
  pos[packed_dim] *= 4;
  // Construct the initial tensor index via swizzling
#if BATCH_AXIS == 2
  ivec4 tensor_idx = pos.xyzz;
#endif
#if BATCH_AXIS == 1
  ivec4 tensor_idx = pos.xyzy;
#endif
#if BATCH_AXIS == 0
  ivec4 tensor_idx = pos.xyzx;
#endif
  // Adjust the axis that the batch dim runs along
  tensor_idx[3] /= sizes[BATCH_AXIS];
  tensor_idx[BATCH_AXIS] %= sizes[BATCH_AXIS];

  return tensor_idx;
}

/*
 * Input: (w, h, c, n) tensor index, (W, H, C, N) sizes of a tensor, which dim
 *        is packed along a texel
 * Returns: the (x, y, z) texture position containing element of the tensor at
 *          the specified index
 */
ivec3 to_texture_pos(ivec4 idx, ivec4 sizes, int packed_dim) {
  // Align packed dim to next multiple of 4 to account for texel padding
  sizes[packed_dim] = alignup4(sizes[packed_dim]);

  ivec3 pos = idx.xyz;
  pos[BATCH_AXIS] += idx.w * sizes[BATCH_AXIS];
  pos[packed_dim] /= 4;
  return pos;
}

/*
 * Input: (w, h, c, n) tensor index, (W, H, C, N) sizes of the tensor, which dim
 *        is packed along a texel
 * Returns: the (x, y, z, i) texture position containing the element of the
 *          tensor at the specified index, where i is the component within the
 *          texel to which the element belongs
 */
ivec4 to_texture_elem_pos(ivec4 idx, ivec4 sizes, int packed_dim) {
  // Align packed dim to next multiple of 4 to account for texel padding
  sizes[packed_dim] = alignup4(sizes[packed_dim]);

  //  pos[4] is set to a placeholder value
  ivec4 pos = idx.xyzx;
  pos[BATCH_AXIS] += idx.w * sizes[BATCH_AXIS];
  pos[packed_dim] /= 4;
  pos.w = idx[packed_dim] % 4;
  return pos;
}

//
// Miscellaneous Utility Functions and Macros
//

// Given a buffer(1-D) index cur, compute a new index where the corresponding
// tensor(N-D)'s adjacent dimensions are swapped. The parameters x,y and plane
// describe sizes. As an example, let's say we want to swap dimensions 0,1 for a
// tensor of shape {4,3,2,24} to obtain {3,4,2,24}. Then, x=4, y=3 and
// plane=2*24=48.
#define swap_adj_dims(cur, x, y, plane)                        \
  cur +                                                        \
      plane *                                                  \
          ((1 - y) * ((cur % (x * y * plane)) / (y * plane)) + \
           (x - 1) * ((cur % (y * plane)) / plane))
