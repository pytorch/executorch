/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Describes which texture axis the "batches" dimension runs along in a 4D
 * texture.
 */
#define BATCH_AXIS 2

/*
 * Divides input and rounds up to 4
 */
int divup4(int x) {
  return (x + 3) / 4;
}

/*
 * Aligns input to the next multiple of 4
 */
int alignup4(int x) {
  return (x + 3) & -4;
}

/*
 * Input: sizes of the tensor, index of which dimension is packed
 * Returns: sizes of the tensor with the size of the packed dimension aligned
 * up to the next multiple of 4
 */
ivec4 get_gpu_sizes(ivec4 sizes, int packed_dim) {
  sizes[packed_dim] = alignup4(sizes[packed_dim]);
  return sizes;
}

/*
 * Input: sizes of the tensor, dim to retrieve
 * Returns: the stride of the tensor, assuming contiguous memory layout, at the
 * specified dimension
 */
int get_nchw_stride(ivec4 sizes, int packed_dim) {
  if (packed_dim == 2) {
    return sizes.x * sizes.y;
  } else if (packed_dim == 1) {
    return sizes.x;
  } else {
    return 1;
  }
}

/*
 * Input: 4D index of the tensor, sizes of the tensor
 * Returns: the corresponding index to the tensors data buffer, assuming
 * contiguous memory layout
 */
int to_nchw_i(ivec4 idx, ivec4 sizes) {
  return (
      idx.x + idx.y * sizes.x + idx.z * sizes.y * sizes.x +
      idx.w * sizes.z * sizes.y * sizes.x);
}

ivec4 from_nchw_i(int buf_i, ivec4 sizes) {
  return ivec4(
      buf_i % sizes.x,
      (buf_i / (sizes.x)) % sizes.y,
      (buf_i / (sizes.x * sizes.y)) % sizes.z,
      (buf_i / (sizes.x * sizes.y * sizes.z)));
}

// Inverse of to_buffer_i
// Input: buffer_idx in the continuous nchw-buffer, sizes is the tensor shape
// Output: ivec4 user-level (w, h, c, n) coorindate
#define from_buffer_i(buf_i, sizes)            \
  ivec4(                                       \
      buf_i % sizes.x,                         \
      (buf_i / (sizes.x)) % sizes.y,           \
      (buf_i / (sizes.x * sizes.y)) % sizes.z, \
      (buf_i / (sizes.x * sizes.y * sizes.z)))

/*
 * Input: 3D texel position, sizes of the tensor, which dim is packed
 * Returns: the 4D tensor index cooresponding to the first element of the texel
 */
ivec4 to_tensor_idx(ivec3 pos, ivec4 sizes, int packed_dim) {
  ivec4 gpu_sizes = get_gpu_sizes(sizes, packed_dim);
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
  tensor_idx[3] /= gpu_sizes[BATCH_AXIS];
  tensor_idx[BATCH_AXIS] %= gpu_sizes[BATCH_AXIS];

  return tensor_idx;
}

/*
 * Input: 4D tensor index, sizes of the tensor, which dim is packed
 * Returns: the 3D texture position containing that element of the tensor
 */
ivec3 to_texture_pos(ivec4 idx, ivec4 sizes, int packed_dim) {
  ivec4 gpu_sizes = get_gpu_sizes(sizes, packed_dim);
  ivec3 pos = idx.xyz;
  pos[BATCH_AXIS] += idx.w * gpu_sizes[BATCH_AXIS];
  pos[packed_dim] /= 4;
  return pos;
}

/*
 * Input: 4D tensor index, sizes of the tensor, which dim is packed
 * Returns: the 3D texture position containing that element of the tensor,
 * along with the element within that texel to which the element belongs
 */
ivec4 to_texture_elem_pos(ivec4 idx, ivec4 sizes, int packed_dim) {
  ivec4 gpu_sizes = get_gpu_sizes(sizes, packed_dim);
  //return ivec4(idx.x, idx.y, (idx.z + idx.w * gpu_sizes.z) / 4, idx.z % 4);
  // pos[4] is set to a placeholder value:w
  ivec4 pos = idx.xyzx;
  pos[BATCH_AXIS] += idx.w * gpu_sizes[BATCH_AXIS];
  pos[packed_dim] /= 4;
  pos.w = idx[packed_dim] % 4;
  return pos;
}

// Input: idx is a user-level (w, h, c, n) coordinate. size is a pack-aligned
// size with the index in the texel.
// Output: ivec4, xyz is the texture position, w is the element index in the
// texel.
#define to_texture_pos_elem_C_packed(idx, sizes) \
  ivec4(idx.x, idx.y, (idx.z + idx.w * sizes.z) / 4, idx.z % 4)

#define to_texture_pos_elem_W_packed(idx, sizes) \
  ivec4(idx.x / 4, idx.y, (idx.z + idx.w * sizes.z), idx.x % 4)

#define to_texture_pos_elem_H_packed(idx, sizes) \
  ivec4(idx.x, idx.y / 4, (idx.z + idx.w * sizes.z), idx.y % 4)

// Input: idx is a ivec4 user-level coordinate, sizes is the tensor shape
// Output: buffer_idx in the continuous nchw-buffer.
#define to_buffer_i(idx, sizes)                          \
  (idx.x + idx.y * sizes.x + idx.z * sizes.y * sizes.x + \
   idx.w * sizes.z * sizes.y * sizes.x)

// Input: pos is a texture position, sizes is a pack-aligned size.
// Output: a user-level (w, h, c, n) coordinate
#define to_tensor_idx_C_packed(pos, sizes) \
  ivec4(pos.x, pos.y, (pos.z * 4) % sizes.z, (pos.z * 4) / sizes.z)

#define to_tensor_idx_W_packed(pos, sizes) \
  ivec4((pos.x * 4), pos.y, pos.z % sizes.z, pos.z / sizes.z)

#define to_tensor_idx_H_packed(pos, sizes) \
  ivec4(pos.x, (pos.y * 4), pos.z % sizes.z, pos.z / sizes.z)

// Input: idx is a user-level (w, h, c, n) coordinate. size is a pack-aligned
// size.
// Output: texture location
#define to_texture_pos_C_packed(idx, sizes) \
  ivec3(idx.x, idx.y, (idx.z + idx.w * sizes.z) / 4)

#define to_texture_pos_W_packed(idx, sizes) \
  ivec3(idx.x / 4, idx.y, (idx.z + idx.w * sizes.z))

#define to_texture_pos_H_packed(idx, sizes) \
  ivec3(idx.x, idx.y / 4, (idx.z + idx.w * sizes.z))

// Input: idx is a user-level (w, h, c, n) coordinate. size is a pack-aligned
// size with the index in the texel.
// Output: ivec4, xyz is the texture position, w is the element index in the
// texel.
#define to_texture_pos_elem_C_packed(idx, sizes) \
  ivec4(idx.x, idx.y, (idx.z + idx.w * sizes.z) / 4, idx.z % 4)

#define to_texture_pos_elem_W_packed(idx, sizes) \
  ivec4(idx.x / 4, idx.y, (idx.z + idx.w * sizes.z), idx.x % 4)

#define to_texture_pos_elem_H_packed(idx, sizes) \
  ivec4(idx.x, idx.y / 4, (idx.z + idx.w * sizes.z), idx.y % 4)

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
