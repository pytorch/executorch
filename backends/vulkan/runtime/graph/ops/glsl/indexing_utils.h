/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef INDEXING_UTILS_H
#define INDEXING_UTILS_H

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
 * Fast division by 4 using bit shifting
 */
#define div4(x) (x >> 2)

/*
 * Divides input and rounds up to 4
 */
#define divup4(x) ((x + 3) / 4)

/*
 * Aligns input to the next multiple of 4
 */
#define alignup4(x) ((x + 3) & -4)

/*
 * Input: (W, H, C, N) strides of a tensor
 * Returns: the WHCN index of the fastest moving dimension
 */
int find_packed_dim(const ivec4 strides) {
  int packed_dim = 0;
  for (int i = 0; i <= 3; i++) {
    if (strides[i] == 1) {
      packed_dim = i;
      break;
    }
  }
  return packed_dim;
}

/*
 * Return the elements of a texture position such that the first element is the
 * texture coordinate corresponding to the width dimension, the second element
 * is the texture coordinate corresponding to the height dimension, and the
 * third element is the texture coordinate corresponding to the channels
 * dimension.
 */
ivec3 get_logical_pos(const ivec3 pos, const ivec4 axis_map) {
  return ivec3(pos[axis_map.x], pos[axis_map.y], pos[axis_map.z]);
}

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

int to_nchw_buffer_i(const ivec4 tensor_idx, const ivec4 sizes) {
  return tensor_idx.w * sizes.x * sizes.y * sizes.z +
      tensor_idx.z * sizes.x * sizes.y + tensor_idx.y * sizes.x + tensor_idx.x;
}

/*
 * Input: Texel buffer index, (W, H, C, N) strides of a tensor, which dim is
 *        packed along a texel
 * Returns: The (w, h, c, n) tensor index corresponding to the buffer element
 */
ivec4 to_tensor_idx(int buffer_id, const ivec4 strides, const int packed_dim) {
  ivec4 idx;
  for (int i = 3; i >= 0; i--) {
    if (i != packed_dim) {
      idx[i] = buffer_id / strides[i];
      buffer_id %= strides[i];
    }
  }
  idx[packed_dim] = buffer_id;
  return idx;
}

/*
 * Input: Texel buffer index, (W, H, C, N) strides of a tensor
 * Returns: The (w, h, c, n) tensor index corresponding to the buffer element
 *
 * This is a convenience overload of the above function. If the packed dim is
 * not known, it can be found by finding the first dimension with a stride of 1.
 * However, this process adds some overhead, so if performance is a concern then
 * the above function should be used instead so that the packed dim is provided.
 */
ivec4 to_tensor_idx(int buffer_id, const ivec4 strides) {
  int packed_dim = find_packed_dim(strides);
  return to_tensor_idx(buffer_id, strides, packed_dim);
}

/*
 * Input: (w, h, c, n) tensor index, (W, H, C, N) strides of the tensor buffer
 * Returns: the buffer index corresponding to the specified tensor index
 */
int to_buffer_id(const ivec4 tensor_idx, ivec4 strides) {
  return tensor_idx.x * strides.x + tensor_idx.y * strides.y +
      tensor_idx.z * strides.z + tensor_idx.w * strides.w;
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
 * Derive (w,h,c,n) tensor indices from (x,y,z) texture position using axis
 * mapping.
 */
ivec4 to_tensor_idx(
    ivec3 pos,
    ivec4 sizes,
    const ivec4 axis_map,
    const int packed_dim) {
  // Align packed dim to next multiple of 4 to account for texel padding
  sizes[packed_dim] = alignup4(sizes[packed_dim]);

  // Packed dim contains 4 elements per texel, so moving 1 unit traverses 4
  // elements in the tensor.
  pos[axis_map[packed_dim]] *= 4;

  ivec4 tensor_idx;
  for (int dim = 0; dim < 3; ++dim) {
    tensor_idx[dim] = pos[axis_map[dim]];
  }

  // Early return if batch is 1. Batch index will be 0.
  if (sizes.w == 1) {
    tensor_idx.w = 0;
    return tensor_idx;
  }

  // Else, adjust the dim that's concatenated with batch. Note that the axis
  // mapping for the batch dim indicates WHCN dim index of the dim that it is
  // concatenated with, not a texture axis.
  tensor_idx.w = tensor_idx[axis_map[3]] / sizes[axis_map[3]];
  tensor_idx[axis_map[3]] %= sizes[axis_map[3]];

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
 * Derive (x,y,z) texture position from (w,h,c,n) tensor indices using axis
 * mapping.
 */
ivec3 to_texture_pos(
    const ivec4 idx,
    ivec4 sizes,
    const ivec4 axis_map,
    const int packed_dim) {
  // Align packed dim to next multiple of 4 to account for texel padding
  sizes[packed_dim] = alignup4(sizes[packed_dim]);

  ivec3 pos;
  for (int dim = 0; dim < 3; ++dim) {
    pos[axis_map[dim]] = idx[dim];
  }

  // Adjust batch dim if needed
  if (sizes.w > 1) {
    pos[axis_map[axis_map[3]]] += idx.w * sizes.w;
  }

  // Adjust packed dim. Moving 1 texel unit along the packed dim traverses 4
  // tensor elements in that dim.
  pos[axis_map[packed_dim]] /= 4;
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

/*
 * Derive (x,y,z,i) texel element position from the (w,h,c,n) tensor index using
 * the axis mapping.
 */
ivec4 to_texture_elem_pos(
    const ivec4 idx,
    ivec4 sizes,
    const ivec4 axis_map,
    const int packed_dim) {
  // Align packed dim to next multiple of 4 to account for texel padding
  sizes[packed_dim] = alignup4(sizes[packed_dim]);

  ivec4 pos;
  for (int dim = 0; dim < 3; ++dim) {
    pos[axis_map[dim]] = idx[dim];
  }

  // Adjust batch dim if needed
  if (sizes.w > 1) {
    pos[axis_map[axis_map[3]]] += idx.w * sizes.w;
  }

  // Adjust packed dim. Moving 1 texel unit along the packed dim traverses 4
  // tensor elements in that dim.
  pos[axis_map[packed_dim]] /= 4;
  pos.w = idx[packed_dim] % 4;
  return pos;
}

//
// Convert between physical texture position and logical tensor position
//

/*
 * Derive (x,y,z) physical texture position from (w,h,d) logical texture
 * position using the axis mapping.
 */
ivec3 to_texture_pos(const ivec3 logical_pos, const ivec4 axis_map) {
  ivec3 pos;
  pos[axis_map.x] = logical_pos.x;
  pos[axis_map.y] = logical_pos.y;
  pos[axis_map.z] = logical_pos.z;
  return pos;
}

//
// Texel Access and Storage
//

#ifdef USING_BUFFER
#define load_texel(buf, idx) buf[idx]
#elif defined(USING_TEXTURE2D)
#define load_texel(im, pos) texelFetch(im, pos.xy, 0)
#else // defined(USING_TEXTURE3D)
#define load_texel(im, pos) texelFetch(im, pos, 0)
#endif

#ifdef USING_BUFFER
#define write_texel(buf, idx, texel) buf[idx] = texel
#elif defined(USING_TEXTURE2D)
#define write_texel(im, pos, texel) imageStore(im, pos.xy, texel)
#else // defined(USING_TEXTURE3D)
#define write_texel(im, pos, texel) imageStore(im, pos, texel)
#endif

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

// Return the x, y, z and index value the channel-packed 3D tensor from the {n,
// c, h, w}-index.
ivec4 get_channel_packed_pos_from_index(ivec4 nchw, ivec4 sizes) {
  int aligned_c = alignup4(sizes.y);
  int c_stride = aligned_c / 4;

  return ivec4(nchw.w, nchw.z, nchw.x * c_stride + nchw.y / 4, nchw.y % 4);
}

#endif // INDEXING_UTILS_H
