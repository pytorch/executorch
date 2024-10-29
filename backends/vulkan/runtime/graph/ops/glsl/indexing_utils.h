/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef INDEXING_UTILS_H
#define INDEXING_UTILS_H

/*
 * The functions defined in this header file use the following shorthand to
 * represent tensor related data structures.
 *
 * tidx  - ivec4 tensor indices, listed in WHCN order.
 *
 * pos   - ivec3 texel position, used to fetch from an image texture via the
 *         texelFetch(image, pos, lod) GLSL function.
 * posi  - ivec4 texel element position. It is the same as pos, except with an
 *         additional component of the index of an element within the texel.
 * lpos  - ivec3 logical position, listed in WHC order. This is a permutation of
 *         texture position based on a tensor's axis_map. lpos.x is the position
 *         component that corresponds to the tensor's width dimension, lpos.y is
 *         the position component that corresponds to the tensor's height dim,
 *         and so on.
 *
 * bufi  - int index into a GPU buffer that backs a tensor.
 * nchwi - int index into a staging buffer for a tensor. The data in the
 *         staging buffer is stored in contiguous data layout, irrespective of
 *         the tensor's strides.
 */

// Width Dim Index, assuming WHCN order
#define W_DIM 0
// Height, assuming WHCN order
#define H_DIM 1
// Channels, assuming WHCN order
#define C_DIM 2

/*
 * Fast division by 4 using bit shifting
 */
#define div4(x) (x >> 2)

/*
 * Divides input and rounds up to 4
 */
#define divup4(x) ((x + 3) >> 2)

/*
 * Aligns input to the next multiple of 4
 */
#define alignup4(x) ((x + 3) & -4)

/*
 * Fast modulo by 4 using bit masking
 */
#define mod4(x) (x & 3)

/*
 * Find the packed dimension of a tensor given its strides. The packed dimension
 * is the "fastest moving" dimension which will have a stride of 1.
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
 * Get the staging buffer indices that contain the data of the texel that
 * corresponds to the provided tensor index. Since the texel have 4 elements,
 * 4 buffer indices will be retrieved.
 */
ivec4 tidx_to_nchwi(const ivec4 tidx, const ivec4 sizes, const int packed_dim) {
  ivec4 strides =
      ivec4(1, sizes.x, sizes.x * sizes.y, sizes.x * sizes.y * sizes.z);

  int base_i = tidx.x * strides.x + tidx.y * strides.y + tidx.z * strides.z +
      tidx.w * strides.w;

  return base_i + ivec4(0, 1, 2, 3) * strides[packed_dim];
}

ivec4 nchwi_to_tidx(const int nchwi, const ivec4 sizes) {
  return ivec4(
      nchwi % sizes.x,
      (nchwi / (sizes.x)) % sizes.y,
      (nchwi / (sizes.x * sizes.y)) % sizes.z,
      (nchwi / (sizes.x * sizes.y * sizes.z)));
}

int tidx_to_nchwi(const ivec4 tidx, const ivec4 sizes) {
  return tidx.w * sizes.x * sizes.y * sizes.z + tidx.z * sizes.x * sizes.y +
      tidx.y * sizes.x + tidx.x;
}

// TODO(ssjia): make this function use dim order so that it can work with any
// dim order. Currently it assumes that the dim order is contiguous, except for
// the packed dim.
ivec4 bufi_to_tidx(int bufi, const ivec4 strides, const int packed_dim) {
  ivec4 idx;
  for (int i = 3; i >= 0; i--) {
    if (i != packed_dim) {
      idx[i] = bufi / strides[i];
      bufi %= strides[i];
    }
  }
  idx[packed_dim] = bufi;
  return idx;
}

// Convenience overload of the above function, which will determine the packed
// dim from the strides automatically so it doesn't have to be passed in as a
// function argument.
ivec4 bufi_to_tidx(const int bufi, const ivec4 strides) {
  int packed_dim = find_packed_dim(strides);
  return bufi_to_tidx(bufi, strides, packed_dim);
}

int tidx_to_bufi(const ivec4 tidx, ivec4 strides) {
  return tidx.x * strides.x + tidx.y * strides.y + tidx.z * strides.z +
      tidx.w * strides.w;
}

ivec4 lpos_to_tidx(
    ivec3 lpos,
    ivec4 sizes,
    const int batch_inner_dim,
    const int packed_dim) {
  // Align packed dim to next multiple of 4 to account for texel padding
  sizes[packed_dim] = alignup4(sizes[packed_dim]);
  // Moving 1 texel along the packed dim traverses 4 tensor elements
  lpos[packed_dim] *= 4;

  ivec4 tidx = ivec4(lpos, 0);

  if (sizes.w > 1) {
    tidx.w = tidx[batch_inner_dim] / sizes[batch_inner_dim];
    tidx[batch_inner_dim] %= sizes[batch_inner_dim];
  }
  return tidx;
}

ivec3 tidx_to_lpos(
    ivec4 tidx,
    ivec4 sizes,
    const int batch_inner_dim,
    const int packed_dim) {
  // Align packed dim to next multiple of 4 to account for texel padding
  sizes[packed_dim] = alignup4(sizes[packed_dim]);

  ivec3 lpos = tidx.xyz;

  // Adjust batch inner dim by batch index if needed
  if (sizes.w > 1) {
    lpos[batch_inner_dim] += tidx.w * sizes[batch_inner_dim];
  }
  // Fast division by 4, since moving 1 texel along the packed dim traverses 4
  // tensor elements.
  lpos[packed_dim] >>= 2;
  return lpos;
}

ivec3 tidx_to_pos(
    ivec4 tidx,
    ivec4 sizes,
    const ivec4 axis_map,
    const int packed_dim) {
  // Align packed dim to next multiple of 4 to account for texel padding
  sizes[packed_dim] = alignup4(sizes[packed_dim]);

  ivec3 pos;
  for (int dim = 0; dim < 3; ++dim) {
    pos[axis_map[dim]] = tidx[dim];
  }

  // Adjust batch inner dim by batch index if needed
  if (sizes.w > 1) {
    pos[axis_map[axis_map.w]] += tidx.w * sizes[axis_map.w];
  }
  // Fast division by 4, since moving 1 texel along the packed dim traverses 4
  // tensor elements.
  pos[axis_map[packed_dim]] >>= 2;
  return pos;
}

ivec4 tidx_to_posi(
    ivec4 tidx,
    ivec4 sizes,
    const ivec4 axis_map,
    const int packed_dim) {
  return ivec4(
      tidx_to_pos(tidx, sizes, axis_map, packed_dim), tidx[packed_dim] % 4);
}

ivec3 lpos_to_pos(const ivec3 lpos, const ivec4 axis_map) {
  ivec3 pos;
  pos[axis_map.x] = lpos.x;
  pos[axis_map.y] = lpos.y;
  pos[axis_map.z] = lpos.z;
  return pos;
}

#ifdef USING_BUFFER
#define load_texel(buf, idx) buf[idx]
#elif defined(USING_TEXTURE2D)
#define load_texel(im, pos) texelFetch(im, pos.xy, 0)
#define load_texel_lpos(im, lpos, axis_map) \
  texelFetch(im, lpos_to_pos(lpos, axis_map).xy, 0)
#else // defined(USING_TEXTURE3D)
#define load_texel(im, pos) texelFetch(im, pos, 0)
#define load_texel_lpos(im, lpos, axis_map) \
  texelFetch(im, lpos_to_pos(lpos, axis_map), 0)
#endif

#ifdef USING_BUFFER
#define write_texel(buf, idx, texel) buf[idx] = texel
#elif defined(USING_TEXTURE2D)
#define write_texel(im, pos, texel) imageStore(im, pos.xy, texel)
#define write_texel_lpos(im, lpos, texel, axis_map) \
  imageStore(im, lpos_to_pos(lpos, axis_map).xy, texel)
#else // defined(USING_TEXTURE3D)
#define write_texel(im, pos, texel) imageStore(im, pos, texel)
#define write_texel_lpos(im, lpos, texel, axis_map) \
  imageStore(im, lpos_to_pos(lpos, axis_map), texel)
#endif

// Converts hashed axis mapping and packed dim to a ivec4
// e.g. 0x000102, 2 -> ivec4(0, 1, 2, 2)
// e.g. 0x010200, 1 -> ivec4(1, 2, 0, 1)
#define UNHASH_AXIS_MAP(hash, packed_dim) \
  ivec4(hash >> 16, (hash >> 8) & 0xFF, hash & 0xFF, packed_dim)
#define DEFAULT_AXIS_MAP_HASH 0x000102

/************************
 * Deprecated Functions *
 ************************/

// The below functions and macros are in the process of being deprecated in
// favor of newer indexing functions that account for axis mapping and have more
// explicit function names and more updated terminology.

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

// Return the x, y, z and index value the channel-packed 3D tensor from the {n,
// c, h, w}-index.
ivec4 get_channel_packed_pos_from_index(ivec4 nchw, ivec4 sizes) {
  int aligned_c = alignup4(sizes.y);
  int c_stride = aligned_c / 4;

  return ivec4(nchw.w, nchw.z, nchw.x * c_stride + nchw.y / 4, nchw.y % 4);
}

#endif // INDEXING_UTILS_H
