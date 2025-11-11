/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace example {

/**
 * Computes the cross-attention mask for text + image inputs. Text tokens that
 * participate in cross-attention with an image token will show True in the mask
 * and follow the interleaved structure laid out in Fig. 7 of the Flamingo paper
 * (https://arxiv.org/pdf/2204.14198):
 *
 *     (1) Text tokens immediately following the image token up until the next
 * image token (2) Consecutive image tokens attend to subsequent text tokens
 *
 * ::
 *
 *           ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
 *      img1 │ ■ │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │   │ │   │ │   │ │   │ │   │
 *           └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
 *           ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
 *      img2 │   │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │   │ │   │ │   │ │   │ │   │
 *           └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
 *           ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
 *      img3 │   │ │   │ │   │ │   │ │   │ │   │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │ ■ │
 *           └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
 *         <img1> <img2>These  are   two  dogs. <img3> This   is    a    cat.
 *
 *
 *
 * Resultant mask is constructed per image and is of shape (text_seq_len,
 * image_seq_len), where True indicates that the token outputted from the image
 * encoder attends to the token in the text sequence in cross-attention. A list
 * of these masks are returned with length equal to number of images in the
 * sample.
 *
 * @param tokens Vector of tokens participating in the cross attention.
 * @param images Vector of images participating in the cross attention.
 * @param tile_size The size of the image tiles from the image transform.
 * @param patch_size The size of each patch. Used to divide the tiles into
 * patches. E.g. for patch_size = 40, a tile of shape (400, 400) will have 10x10
 * grid of patches with shape (40, 40) each. image_token_id (int): Token ID of
 * the image special token.
 * @param image_token_id The value of the image token.
 * @param out Out vector holding the raw data wrapped by the returned cross
 * attention masks.
 *
 * @returns A vector of cross attention masks, as Tensors, one for each image.
 */
std::vector<::executorch::extension::TensorPtr> cross_attention_mask(
    const std::vector<int>& tokens,
    const std::vector<::executorch::aten::Tensor>& images,
    size_t tile_size,
    size_t patch_size,
    int image_token_id,
    std::vector<std::vector<int>>& out);

} // namespace example
