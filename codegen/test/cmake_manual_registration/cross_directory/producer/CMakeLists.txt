# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

gen_selected_ops(LIB_NAME cross_dir_ops)
generate_bindings_for_kernels(
  LIB_NAME cross_dir_ops MANUAL_REGISTRATION REGISTRATION_NAME stable_cross_dir
)
