# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""How the CUDA backend picks the best config of each autotuned Triton kernel
(Inductor-generated and ours) while AOTInductor compiles."""
