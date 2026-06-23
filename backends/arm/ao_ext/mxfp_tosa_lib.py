# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.library import Library

# MXFP TOSA library definition for the Arm backend containing.
# This library will generate custom ops like the following example:
#   torch.ops.tosa_mxfp.linear.default
#   torch.ops.tosa_mxfp.conv2d.default
MXFP_TOSA_LIB = Library("tosa_mxfp", "DEF")
