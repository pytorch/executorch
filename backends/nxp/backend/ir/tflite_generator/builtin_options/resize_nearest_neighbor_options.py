# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import flatbuffers as fb

from executorch.backends.nxp.backend.ir.lib.tflite import ResizeNearestNeighborOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


# noinspection SpellCheckingInspection
class ResizeNearestNeighbor(meta.BuiltinOptions):
    align_corners: bool
    half_pixel_centers: bool

    def __init__(self, align_corners: bool, half_pixel_centers: bool) -> None:
        super().__init__(
            BuiltinOptions.ResizeNearestNeighborOptions,
            BuiltinOperator.RESIZE_NEAREST_NEIGHBOR,
        )
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers

    def gen_tflite(self, builder: fb.Builder):
        ResizeNearestNeighborOptions.Start(builder)

        ResizeNearestNeighborOptions.AddAlignCorners(builder, self.align_corners)
        ResizeNearestNeighborOptions.AddHalfPixelCenters(
            builder, self.half_pixel_centers
        )

        return ResizeNearestNeighborOptions.End(builder)
