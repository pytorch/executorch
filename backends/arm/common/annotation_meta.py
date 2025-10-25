# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmAnnotationInfo:
    """
    Data class to carry Arm-specific annotation information through the pipeline.
    This is intended to be attached to node.meta['custom'] and propagated
    through partitioning and backend stages. As it's propagated through the pipeline,
    it's intentionally minimal and only carries whether the node is quantized or not.
    """

    quantized: bool
    CUSTOM_META_KEY: str = "_arm_annotation_info"
