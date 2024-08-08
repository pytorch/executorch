# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Type

from executorch.backends.xnnpack.partition.config.gemm_configs import (
    AddmmConfig,
    ConvolutionConfig,
    LinearConfig,
)

from executorch.backends.xnnpack.partition.config.generic_node_configs import (
    AbsConfig,
    AddConfig,
    AvgPoolingConfig,
    CatConfig,
    CeilConfig,
    ClampConfig,
    DeQuantizedPerTensorConfig,
    DivConfig,
    FloorConfig,
    HardswishConfig,
    # EluConfig,
    HardtanhConfig,
    LeakyReLUConfig,
    MaximumConfig,
    MaxPool2dConfig,
    MeanDimConfig,
    MinimumConfig,
    MulConfig,
    NegConfig,
    PermuteConfig,
    PowConfig,
    QuantizedPerTensorConfig,
    ReLUConfig,
    SigmoidConfig,
    SliceCopyConfig,
    SoftmaxConfig,
    UpsampleBilinear2dConfig,
)
from executorch.backends.xnnpack.partition.config.node_configs import (
    BatchNormConfig,
    MaxDimConfig,
    PreluConfig,
)
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    XNNPartitionerConfig,
)

ALL_PARTITIONER_CONFIGS: List[Type[XNNPartitionerConfig]] = [
    # GEMM-like Configs
    AddmmConfig,
    LinearConfig,
    ConvolutionConfig,
    # BatchNorm Config
    BatchNormConfig,
    # Single Node Configs
    AbsConfig,
    AvgPoolingConfig,
    AddConfig,
    CatConfig,
    CeilConfig,
    ClampConfig,
    DivConfig,
    FloorConfig,
    HardtanhConfig,
    HardswishConfig,
    LeakyReLUConfig,
    MaxDimConfig,
    MaximumConfig,
    MaxPool2dConfig,
    MeanDimConfig,
    MinimumConfig,
    MulConfig,
    NegConfig,
    PowConfig,
    PreluConfig,
    SoftmaxConfig,
    SigmoidConfig,
    SliceCopyConfig,
    PermuteConfig,
    # EluConfig, # Waiting for PyTorch Pin Update
    ReLUConfig,
    UpsampleBilinear2dConfig,
    # Quantization Op Configs
    QuantizedPerTensorConfig,
    DeQuantizedPerTensorConfig,
]
