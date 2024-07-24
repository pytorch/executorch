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

from executorch.backends.xnnpack.partition.config.single_node_configs import (
    AddConfig,
    DeQuantizedPerTensorConfig,
    HardtanhConfig,
    QuantizedPerTensorConfig,
    ReLUConfig,
)
from executorch.backends.xnnpack.partition.config.special_node_configs import (
    BatchNormConfig,
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
    HardtanhConfig,
    AddConfig,
    ReLUConfig,
    # Quantization Op Configs
    QuantizedPerTensorConfig,
    DeQuantizedPerTensorConfig,
]
