# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from .quantization_config import QuantizationConfig  # noqa  # usort: skip
from .arm_quantizer import (  # noqa
    EthosUQuantizer,
    get_symmetric_quantization_config,
    TOSAQuantizer,
)

# Used in tests
from .arm_quantizer_utils import is_annotated  # noqa

# Load quantized ops library.
try:
    import executorch.extension.pybindings.portable_lib
    import executorch.kernels.quantized  # noqa
except:
    import logging

    logging.info(
        "Failed to load portable_lib and quantized_aot_lib. To run quantized kernels AOT, either build "
        "Executorch with pybindings, or load your own custom built op library using torch.ops.load_library."
    )
    del logging
