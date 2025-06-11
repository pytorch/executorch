# Copyright 2023 Martin Pavella
# Copyright 2024-2025 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    translator

Module contains functions for context-free conversion of various
things from Torch ATEN to TFLite.
"""

from typing import Optional, Tuple

import executorch.backends.nxp.backend.ir.lib.tflite.Padding as tflPadding
import executorch.backends.nxp.backend.ir.logger as logger


def torch_explicit_padding_to_tflite(torch_padding: list[int]) -> list[list[int]]:
    """Convert the attribute or input 'pad' of the Torch 'Pad' operator to the 'paddings' input of the TFLite 'Pad'
     class of operators.

    This function does NOT take tensor formats into consideration.
    """
    return [[dim_padding, dim_padding] for dim_padding in torch_padding]


def torch_padding_to_tflite_explicit_padding(
    torch_padding: list[int],
) -> list[list[int]]:
    """Convert a Torch attribute 'padding' of operators such as Conv, MaxPool or AveragePool, to a list of ints which
    is compatible with the TFLite 'Pad' operator.
    """
    tflite_padding = torch_explicit_padding_to_tflite(torch_padding)

    # TFLite also allows padding to the 'batch' and 'channels'. Torch does not
    tflite_padding.insert(0, [0, 0])
    tflite_padding.append([0, 0])

    return tflite_padding


def convert_padding(
    t_padding: list[int],
) -> Tuple[tflPadding.Padding, Optional[list[list[int]]]]:
    """Convert Torch operator attributes 'pads' and 'auto_pad' to TFLite.

    :param t_padding: Torch operator attribute 'padding'
    :return: A tuple.
                The first element is the converted TFLite padding.
                The second is None, if conversion is finished. Or it is a list of ints representing the explicit
                padding in TFLite format (compatible with the 'Pad' operator), which needs to be provided by a
                'Pad' operator. Caller must add this operator using model_builder!
    """

    if t_padding == [0, 0]:
        return tflPadding.Padding.VALID, None
    else:
        # 'padding' cannot be converted directly. Return 'VALID' and the required explicit padding and caller must
        # implement conversion by adding a 'Pad' operator.

        logger.d(
            "Explicit Torch 'padding' cannot be represented directly as 'VALID'. "
            "Inserting an extra 'Pad' operator."
        )

        # Torch 'padding' uses different format than TFLite 'Pad' operator. Convert the explicit padding.
        tflite_explicit_padding = torch_padding_to_tflite_explicit_padding(t_padding)

        return tflPadding.Padding.VALID, tflite_explicit_padding
