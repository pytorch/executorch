# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from executorch.backends.nxp.backend.ir import logger as logger
from executorch.backends.nxp.backend.ir.tflite_generator import (
    tflite_model as tflite_model,
)


def _buffer_has_data(t_buffer: tflite_model.Buffer) -> Optional[bool]:
    """Determine if given buffer has any data in it."""

    try:
        if t_buffer.data is None:
            return False

        size = t_buffer.data.size
        return size != 0

    except Exception as e:
        logger.d("'ModelBuilder.bufferHasData()' failed!")
        print(e)
        return None


def tensor_has_data(t_tensor: tflite_model.Tensor) -> bool:
    """Determine if given TFLite tensor has any data."""

    if t_tensor.tmp_buffer is None:
        return False

    res = _buffer_has_data(t_tensor.tmp_buffer)
    if res is None:
        res = False

    return res


def all_tensors_are_static(*list_of_tensors) -> bool:
    """Return True, if all tensors in 'list_of_tensors' have data stored in them.

    :param list_of_tensors: List of TFLite tensors to check.
    :return: True, if all tensors are static. False, if at least 1 is not static.
    """

    return all(tensor_has_data(t) for t in list_of_tensors)
