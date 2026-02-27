from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from functorch import make_fx
from functorch.experimental import functionalize
from torch_tensorrt._utils import sanitized_torch_version
from torch_tensorrt.fx.passes.lower_basic_pass import (
    replace_op_with_indices,
    run_const_fold,
)

from packaging import version

from .types import Shape, TRTDataType


class Frameworks(Enum):
    NUMPY = "numpy"
    TORCH = "torch"
    TRT = "trt"


DataTypeEquivalence: Dict[
    TRTDataType, Dict[Frameworks, Union[TRTDataType, np.dtype, torch.dtype]]
] = {
    trt.int8: {
        Frameworks.NUMPY: np.int8,
        Frameworks.TORCH: torch.int8,
        Frameworks.TRT: trt.int8,
    },
    trt.int32: {
        Frameworks.NUMPY: np.int32,
        Frameworks.TORCH: torch.int32,
        Frameworks.TRT: trt.int32,
    },
    trt.int64: {
        Frameworks.NUMPY: np.int64,
        Frameworks.TORCH: torch.int64,
        Frameworks.TRT: trt.int64,
    },
    trt.float16: {
        Frameworks.NUMPY: np.float16,
        Frameworks.TORCH: torch.float16,
        Frameworks.TRT: trt.float16,
    },
    trt.float32: {
        Frameworks.NUMPY: np.float32,
        Frameworks.TORCH: torch.float32,
        Frameworks.TRT: trt.float32,
    },
    trt.bool: {
        Frameworks.NUMPY: bool,
        Frameworks.TORCH: torch.bool,
        Frameworks.TRT: trt.bool,
    },
}

if trt.__version__ >= "7.0":
    DataTypeEquivalence[trt.bool] = {
        Frameworks.NUMPY: np.bool_,
        Frameworks.TORCH: torch.bool,
        Frameworks.TRT: trt.bool,
    }


class LowerPrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    BF16 = "bf16"

    @staticmethod
    def from_str(label: str) -> Optional["LowerPrecision"]:
        if label in ("fp32", "float32", "float", "torch.float32"):
            return LowerPrecision.FP32
        elif label in ("fp16", "float16", "half", "torch.half", "torch.float16"):
            return LowerPrecision.FP16
        elif label in ("int8"):
            return LowerPrecision.INT8
        elif label in ("bf16", "bfloat16", "torch.bfloat16"):
            return LowerPrecision.BF16
        else:
            return None


def unified_dtype_converter(
    dtype: Union[TRTDataType, torch.dtype, np.dtype], to: Frameworks
) -> Union[np.dtype, torch.dtype, TRTDataType]:
    """
    Convert TensorRT, Numpy, or Torch data types to any other of those data types.

    Args:
        dtype (TRTDataType, torch.dtype, np.dtype): A TensorRT, Numpy, or Torch data type.
        to (Frameworks): The framework to convert the data type to.

    Returns:
        The equivalent data type in the requested framework.
    """
    assert to in Frameworks, f"Expected valid Framework for translation, got {to}"
    trt_major_version = int(trt.__version__.split(".")[0])
    if dtype in (np.int8, torch.int8, trt.int8):
        return DataTypeEquivalence[trt.int8][to]
    elif trt_major_version >= 7 and dtype in (np.bool_, torch.bool, trt.bool):
        return DataTypeEquivalence[trt.bool][to]
    elif dtype in (np.int32, torch.int32, trt.int32):
        return DataTypeEquivalence[trt.int32][to]
    elif dtype in (np.int64, torch.int64, trt.int64):
        return DataTypeEquivalence[trt.int64][to]
    elif dtype in (np.float16, torch.float16, trt.float16):
        return DataTypeEquivalence[trt.float16][to]
    elif dtype in (np.float32, torch.float32, trt.float32):
        return DataTypeEquivalence[trt.float32][to]
    else:
        raise TypeError("%s is not a supported dtype" % dtype)


def get_dynamic_dims(shape: Shape) -> List[int]:
    """
    This function finds the dynamic dimensions in the given
    shape. A dimension is dynamic if it's -1.

    Args:
        shape (Shape): A sequence of integer that represents
            the shape of a tensor.

    Returns:
        A list of integers contains all the dynamic dimensions
        in the given shape
    """
    dynamic_dims = []

    for i, s in enumerate(shape):
        if s == -1:
            dynamic_dims.append(i)

    return dynamic_dims


def proxytensor_trace(mod, inputs):
    mod.eval()

    def f(*inp):
        return mod(*inp)

    mod = make_fx(functionalize(f))(*inputs)

    # Remove const operation. For ex, nn.Linear has transpose operation on weight
    mod.graph.eliminate_dead_code()
    mod = run_const_fold(mod)
    mod = replace_op_with_indices(mod)
    return mod


def req_torch_version(min_torch_version: str = "2.dev"):
    """
    Create a decorator which verifies the Torch version installed
    against a specified version range

    Args:
        min_torch_version (str): The minimum required Torch version
        for the decorated function to work properly

    Returns:
        A decorator which raises a descriptive error message if
        an unsupported Torch version is used
    """

    def nested_decorator(f: Callable):
        def function_wrapper(*args, **kwargs):
            # Parse minimum and current Torch versions
            min_version = version.parse(min_torch_version)
            current_version = version.parse(sanitized_torch_version())

            if current_version < min_version:
                raise AssertionError(
                    f"Expected Torch version {min_torch_version} or greater, "
                    + f"when calling {f}. Detected version {torch.__version__}"
                )
            else:
                return f(*args, **kwargs)

        return function_wrapper

    return nested_decorator
