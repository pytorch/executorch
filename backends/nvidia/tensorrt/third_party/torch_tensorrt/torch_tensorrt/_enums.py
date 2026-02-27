from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Optional, Type, Union

import numpy as np
import tensorrt as trt
import torch
from torch_tensorrt._features import ENABLED_FEATURES, needs_torch_tensorrt_runtime
from torch_tensorrt._utils import is_tensorrt_version_supported


class dtype(Enum):
    """Enum to describe data types to Torch-TensorRT, has compatibility with torch, tensorrt and numpy dtypes"""

    # Supported types in Torch-TensorRT
    unknown = auto()
    """Sentinel value

    :meta hide-value:
    """

    u8 = auto()
    """Unsigned 8 bit integer, equivalent to ``dtype.uint8``

    :meta hide-value:
    """

    i8 = auto()
    """Signed 8 bit integer, equivalent to ``dtype.int8``, when enabled as a kernel precision typically requires the model to support quantization

    :meta hide-value:
    """

    i32 = auto()
    """Signed 32 bit integer, equivalent to ``dtype.int32`` and ``dtype.int``

    :meta hide-value:
    """

    i64 = auto()
    """Signed 64 bit integer, equivalent to ``dtype.int64`` and ``dtype.long``

    :meta hide-value:
    """

    f16 = auto()
    """16 bit floating-point number, equivalent to ``dtype.half``, ``dtype.fp16`` and ``dtype.float16``

    :meta hide-value:
    """

    f32 = auto()
    """32 bit floating-point number, equivalent to ``dtype.float``, ``dtype.fp32`` and ``dtype.float32``

    :meta hide-value:
    """

    f64 = auto()
    """64 bit floating-point number, equivalent to ``dtype.double``, ``dtype.fp64`` and ``dtype.float64``

    :meta hide-value:
    """

    b = auto()
    """Boolean value, equivalent to ``dtype.bool``

    :meta hide-value:
    """

    bf16 = auto()
    """16 bit "Brain" floating-point number, equivalent to ``dtype.bfloat16``

    :meta hide-value:
    """

    f8 = auto()
    """8 bit floating-point number, equivalent to ``dtype.fp8`` and ``dtype.float8``

    :meta hide-value:
    """

    f4 = auto()
    """4 bit floating-point number, equivalent to ``dtype.fp4`` and ``dtype.float4``

    :meta hide-value:
    """

    uint8 = u8
    int8 = i8

    int32 = i32

    long = i64
    int64 = i64

    float8 = f8
    fp8 = f8

    float4 = f4
    fp4 = f4

    half = f16
    fp16 = f16
    float16 = f16

    float = f32
    fp32 = f32
    float32 = f32

    double = f64
    fp64 = f64
    float64 = f64

    bfloat16 = bf16

    @staticmethod
    def _is_np_obj(t: Any) -> bool:
        if isinstance(t, np.dtype):
            return True
        elif isinstance(t, type):
            if issubclass(t, np.generic):
                return True
        return False

    @classmethod
    def _from(
        cls,
        t: Union[torch.dtype, trt.DataType, np.dtype, dtype, type],
        use_default: bool = False,
    ) -> dtype:
        """Create a Torch-TensorRT dtype from another library's dtype system.

        Takes a dtype enum from one of numpy, torch, and tensorrt and create a ``torch_tensorrt.dtype``.
        If the source dtype system is not supported or the type is not supported in Torch-TensorRT,
        then an exception will be raised. As such it is not recommended to use this method directly.

        Alternatively use ``torch_tensorrt.dtype.try_from()``

        Arguments:
            t (Union(torch.dtype, tensorrt.DataType, numpy.dtype, dtype)): Data type enum from another library
            use_default (bool): In some cases a catch all type (such as ``torch_tensorrt.dtype.f32``) is sufficient, so instead of throwing an exception, return default value.

        Returns:
            dtype: Equivalent ``torch_tensorrt.dtype`` to ``t``

        Raises:
            TypeError: Unsupported data type or unknown source

        Examples:

            .. code:: py

                # Succeeds
                float_dtype = torch_tensorrt.dtype._from(torch.float) # Returns torch_tensorrt.dtype.f32

                # Throws exception
                float_dtype = torch_tensorrt.dtype._from(torch.complex128)

        """

        # TODO: Ideally implemented with match statement but need to wait for Py39 EoL
        if isinstance(t, torch.dtype):
            if t == torch.uint8:
                return dtype.u8
            elif t == torch.int8:
                return dtype.i8
            elif t == torch.long:
                return dtype.i64
            elif t == torch.int32:
                return dtype.i32
            elif t == torch.float8_e4m3fn:
                return dtype.f8
            elif t == torch.float4_e2m1fn_x2:
                return dtype.f4
            elif t == torch.half:
                return dtype.f16
            elif t == torch.float:
                return dtype.f32
            elif t == torch.float64:
                return dtype.f64
            elif t == torch.bool:
                return dtype.b
            elif t == torch.bfloat16:
                return dtype.bf16
            elif use_default:
                logging.warning(
                    f"Given dtype that does not have direct mapping to Torch-TensorRT supported types ({t}), defaulting to torch_tensorrt.dtype.float"
                )
                return dtype.float
            else:
                raise TypeError(
                    f"Provided an unsupported data type as a data type for translation (support: bool, int, long, half, float, bfloat16), got: {t}"
                )
        elif isinstance(t, trt.DataType):
            if t == trt.DataType.UINT8:
                return dtype.u8
            elif t == trt.DataType.INT8:
                return dtype.i8
            elif t == trt.DataType.FP8:
                return dtype.f8
            elif t == trt.DataType.INT32:
                return dtype.i32
            elif t == trt.DataType.INT64:
                return dtype.i64
            elif t == trt.DataType.HALF:
                return dtype.f16
            elif t == trt.DataType.FLOAT:
                return dtype.f32
            elif t == trt.DataType.BOOL:
                return dtype.b
            elif t == trt.DataType.BF16:
                return dtype.bf16
            else:
                if is_tensorrt_version_supported("10.8.0") and t == trt.DataType.FP4:
                    return dtype.fp4
                raise TypeError(
                    f"Provided an unsupported data type as a data type for translation (support: bool, int, half, float, bfloat16), got: {t}"
                )

        elif dtype._is_np_obj(t):
            if t == np.uint8:
                return dtype.u8
            elif t == np.int8:
                return dtype.i8
            elif t == np.int32:
                return dtype.i32
            elif t == np.int64:
                return dtype.i64
            elif t == np.float16:
                return dtype.f16
            elif t == np.float32:
                return dtype.f32
            elif t == np.float64:
                return dtype.f64
            elif t == np.bool_:
                return dtype.b
            # TODO: Consider using ml_dtypes when issues like this are resolved:
            # https://github.com/pytorch/pytorch/issues/109873
            # elif t == ml_dtypes.bfloat16:
            #    return dtype.bf16
            elif use_default:
                logging.warning(
                    f"Given dtype that does not have direct mapping to Torch-TensorRT supported types ({t}), defaulting to torch_tensorrt.dtype.float"
                )
                return dtype.float
            else:
                raise TypeError(
                    "Provided an unsupported data type as an input data type (support: bool, int, long, half, float, bfloat16), got: "
                    + str(t)
                )

        elif isinstance(t, dtype):
            return t

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if isinstance(t, _C.dtype):
                if t == _C.dtype.long:
                    return dtype.i64
                elif t == _C.dtype.int32:
                    return dtype.i32
                elif t == _C.dtype.int8:
                    return dtype.i8
                elif t == _C.dtype.half:
                    return dtype.f16
                elif t == _C.dtype.float:
                    return dtype.f32
                elif t == _C.dtype.double:
                    return dtype.f64
                elif t == _C.dtype.bool:
                    return dtype.b
                elif t == _C.dtype.unknown:
                    return dtype.unknown
                else:
                    raise TypeError(
                        f"Provided an unsupported data type as an input data type (support: bool, int32, long, half, float), got: {t}"
                    )
        # else: # commented out for mypy
        raise TypeError(
            f"Provided unsupported source type for dtype conversion (got: {t})"
        )

    @classmethod
    def try_from(
        cls,
        t: Union[torch.dtype, trt.DataType, np.dtype, dtype],
        use_default: bool = False,
    ) -> Optional[dtype]:
        """Create a Torch-TensorRT dtype from another library's dtype system.

        Takes a dtype enum from one of numpy, torch, and tensorrt and create a ``torch_tensorrt.dtype``.
        If the source dtype system is not supported or the type is not supported in Torch-TensorRT,
        then returns ``None``.


        Arguments:
            t (Union(torch.dtype, tensorrt.DataType, numpy.dtype, dtype)): Data type enum from another library
            use_default (bool): In some cases a catch all type (such as ``torch_tensorrt.dtype.f32``) is sufficient, so instead of throwing an exception, return default value.

        Returns:
            Optional(dtype): Equivalent ``torch_tensorrt.dtype`` to ``t`` or ``None``

        Examples:

            .. code:: py

                # Succeeds
                float_dtype = torch_tensorrt.dtype.try_from(torch.float) # Returns torch_tensorrt.dtype.f32

                # Unsupported type
                float_dtype = torch_tensorrt.dtype.try_from(torch.complex128) # Returns None

        """

        try:
            casted_format = dtype._from(t, use_default=use_default)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"Conversion from {t} to torch_tensorrt.dtype failed", exc_info=True
            )
            return None

    def to(
        self,
        t: Union[Type[torch.dtype], Type[trt.DataType], Type[np.dtype], Type[dtype]],
        use_default: bool = False,
    ) -> Union[torch.dtype, trt.DataType, np.dtype, dtype]:
        """Convert dtype into the equivalent type in [torch, numpy, tensorrt]

        Converts ``self`` into one of numpy, torch, and tensorrt equivalent dtypes.
        If  ``self`` is not supported in the target library, then an exception will be raised.
        As such it is not recommended to use this method directly.

        Alternatively use ``torch_tensorrt.dtype.try_to()``

        Arguments:
            t (Union(Type(torch.dtype), Type(tensorrt.DataType), Type(numpy.dtype), Type(dtype))): Data type enum from another library to convert to
            use_default (bool): In some cases a catch all type (such as ``torch.float``) is sufficient, so instead of throwing an exception, return default value.

        Returns:
            Union(torch.dtype, tensorrt.DataType, numpy.dtype, dtype): dtype equivalent ``torch_tensorrt.dtype`` from library enum ``t``

        Raises:
            TypeError: Unsupported data type or unknown target

        Examples:

            .. code:: py

                # Succeeds
                float_dtype = torch_tensorrt.dtype.f32.to(torch.dtype) # Returns torch.float

                # Failure
                float_dtype = torch_tensorrt.dtype.bf16.to(numpy.dtype) # Throws exception

        """

        # TODO: Ideally implemented with match statement but need to wait for Py39 EoL
        if t == torch.dtype:
            if self == dtype.u8:
                return torch.uint8
            elif self == dtype.i8:
                return torch.int8
            elif self == dtype.i32:
                return torch.int
            elif self == dtype.i64:
                return torch.long
            elif self == dtype.f8:
                return torch.float8_e4m3fn
            elif self == dtype.f4:
                return torch.float4_e2m1fn_x2
            elif self == dtype.f16:
                return torch.half
            elif self == dtype.f32:
                return torch.float
            elif self == dtype.f64:
                return torch.double
            elif self == dtype.b:
                return torch.bool
            elif self == dtype.bf16:
                return torch.bfloat16
            elif use_default:
                logging.warning(
                    f"Given dtype that does not have direct mapping to torch ({self}), defaulting to torch.float"
                )
                return torch.float
            else:
                raise TypeError(f"Unsupported torch dtype (had: {self})")

        elif t == trt.DataType:
            if self == dtype.u8:
                return trt.DataType.UINT8
            if self == dtype.i8:
                return trt.DataType.INT8
            elif self == dtype.i32:
                return trt.DataType.INT32
            elif self == dtype.f8:
                return trt.DataType.FP8
            elif self == dtype.i64:
                return trt.DataType.INT64
            elif self == dtype.f16:
                return trt.DataType.HALF
            elif self == dtype.f32:
                return trt.DataType.FLOAT
            elif self == dtype.b:
                return trt.DataType.BOOL
            elif self == dtype.bf16:
                return trt.DataType.BF16
            elif use_default:
                return trt.DataType.FLOAT
            else:
                if is_tensorrt_version_supported("10.8.0") and self == dtype.f4:
                    return trt.DataType.FP4
                raise TypeError("Unsupported tensorrt dtype")

        elif t == np.dtype:
            if self == dtype.u8:
                return np.uint8
            elif self == dtype.i8:
                return np.int8
            elif self == dtype.i32:
                return np.int32
            elif self == dtype.i64:
                return np.int64
            elif self == dtype.f16:
                return np.float16
            elif self == dtype.f4:
                return np.float4_e2m1fn_x2
            elif self == dtype.f32:
                return np.float32
            elif self == dtype.f64:
                return np.float64
            elif self == dtype.b:
                return np.bool_
            # TODO: Consider using ml_dtypes when issues like this are resolved:
            # https://github.com/pytorch/pytorch/issues/109873
            # elif self == dtype.bf16:
            #    return ml_dtypes.bfloat16
            elif use_default:
                return np.float32
            else:
                raise TypeError("Unsupported numpy dtype")

        elif t == dtype:
            return self

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if t == _C.dtype:
                if self == dtype.i64:
                    return _C.dtype.long
                elif self == dtype.i8:
                    return _C.dtype.int8
                elif self == dtype.i32:
                    return _C.dtype.int32
                elif self == dtype.f16:
                    return _C.dtype.half
                elif self == dtype.f32:
                    return _C.dtype.float
                elif self == dtype.f64:
                    return _C.dtype.double
                elif self == dtype.b:
                    return _C.dtype.bool
                elif self == dtype.unknown:
                    return _C.dtype.unknown
                else:
                    raise TypeError(
                        f"Provided an unsupported data type as an input data type (support: bool, int32, long, half, float), got: {self}"
                    )
        # else: # commented out for mypy
        raise TypeError(
            f"Provided unsupported destination type for dtype conversion {t}"
        )

    def try_to(
        self,
        t: Union[Type[torch.dtype], Type[trt.DataType], Type[np.dtype], Type[dtype]],
        use_default: bool,
    ) -> Optional[Union[torch.dtype, trt.DataType, np.dtype, dtype]]:
        """Convert dtype into the equivalent type in [torch, numpy, tensorrt]

        Converts ``self`` into one of numpy, torch, and tensorrt equivalent dtypes.
        If  ``self`` is not supported in the target library, then returns ``None``.

        Arguments:
            t (Union(Type(torch.dtype), Type(tensorrt.DataType), Type(numpy.dtype), Type(dtype))): Data type enum from another library to convert to
            use_default (bool): In some cases a catch all type (such as ``torch.float``) is sufficient, so instead of throwing an exception, return default value.

        Returns:
            Optional(Union(torch.dtype, tensorrt.DataType, numpy.dtype, dtype)): dtype equivalent ``torch_tensorrt.dtype`` from library enum ``t``

        Examples:

            .. code:: py

                # Succeeds
                float_dtype = torch_tensorrt.dtype.f32.to(torch.dtype) # Returns torch.float

                # Failure
                float_dtype = torch_tensorrt.dtype.bf16.to(numpy.dtype) # Returns None

        """

        try:
            casted_format = self.to(t, use_default)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"torch_tensorrt.dtype conversion to target type {t} failed",
                exc_info=True,
            )
            return None

    def __eq__(self, other: Union[torch.dtype, trt.DataType, np.dtype, dtype]) -> bool:
        other_ = dtype._from(other)
        return bool(self.value == other_.value)

    def __hash__(self) -> int:
        return hash(self.value)

    # Putting aliases here that mess with mypy
    bool = b
    int = i32


class memory_format(Enum):
    """"""

    # TensorRT supported memory layouts
    linear = auto()
    """Row major linear format.

    For a tensor with dimensions {N, C, H, W}, the W axis always has unit stride, and the stride of every other axis is at least the product of the next dimension times the next stride. the strides are the same as for a C array with dimensions [N][C][H][W].

    Equivient to ``memory_format.contiguous``

    :meta hide-value:
    """

    chw2 = auto()
    """Two wide channel vectorized row major format.

    This format is bound to FP16 in TensorRT. It is only available for dimensions >= 3.

    For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+1)/2][H][W][2], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c/2][h][w][c%2].

    :meta hide-value:
    """

    hwc8 = auto()
    """Eight channel format where C is padded to a multiple of 8.

    This format is bound to FP16. It is only available for dimensions >= 3.

    For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to the array with dimensions [N][H][W][(C+7)/8*8], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][h][w][c].

    :meta hide-value:
    """

    chw4 = auto()
    """Four wide channel vectorized row major format. This format is bound to INT8. It is only available for dimensions >= 3.

    For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+3)/4][H][W][4], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c/4][h][w][c%4].

    :meta hide-value:
    """

    chw16 = auto()
    """Sixteen wide channel vectorized row major format.

    This format is bound to FP16. It is only available for dimensions >= 3.

    For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+15)/16][H][W][16], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c/16][h][w][c%16].

    :meta hide-value:
    """

    chw32 = auto()
    """Thirty-two wide channel vectorized row major format.

    This format is only available for dimensions >= 3.

    For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+31)/32][H][W][32], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c/32][h][w][c%32].

    :meta hide-value:
    """

    dhwc8 = auto()
    """Eight channel format where C is padded to a multiple of 8.

    This format is bound to FP16, and it is only available for dimensions >= 4.

    For a tensor with dimensions {N, C, D, H, W}, the memory layout is equivalent to an array with dimensions [N][D][H][W][(C+7)/8*8], with the tensor coordinates (n, c, d, h, w) mapping to array subscript [n][d][h][w][c].

    :meta hide-value:
    """

    cdhw32 = auto()
    """Thirty-two wide channel vectorized row major format with 3 spatial dimensions.

    This format is bound to FP16 and INT8. It is only available for dimensions >= 4.

    For a tensor with dimensions {N, C, D, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+31)/32][D][H][W][32], with the tensor coordinates (n, d, c, h, w) mapping to array subscript [n][c/32][d][h][w][c%32].

    :meta hide-value:
    """

    hwc = auto()
    """Non-vectorized channel-last format. This format is bound to FP32 and is only available for dimensions >= 3.

    Equivient to ``memory_format.channels_last``

    :meta hide-value:
    """

    dla_linear = auto()
    """ DLA planar format. Row major format. The stride for stepping along the H axis is rounded up to 64 bytes.

    This format is bound to FP16/Int8 and is only available for dimensions >= 3.

    For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][C][H][roundUp(W, 64/elementSize)] where elementSize is 2 for FP16 and 1 for Int8, with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c][h][w].

    :meta hide-value:
    """

    dla_hwc4 = auto()
    """DLA image format. channel-last format. C can only be 1, 3, 4. If C == 3 it will be rounded to 4. The stride for stepping along the H axis is rounded up to 32 bytes.

    This format is bound to FP16/Int8 and is only available for dimensions >= 3.

    For a tensor with dimensions {N, C, H, W}, with C’ is 1, 4, 4 when C is 1, 3, 4 respectively, the memory layout is equivalent to a C array with dimensions [N][H][roundUp(W, 32/C’/elementSize)][C’] where elementSize is 2 for FP16 and 1 for Int8, C’ is the rounded C. The tensor coordinates (n, c, h, w) maps to array subscript [n][h][w][c].

    :meta hide-value:
    """

    hwc16 = auto()
    """Sixteen channel format where C is padded to a multiple of 16. This format is bound to FP16. It is only available for dimensions >= 3.

    For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to the array with dimensions [N][H][W][(C+15)/16*16], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][h][w][c].

    :meta hide-value:
    """

    dhwc = auto()
    """Non-vectorized channel-last format. This format is bound to FP32. It is only available for dimensions >= 4.

    Equivient to ``memory_format.channels_last_3d``

    :meta hide-value:
    """

    # PyTorch aliases for TRT layouts
    contiguous = linear
    channels_last = hwc
    channels_last_3d = dhwc

    @classmethod
    def _from(
        cls, f: Union[torch.memory_format, trt.TensorFormat, memory_format]
    ) -> memory_format:
        """Create a Torch-TensorRT memory format enum from another library memory format enum.

        Takes a memory format enum from one of torch, and tensorrt and create a ``torch_tensorrt.memory_format``.
        If the source is not supported or the memory format is not supported in Torch-TensorRT,
        then an exception will be raised. As such it is not recommended to use this method directly.

        Alternatively use ``torch_tensorrt.memory_format.try_from()``

        Arguments:
            f (Union(torch.memory_format, tensorrt.TensorFormat, memory_format)): Memory format enum from another library

        Returns:
            memory_format: Equivalent ``torch_tensorrt.memory_format`` to ``f``

        Raises:
            TypeError: Unsupported memory format or unknown source

        Examples:

            .. code:: py

                torchtrt_linear = torch_tensorrt.memory_format._from(torch.contiguous)

        """
        # TODO: Ideally implemented with match statement but need to wait for Py39 EoL
        if isinstance(f, torch.memory_format):
            if f == torch.contiguous_format:
                return memory_format.contiguous
            elif f == torch.channels_last:
                return memory_format.channels_last
            elif f == torch.channels_last_3d:
                return memory_format.channels_last_3d
            else:
                raise TypeError(
                    f"Provided an unsupported memory format for tensor, got: {dtype}"
                )

        elif isinstance(f, trt.DataType):
            if f == trt.TensorFormat.LINEAR:
                return memory_format.linear
            elif f == trt.TensorFormat.CHW2:
                return memory_format.chw2
            elif f == trt.TensorFormat.HWC8:
                return memory_format.hwc8
            elif f == trt.TensorFormat.CHW4:
                return memory_format.chw4
            elif f == trt.TensorFormat.CHW16:
                return memory_format.chw16
            elif f == trt.TensorFormat.CHW32:
                return memory_format.chw32
            elif f == trt.TensorFormat.DHWC8:
                return memory_format.dhwc8
            elif f == trt.TensorFormat.CDHW32:
                return memory_format.cdhw32
            elif f == trt.TensorFormat.HWC:
                return memory_format.hwc
            elif f == trt.TensorFormat.DLA_LINEAR:
                return memory_format.dla_linear
            elif f == trt.TensorFormat.DLA_HWC4:
                return memory_format.dla_hwc4
            elif f == trt.TensorFormat.HWC16:
                return memory_format.hwc16
            elif f == trt.TensorFormat.DHWC:
                return memory_format.dhwc
            else:
                raise TypeError(
                    f"Provided an unsupported tensor format for tensor, got: {dtype}"
                )

        elif isinstance(f, memory_format):
            return f

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if isinstance(f, _C.TensorFormat):
                if f == _C.TensorFormat.contiguous:
                    return memory_format.contiguous
                elif f == _C.TensorFormat.channels_last:
                    return memory_format.channels_last
                else:
                    raise ValueError(
                        "Provided an unsupported tensor format (support: NCHW/contiguous_format, NHWC/channel_last)"
                    )
        # else: # commented out for mypy
        raise TypeError("Provided unsupported source type for memory_format conversion")

    @classmethod
    def try_from(
        cls, f: Union[torch.memory_format, trt.TensorFormat, memory_format]
    ) -> Optional[memory_format]:
        """Create a Torch-TensorRT memory format enum from another library memory format enum.

        Takes a memory format enum from one of torch, and tensorrt and create a ``torch_tensorrt.memory_format``.
        If the source is not supported or the memory format is not supported in Torch-TensorRT,
        then ``None`` will be returned.


        Arguments:
            f (Union(torch.memory_format, tensorrt.TensorFormat, memory_format)): Memory format enum from another library

        Returns:
            Optional(memory_format): Equivalent ``torch_tensorrt.memory_format`` to ``f``

        Examples:

            .. code:: py

                torchtrt_linear = torch_tensorrt.memory_format.try_from(torch.contiguous)

        """
        try:
            casted_format = memory_format._from(f)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"Conversion from {f} to torch_tensorrt.memory_format failed",
                exc_info=True,
            )
            return None

    def to(
        self,
        t: Union[
            Type[torch.memory_format], Type[trt.TensorFormat], Type[memory_format]
        ],
    ) -> Union[torch.memory_format, trt.TensorFormat, memory_format]:
        """Convert ``memory_format`` into the equivalent type in torch or tensorrt

        Converts ``self`` into one of torch or tensorrt equivalent memory format.
        If  ``self`` is not supported in the target library, then an exception will be raised.
        As such it is not recommended to use this method directly.

        Alternatively use ``torch_tensorrt.memory_format.try_to()``

        Arguments:
            t (Union(Type(torch.memory_format), Type(tensorrt.TensorFormat), Type(memory_format))): Memory format type enum from another library to convert to

        Returns:
            Union(torch.memory_format, tensorrt.TensorFormat, memory_format): Memory format equivalent ``torch_tensorrt.memory_format`` in enum ``t``

        Raises:
            TypeError: Unknown target type or unsupported memory format

        Examples:

            .. code:: py

                # Succeeds
                tf = torch_tensorrt.memory_format.linear.to(torch.dtype) # Returns torch.contiguous
        """

        if t == torch.memory_format:
            if self == memory_format.contiguous:
                return torch.contiguous_format
            elif self == memory_format.channels_last:
                return torch.channels_last
            elif self == memory_format.channels_last_3d:
                return torch.channels_last_3d
            else:
                raise TypeError("Unsupported torch dtype")

        elif t == trt.TensorFormat:
            if self == memory_format.linear:
                return trt.TensorFormat.LINEAR
            elif self == memory_format.chw2:
                return trt.TensorFormat.CHW2
            elif self == memory_format.hwc8:
                return trt.TensorFormat.HWC8
            elif self == memory_format.chw4:
                return trt.TensorFormat.CHW4
            elif self == memory_format.chw16:
                return trt.TensorFormat.CHW16
            elif self == memory_format.chw32:
                return trt.TensorFormat.CHW32
            elif self == memory_format.dhwc8:
                return trt.TensorFormat.DHWC8
            elif self == memory_format.cdhw32:
                return trt.TensorFormat.CDHW32
            elif self == memory_format.hwc:
                return trt.TensorFormat.HWC
            elif self == memory_format.dla_linear:
                return trt.TensorFormat.DLA_LINEAR
            elif self == memory_format.dla_hwc4:
                return trt.TensorFormat.DLA_HWC4
            elif self == memory_format.hwc16:
                return trt.TensorFormat.HWC16
            elif self == memory_format.dhwc:
                return trt.TensorFormat.DHWC
            else:
                raise TypeError("Unsupported tensorrt memory format")

        elif t == memory_format:
            return self

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if t == _C.TensorFormat:
                if self == memory_format.contiguous:
                    return _C.TensorFormat.contiguous
                elif self == memory_format.channels_last:
                    return _C.TensorFormat.channels_last
                else:
                    raise ValueError(
                        "Provided an unsupported tensor format (support: NCHW/contiguous_format, NHWC/channel_last)"
                    )
        # else: # commented out for mypy
        raise TypeError(
            "Provided unsupported destination type for memory format conversion"
        )

    def try_to(
        self,
        t: Union[
            Type[torch.memory_format], Type[trt.TensorFormat], Type[memory_format]
        ],
    ) -> Optional[Union[torch.memory_format, trt.TensorFormat, memory_format]]:
        """Convert ``memory_format`` into the equivalent type in torch or tensorrt

        Converts ``self`` into one of torch or tensorrt equivalent memory format.
        If  ``self`` is not supported in the target library, then ``None`` will be returned

        Arguments:
            t (Union(Type(torch.memory_format), Type(tensorrt.TensorFormat), Type(memory_format))): Memory format type enum from another library to convert to

        Returns:
            Optional(Union(torch.memory_format, tensorrt.TensorFormat, memory_format)): Memory format equivalent ``torch_tensorrt.memory_format`` in enum ``t``

        Examples:

            .. code:: py

                # Succeeds
                tf = torch_tensorrt.memory_format.linear.to(torch.dtype) # Returns torch.contiguous
        """

        try:
            casted_format = self.to(t)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"torch_tensorrt.memory_format conversion to target type {t} failed",
                exc_info=True,
            )
            return None

    def __eq__(
        self, other: Union[torch.memory_format, trt.TensorFormat, memory_format]
    ) -> bool:
        other_ = memory_format._from(other)
        return self.value == other_.value

    def __hash__(self) -> int:
        return hash(self.value)


class DeviceType(Enum):
    """Type of device TensorRT will target"""

    UNKNOWN = auto()
    """
    Sentinel value

    :meta hide-value:
    """

    GPU = auto()
    """
    Target is a GPU

    :meta hide-value:
    """

    DLA = auto()
    """
    Target is a DLA core

    :meta hide-value:
    """

    @classmethod
    def _from(cls, d: Union[trt.DeviceType, DeviceType]) -> DeviceType:
        """Create a Torch-TensorRT device type enum from a TensorRT device type enum.

        Takes a device type enum from tensorrt and create a ``torch_tensorrt.DeviceType``.
        If the source is not supported or the device type is not supported in Torch-TensorRT,
        then an exception will be raised. As such it is not recommended to use this method directly.

        Alternatively use ``torch_tensorrt.DeviceType.try_from()``

        Arguments:
            d (Union(tensorrt.DeviceType, DeviceType)): Device type enum from another library

        Returns:
            DeviceType: Equivalent ``torch_tensorrt.DeviceType`` to ``d``

        Raises:
            TypeError: Unknown source type or unsupported device type

        Examples:

            .. code:: py

                torchtrt_dla = torch_tensorrt.DeviceType._from(tensorrt.DeviceType.DLA)

        """
        if isinstance(d, trt.DeviceType):
            if d == trt.DeviceType.GPU:
                return DeviceType.GPU
            elif d == trt.DeviceType.DLA:
                return DeviceType.DLA
            else:
                raise ValueError(
                    "Provided an unsupported device type (support: GPU/DLA)"
                )

        elif isinstance(d, DeviceType):
            return d

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if isinstance(d, _C.DeviceType):
                if d == _C.DeviceType.GPU:
                    return DeviceType.GPU
                elif d == _C.DeviceType.DLA:
                    return DeviceType.DLA
                else:
                    raise ValueError(
                        "Provided an unsupported device type (support: GPU/DLA)"
                    )
        # else: # commented out for mypy
        raise TypeError("Provided unsupported source type for DeviceType conversion")

    @classmethod
    def try_from(cls, d: Union[trt.DeviceType, DeviceType]) -> Optional[DeviceType]:
        """Create a Torch-TensorRT device type enum from a TensorRT device type enum.

        Takes a device type enum from tensorrt and create a ``torch_tensorrt.DeviceType``.
        If the source is not supported or the device type is not supported in Torch-TensorRT,
        then an exception will be raised. As such it is not recommended to use this method directly.

        Alternatively use ``torch_tensorrt.DeviceType.try_from()``

        Arguments:
            d (Union(tensorrt.DeviceType, DeviceType)): Device type enum from another library

        Returns:
            DeviceType: Equivalent ``torch_tensorrt.DeviceType`` to ``d``

        Examples:

            .. code:: py

                torchtrt_dla = torch_tensorrt.DeviceType._from(tensorrt.DeviceType.DLA)

        """
        try:
            casted_format = DeviceType._from(d)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"Conversion from {d} to torch_tensorrt.DeviceType failed",
                exc_info=True,
            )
            return None

    def to(
        self,
        t: Union[Type[trt.DeviceType], Type[DeviceType]],
        use_default: bool = False,
    ) -> Union[trt.DeviceType, DeviceType]:
        """Convert ``DeviceType`` into the equivalent type in tensorrt

        Converts ``self`` into one of torch or tensorrt equivalent device type.
        If  ``self`` is not supported in the target library, then an exception will be raised.
        As such it is not recommended to use this method directly.

        Alternatively use ``torch_tensorrt.DeviceType.try_to()``

        Arguments:
            t (Union(Type(tensorrt.DeviceType), Type(DeviceType))): Device type enum from another library to convert to

        Returns:
            Union(tensorrt.DeviceType, DeviceType): Device type equivalent ``torch_tensorrt.DeviceType`` in enum ``t``

        Raises:
            TypeError: Unknown target type or unsupported device type

        Examples:

            .. code:: py

                # Succeeds
                trt_dla = torch_tensorrt.DeviceType.DLA.to(tensorrt.DeviceType) # Returns tensorrt.DeviceType.DLA
        """

        if t == trt.DeviceType:
            if self == DeviceType.GPU:
                return trt.DeviceType.GPU
            elif self == DeviceType.DLA:
                return trt.DeviceType.DLA
            elif use_default:
                return trt.DeviceType.GPU
            else:
                raise ValueError(
                    "Provided an unsupported device type (support: GPU/DLA)"
                )

        elif t == DeviceType:
            return self

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if t == _C.DeviceType:
                if self == DeviceType.GPU:
                    return _C.DeviceType.GPU
                elif self == DeviceType.DLA:
                    return _C.DeviceType.DLA
                else:
                    raise ValueError(
                        "Provided an unsupported device type (support: GPU/DLA)"
                    )
        # else: # commented out for mypy
        raise TypeError(
            "Provided unsupported destination type for device type conversion"
        )

    def try_to(
        self,
        t: Union[Type[trt.DeviceType], Type[DeviceType]],
        use_default: bool = False,
    ) -> Optional[Union[trt.DeviceType, DeviceType]]:
        """Convert ``DeviceType`` into the equivalent type in tensorrt

        Converts ``self`` into one of torch or tensorrt equivalent memory format.
        If  ``self`` is not supported in the target library, then ``None`` will be returned.

        Arguments:
            t (Union(Type(tensorrt.DeviceType), Type(DeviceType))): Device type enum from another library to convert to

        Returns:
            Optional(Union(tensorrt.DeviceType, DeviceType)): Device type equivalent ``torch_tensorrt.DeviceType`` in enum ``t``

        Examples:

            .. code:: py

                # Succeeds
                trt_dla = torch_tensorrt.DeviceType.DLA.to(tensorrt.DeviceType) # Returns tensorrt.DeviceType.DLA
        """
        try:
            casted_format = self.to(t, use_default=use_default)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"torch_tensorrt.DeviceType conversion to target type {t} failed",
                exc_info=True,
            )
            return None

    def __eq__(self, other: Union[trt.DeviceType, DeviceType]) -> bool:
        other_ = DeviceType._from(other)
        return bool(self.value == other_.value)

    def __hash__(self) -> int:
        return hash(self.value)


class EngineCapability(Enum):
    """
    EngineCapability determines the restrictions of a network during build time and what runtime it targets.
    """

    STANDARD = auto()
    """
    EngineCapability.STANDARD does not provide any restrictions on functionality and the resulting serialized engine can be executed with TensorRT’s standard runtime APIs.

    :meta hide-value:
    """

    SAFETY = auto()
    """
    EngineCapability.SAFETY provides a restricted subset of network operations that are safety certified and the resulting serialized engine can be executed with TensorRT’s safe runtime APIs in the tensorrt.safe namespace.

    :meta hide-value:
    """

    DLA_STANDALONE = auto()
    """
    ``EngineCapability.DLA_STANDALONE`` provides a restricted subset of network operations that are DLA compatible and the resulting serialized engine can be executed using standalone DLA runtime APIs.

    :meta hide-value:
    """

    @classmethod
    def _from(
        cls, c: Union[trt.EngineCapability, EngineCapability]
    ) -> EngineCapability:
        """Create a Torch-TensorRT Engine capability enum from a TensorRT Engine capability enum.

        Takes a device type enum from tensorrt and create a ``torch_tensorrt.EngineCapability``.
        If the source is not supported or the engine capability is not supported in Torch-TensorRT,
        then an exception will be raised. As such it is not recommended to use this method directly.

        Alternatively use ``torch_tensorrt.EngineCapability.try_from()``

        Arguments:
            c (Union(tensorrt.EngineCapability, EngineCapability)): Engine capability enum from another library

        Returns:
            EngineCapability: Equivalent ``torch_tensorrt.EngineCapability`` to ``c``

        Raises:
            TypeError: Unknown source type or unsupported engine capability

        Examples:

            .. code:: py

                torchtrt_ec = torch_tensorrt.EngineCapability._from(tensorrt.EngineCapability.SAFETY)

        """
        if isinstance(c, trt.EngineCapability):
            if c == trt.EngineCapability.STANDARD:
                return EngineCapability.STANDARD
            elif c == trt.EngineCapability.SAFETY:
                return EngineCapability.SAFETY
            elif c == trt.EngineCapability.DLA_STANDALONE:
                return EngineCapability.DLA_STANDALONE
            else:
                raise ValueError("Provided an unsupported engine capability")

        elif isinstance(c, EngineCapability):
            return c

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if isinstance(c, _C.EngineCapability):
                if c == _C.EngineCapability.STANDARD:
                    return EngineCapability.STANDARD
                elif c == _C.EngineCapability.SAFETY:
                    return EngineCapability.SAFETY
                elif c == _C.EngineCapability.DLA_STANDALONE:
                    return EngineCapability.DLA_STANDALONE
                else:
                    raise ValueError("Provided an unsupported engine capability")
        # else: # commented out for mypy
        raise TypeError(
            "Provided unsupported source type for EngineCapability conversion"
        )

    @classmethod
    def try_from(
        c: Union[trt.EngineCapability, EngineCapability],
    ) -> Optional[EngineCapability]:
        """Create a Torch-TensorRT engine capability enum from a TensorRT engine capability enum.

        Takes a device type enum from tensorrt and create a ``torch_tensorrt.EngineCapability``.
        If the source is not supported or the engine capability level is not supported in Torch-TensorRT,
        then an exception will be raised. As such it is not recommended to use this method directly.

        Alternatively use ``torch_tensorrt.EngineCapability.try_from()``

        Arguments:
            c (Union(tensorrt.EngineCapability, EngineCapability)): Engine capability enum from another library

        Returns:
            EngineCapability: Equivalent ``torch_tensorrt.EngineCapability`` to ``c``

        Examples:

            .. code:: py

                torchtrt_safety_ec = torch_tensorrt.EngineCapability._from(tensorrt.EngineCapability.SAEFTY)

        """
        try:
            casted_format = EngineCapability._from(c)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"Conversion from {c} to torch_tensorrt.EngineCapablity failed",
                exc_info=True,
            )
            return None

    def to(
        self, t: Union[Type[trt.EngineCapability], Type[EngineCapability]]
    ) -> Union[trt.EngineCapability, EngineCapability]:
        """Convert ``EngineCapability`` into the equivalent type in tensorrt

        Converts ``self`` into one of torch or tensorrt equivalent engine capability.
        If  ``self`` is not supported in the target library, then an exception will be raised.
        As such it is not recommended to use this method directly.

        Alternatively use ``torch_tensorrt.EngineCapability.try_to()``

        Arguments:
            t (Union(Type(tensorrt.EngineCapability), Type(EngineCapability))): Engine capability enum from another library to convert to

        Returns:
            Union(tensorrt.EngineCapability, EngineCapability): Engine capability equivalent ``torch_tensorrt.EngineCapability`` in enum ``t``

        Raises:
            TypeError: Unknown target type or unsupported engine capability

        Examples:

            .. code:: py

                # Succeeds
                torchtrt_dla_ec = torch_tensorrt.EngineCapability.DLA_STANDALONE.to(tensorrt.EngineCapability) # Returns tensorrt.EngineCapability.DLA
        """
        if t == trt.EngineCapability:
            if self == EngineCapability.STANDARD:
                return trt.EngineCapability.STANDARD
            elif self == EngineCapability.SAFETY:
                return trt.EngineCapability.SAFETY
            elif self == EngineCapability.DLA_STANDALONE:
                return trt.EngineCapability.DLA_STANDALONE
            else:
                raise ValueError("Provided an unsupported engine capability")

        elif t == EngineCapability:
            return self

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if t == _C.EngineCapability:
                if self == EngineCapability.STANDARD:
                    return _C.EngineCapability.STANDARD
                elif self == EngineCapability.SAFETY:
                    return _C.EngineCapability.SAFETY
                elif self == EngineCapability.DLA_STANDALONE:
                    return _C.EngineCapability.DLA_STANDALONE
                else:
                    raise ValueError("Provided an unsupported engine capability")
        # else: # commented out for mypy
        raise TypeError(
            "Provided unsupported destination type for engine capability type conversion"
        )

    def try_to(
        self, t: Union[Type[trt.EngineCapability], Type[EngineCapability]]
    ) -> Optional[Union[trt.EngineCapability, EngineCapability]]:
        """Convert ``EngineCapability`` into the equivalent type in tensorrt

        Converts ``self`` into one of torch or tensorrt equivalent engine capability.
        If  ``self`` is not supported in the target library, then ``None`` will be returned.

        Arguments:
            t (Union(Type(tensorrt.EngineCapability), Type(EngineCapability))): Engine capability enum from another library to convert to

        Returns:
            Optional(Union(tensorrt.EngineCapability, EngineCapability)): Engine capability equivalent ``torch_tensorrt.EngineCapability`` in enum ``t``

        Examples:

            .. code:: py

                # Succeeds
                trt_dla_ec = torch_tensorrt.EngineCapability.DLA.to(tensorrt.EngineCapability) # Returns tensorrt.EngineCapability.DLA_STANDALONE
        """
        try:
            casted_format = self.to(t)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"torch_tensorrt.EngineCapablity conversion to target type {t} failed",
                exc_info=True,
            )
            return None

    def __eq__(self, other: Union[trt.EngineCapability, EngineCapability]) -> bool:
        other_ = EngineCapability._from(other)
        return bool(self.value == other_.value)

    def __hash__(self) -> int:
        return hash(self.value)


class Platform(Enum):
    """
    Specifies a target OS and CPU architecture that a Torch-TensorRT program targets
    """

    LINUX_X86_64 = auto()
    """
    OS: Linux, CPU Arch: x86_64

    :meta hide-value:
    """

    LINUX_AARCH64 = auto()
    """
    OS: Linux, CPU Arch: aarch64

    :meta hide-value:
    """

    WIN_X86_64 = auto()
    """
    OS: Windows, CPU Arch: x86_64

    :meta hide-value:
    """

    UNKNOWN = auto()

    @classmethod
    def current_platform(cls) -> Platform:
        """
        Returns an enum for the current platform Torch-TensorRT is running on

        Returns:
            Platform: Current platform
        """
        import platform

        if platform.system().lower().startswith("linux"):
            # linux
            if platform.machine().lower().startswith("aarch64"):
                return Platform.LINUX_AARCH64
            elif platform.machine().lower().startswith("x86_64"):
                return Platform.LINUX_X86_64

        elif platform.system().lower().startswith("windows"):
            # Windows...
            if platform.machine().lower().startswith("amd64"):
                return Platform.WIN_X86_64

        return Platform.UNKNOWN

    def __str__(self) -> str:
        return str(self.name)

    @needs_torch_tensorrt_runtime  # type: ignore
    def _to_serialized_rt_platform(self) -> str:
        val: str = torch.ops.tensorrt._platform_unknown()

        if self == Platform.LINUX_X86_64:
            val = torch.ops.tensorrt._platform_linux_x86_64()
        elif self == Platform.LINUX_AARCH64:
            val = torch.ops.tensorrt._platform_linux_aarch64()
        elif self == Platform.WIN_X86_64:
            val = torch.ops.tensorrt._platform_win_x86_64()

        return val
