#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    types

Module contains helper functions that work with TFLite data types.
"""
from enum import Enum

import executorch.backends.nxp.backend.ir.logger as logger

import flatbuffers as fb
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType

# Lists of types. Used to simplify specification of supported types in conversion modules.
FLOATS = [TensorType.FLOAT16, TensorType.FLOAT32, TensorType.FLOAT64]
INTS = [TensorType.INT8, TensorType.INT16, TensorType.INT32, TensorType.INT64]
UINTS = [TensorType.UINT8, TensorType.UINT16, TensorType.UINT32, TensorType.UINT64]
ALL_TYPES = (
    FLOATS
    + INTS
    + UINTS
    + [TensorType.STRING, TensorType.BOOL, TensorType.COMPLEX64, TensorType.COMPLEX128]
)


class TensorFlowDataType(Enum):
    # The DataType enum used internally by TensorFlow.
    # https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/core/framework/types.proto#L13-L87

    DT_INVALID = 0
    DT_FLOAT = 1
    DT_DOUBLE = 2
    DT_INT32 = 3
    DT_UINT8 = 4
    DT_INT16 = 5
    DT_INT8 = 6
    DT_STRING = 7
    DT_COMPLEX64 = 8
    DT_INT64 = 9
    DT_BOOL = 10
    DT_QINT8 = 11
    DT_QUINT8 = 12
    DT_QINT32 = 13
    DT_BFLOAT16 = 14
    DT_QINT16 = 15
    DT_QUINT16 = 16
    DT_UINT16 = 17
    DT_COMPLEX128 = 18
    DT_HALF = 19
    DT_RESOURCE = 20
    DT_VARIANT = 21
    DT_UINT32 = 22
    DT_UINT64 = 23
    DT_FLOAT8_E5M2 = 24
    DT_FLOAT8_E4M3FN = 25
    DT_INT4 = 29
    DT_UINT4 = 30

    DT_FLOAT_REF = 101
    DT_DOUBLE_REF = 102
    DT_INT32_REF = 103
    DT_UINT8_REF = 104
    DT_INT16_REF = 105
    DT_INT8_REF = 106
    DT_STRING_REF = 107
    DT_COMPLEX64_REF = 108
    DT_INT64_REF = 109
    DT_BOOL_REF = 110
    DT_QINT8_REF = 111
    DT_QUINT8_REF = 112
    DT_QINT32_REF = 113
    DT_BFLOAT16_REF = 114
    DT_QINT16_REF = 115
    DT_QUINT16_REF = 116
    DT_UINT16_REF = 117
    DT_COMPLEX128_REF = 118
    DT_HALF_REF = 119
    DT_RESOURCE_REF = 120
    DT_VARIANT_REF = 121
    DT_UINT32_REF = 122
    DT_UINT64_REF = 123
    DT_FLOAT8_E5M2_REF = 124
    DT_FLOAT8_E4M3FN_REF = 125
    DT_INT4_REF = 129
    DT_UINT4_REF = 130


def is_unsigned(data_type: TensorType) -> bool:
    return data_type in {
        TensorType.UINT8,
        TensorType.UINT16,
        TensorType.UINT32,
        TensorType.UINT64,
    }


def is_signed(data_type: TensorType) -> bool:
    return data_type in {
        TensorType.INT8,
        TensorType.INT16,
        TensorType.INT32,
        TensorType.INT64,
    }


def name_for_type(data_type: TensorType) -> str:
    """Return the name of given TFLite data type."""
    names = [
        "FLOAT32",
        "FLOAT16",
        "INT32",
        "UINT8",
        "INT64",
        "STRING",
        "BOOL",
        "INT16",
        "COMPLEX64",
        "INT8",
        "FLOAT64",
        "COMPLEX128",
        "UINT64",
        "RESOURCE",
        "VARIANT",
        "UINT32",
        "UINT16",
        "INT4",
    ]

    return names[data_type]


def type_size(data_type: TensorType):
    """Return the memory size in bytes of given TFLite data type."""
    if data_type in {TensorType.UINT8, TensorType.INT8}:
        return 1
    elif data_type in {TensorType.UINT16, TensorType.INT16, TensorType.FLOAT16}:
        return 2
    elif data_type in {TensorType.UINT32, TensorType.INT32, TensorType.FLOAT32}:
        return 4
    elif data_type in {
        TensorType.UINT64,
        TensorType.INT64,
        TensorType.FLOAT64,
        TensorType.COMPLEX64,
    }:
        return 8
    elif data_type in {TensorType.COMPLEX128}:
        return 16

    logger.e(
        logger.Code.INTERNAL_ERROR,
        f"Unexpected type '{data_type}' in types.type_size().",
    )


def prepend_function(builder: fb.Builder, data_type: TensorType):  # noqa C901
    """Return the flatbuffer 'Prepend<type>()' function for given type."""
    if data_type == TensorType.UINT8:
        return builder.PrependUint8
    elif data_type == TensorType.UINT16:
        return builder.PrependUint16
    elif data_type == TensorType.UINT32:
        return builder.PrependUint32
    elif data_type == TensorType.UINT64:
        return builder.PrependUint64

    elif data_type == TensorType.INT8:
        return builder.PrependInt8
    elif data_type == TensorType.INT16:
        return builder.PrependInt16
    elif data_type == TensorType.INT32:
        return builder.PrependInt32
    elif data_type == TensorType.INT64:
        return builder.PrependInt64

    elif data_type == TensorType.FLOAT16:
        logger.w(
            "Flatbuffer prepend function for FLOAT16 datatype is not supported! Using default 16b alternative."
        )
        return builder.PrependInt16  # Might not work
    elif data_type == TensorType.FLOAT32:
        return builder.PrependFloat32
    elif data_type == TensorType.FLOAT64:
        return builder.PrependFloat64

    elif data_type == TensorType.BOOL:
        return builder.PrependBool

    logger.e(
        logger.Code.NOT_IMPLEMENTED,
        f"Unsupported flatbuffer prepend function for type '{data_type}'!",
    )
