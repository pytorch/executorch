import serializer.tosa_serializer as ts
from executorch.backends.arm import tosa_mapping

from serializer.tosa_serializer import TosaOp


def getNodeArgs(node):
    return [tosa_mapping.TosaArg(arg) for arg in node.args]


"""
Helper transpose function to match TOSA's shape requirements
 E.g., TOSA 0.80.0 specification - 2.3.3 CONV2D shapes:
 https://www.mlplatform.org/tosa/tosa_spec.html#_conv2d
"""


def buildTranspose(tosa_fb, input_name, input_shape, new_order, out_dtype):
    # Check new_order's length is equal to input rank
    assert len(input_shape) == len(new_order), "Wrong shape order length"

    # Check no duplications
    assert len(set(new_order)) == len(new_order), "Contain duplicated dim numbers"

    # Check all dims are valid
    for idx in new_order:
        if idx < 0:
            assert True, "Negative dim number"
        elif idx >= len(input_shape):
            assert True, "Dim is greater than input rank"

    input_shape_transpoed = [input_shape[i] for i in new_order]
    attr = ts.TosaSerializerAttribute()
    attr.TransposeAttribute(new_order)
    input_transposed = tosa_fb.addIntermediate(input_shape_transpoed, out_dtype)
    tosa_fb.addOperator(
        TosaOp.Op().TRANSPOSE, [input_name], [input_transposed.name], attr
    )
    return input_transposed


""" TOSA reshape returns a tensor with the same type/values as the input.
    No data conversion happens during a reshape operation. """


def buildReshape(tosa_fb, input_name, new_shape, output_name):
    attr = ts.TosaSerializerAttribute()
    attr.ReshapeAttribute(new_shape)
    tosa_fb.addOperator(TosaOp.Op().RESHAPE, [input_name], [output_name], attr)
