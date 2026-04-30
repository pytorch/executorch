from typing import Optional

import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import set_layer_name


def convert_unary(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    operation_type: trt.UnaryOperation,
    input_val: TRTTensor,
) -> TRTTensor:
    """
    Add a TensorRT Unary layer to `network`.

    Args:
        ctx (ConversionContext): TensorRT ConversionContext object.
        input_val (TRTTensor): Input to the unary op. Must be a TensorRT tensor.
        op_type (trt.ElementWiseOperation): Type of the TensorRT unary operation.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT Unary layer.
    """
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{operation_type} received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    layer = ctx.net.add_unary(input_val, operation_type)
    set_layer_name(layer, target, name, source_ir)
    output = layer.get_output(0)
    kind: str = str(target.__name__) if callable(target) else target
    output.name = output.name + "_" + kind
    return layer.get_output(0)
