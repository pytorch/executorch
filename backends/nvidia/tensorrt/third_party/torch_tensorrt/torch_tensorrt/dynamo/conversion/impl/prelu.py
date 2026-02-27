from typing import Optional

from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import set_layer_name


def prelu(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: TRTTensor,
) -> TRTTensor:
    layer = ctx.net.add_parametric_relu(input, weight)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
