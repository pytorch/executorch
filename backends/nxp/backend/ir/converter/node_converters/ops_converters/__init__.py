from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.abs_converter import (
    AbsConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.adaptive_avg_pool_2d_converter import (
    AdaptiveAvgPool2dConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.add_tensor_converter import (
    AddTensorConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.addmm_converter import (
    AddMMConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.avg_pool_2d_converter import (
    AvgPool2dConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.cat_converter import (
    CatConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.clone_converter import (
    CloneConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.constant_pad_nd_converter import (
    ConstantPadNDConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.convolution_converter import (
    ConvolutionConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.hardtanh_converter import (
    HardTanhConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.max_pool_2d_converter import (
    MaxPool2dConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.mean_dim_converter import (
    MeanDimConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.mm_converter import (
    MMConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.permute_copy_converter import (
    PermuteCopyConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.qdq_dequantize_converter import (
    QDQPerChannelDequantizeConverter,
    QDQPerTensorDequantizeConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.qdq_quantize_converter import (
    QDQQuantizeConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.relu_converter import (
    ReLUConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.sigmoid_converter import (
    SigmoidConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.softmax_converter import (
    SoftmaxConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.sub_tensor_converter import (
    SubTensorConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.tanh_converter import (
    TanhConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.view_copy_converter import (
    ViewCopyConverter,
)

__all__ = [
    "AddMMConverter",
    "CatConverter",
    "ConvolutionConverter",
    "MMConverter",
    "PermuteCopyConverter",
    "SoftmaxConverter",
    "ViewCopyConverter",
    "QDQPerTensorDequantizeConverter",
    "QDQPerChannelDequantizeConverter",
    "QDQQuantizeConverter",
    "ConstantPadNDConverter",
    "ReLUConverter",
    "MeanDimConverter",
    "MaxPool2dConverter",
    "AvgPool2dConverter",
    "AddTensorConverter",
    "SubTensorConverter",
    "CloneConverter",
    "AbsConverter",
    "AdaptiveAvgPool2dConverter",
    "HardTanhConverter",
    "SigmoidConverter",
    "TanhConverter",
]
