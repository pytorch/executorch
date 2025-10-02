
import torch

from torchao.quantization.granularity import PerGroup, PerAxis
from torchao.quantization.quant_api import (
    IntxWeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    quantize_,
)
from torchao.utils import unwrap_tensor_subclass
from torch.export import export, ExportedProgram
from executorch.exir import (
    EdgeProgramManager,
    ExecutorchBackendConfig,
    ExecutorchProgramManager,
)
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackFloatingPointPartitioner,
    XnnpackPartitioner,
)
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    to_edge_transform_and_lower,
)

# Note: I think this works fine.
# Quantize embeddings with 8-bits, per channel
# embedding_config = IntxWeightOnlyConfig(
#     weight_dtype=torch.int8,
#     granularity=PerAxis(0),
# )
# qunatize_(
#     eager_model,
#     lambda m, fqn: isinstance(m, torch.nn.Embedding),
# )

torch.manual_seed(0)

class ModuleLinear(torch.nn.Module):
    def __init__(
        self,
        in_size: int = 2,
        input_channels: int = 4,
        output_channels: int = 4,
        dtype: torch.dtype = torch.float,
        use_bias: bool = False
    ):
        super().__init__()
        self.linear = torch.nn.Linear(
            input_channels, output_channels, bias=use_bias
        ).to(dtype=dtype)

        self.ic = input_channels
        self.oc = output_channels
        assert dtype in [torch.float, torch.half], "Unsupported op dtype"
        self.op_dtype = dtype
        self.in_size = in_size

    def forward(self, x: torch.Tensor):
        return self.linear(x)

    def get_random_inputs(self):
        inp = torch.randn(self.in_size, self.ic).to(self.op_dtype)
        return (inp,)

### EAGER.
eager_model = ModuleLinear(
    in_size=1,
    input_channels=32,
    output_channels=2,
)

test_inputs = eager_model.get_random_inputs()
eager_result = eager_model(*test_inputs)
print("eager result: ", eager_result)

### QUANTIZE.
# Quatize linear layers with 8-bit dynamic activations and 4-bit weights
linear_config = Int8DynamicActivationIntxWeightConfig(
    weight_dtype=torch.int4,
    weight_granularity=PerGroup(32),
)
# NOTE: comment this out, and program-data separation works well.
quantize_(eager_model, linear_config)

quantized_result = eager_model(*test_inputs)
print("quantized results: ", quantized_result)
print(torch.allclose(eager_result, quantized_result, atol=1e-1))

unwrap_tensor_subclass(eager_model)
unwrapped_result = eager_model(*test_inputs)
print("unwrapped results: ", unwrapped_result)
print(torch.allclose(quantized_result, unwrapped_result, atol=1e-3))

from executorch.exir.passes.external_constants_pass import (
    delegate_external_constants_pass_unlifted,
)
### EXPORT AND TAG WEIGHTS.
ep1 = export(eager_model, test_inputs, dynamic_shapes=None, strict=True)
exported_result = ep1.module()(*test_inputs)
print("exported program: ", exported_result)
print(torch.allclose(quantized_result, exported_result, atol=1e-3))
print("Graph: ")
ep1.graph_module.print_readable()
# Tag the unlifted ep.module().
tagged_module = ep1.module()
delegate_external_constants_pass_unlifted(
    module=tagged_module,
    gen_tag_fn=lambda x: "model", # This is the filename the weights will be saved to. In this case, weights will be saved as "model.ptd"
)

### RE-EXPORT.
ep = export(tagged_module, test_inputs, dynamic_shapes=None, strict=True)
exported_result = ep.module()(*test_inputs)
print("exported program (after tagging): ", exported_result)
print(torch.allclose(quantized_result, exported_result, atol=1e-3))
# Check tagged nodes:
for node in list(ep.graph.nodes):
    if 'custom' in node.meta:
        print(f"Node: {node.name}, meta: {node.meta['custom']}")

## TO_EDGE_TRANSFORM_AND_LOWER.
DynamicallyQuantizedPartitioner = XnnpackPartitioner(
    config_precisions=ConfigPrecisionType.DYNAMIC_QUANT,
    per_op_mode=True,
)
edge = to_edge_transform_and_lower(
    ep,
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
    partitioner=[XnnpackPartitioner()],
    generate_etrecord=False,
)
# ^ after this, the graph has a single node? torchao_dequantize_affine_default
edge_result = edge.exported_program().module()(*test_inputs)
print("edge program: ", edge_result)
print(torch.allclose(quantized_result, edge_result, atol=1e-3))
edge.exported_program().graph_module.print_readable()

### TO_EXECUTORCH.
exec = edge.to_executorch(ExecutorchBackendConfig())

### SAVE ET MODEL TO DISK.
with open("model.pte", "wb") as f:
    f.write(exec.buffer)

if len(exec._tensor_data) == 0: 
    print("No external data saved")
else:
    exec.write_tensor_data_to_file(".")

### LOAD AND RUN VIA PYBINDINGS.
import torch
from executorch.extension.pybindings import portable_lib
module = portable_lib._load_for_executorch("model.pte", "model.ptd")
exec_result = module.forward(test_inputs)
# Expecting key: a4f6ff98c9db8ecfe5c11e87d07f182a58cb9696f01086d9e0cdc2e986fab003
# Scale: a4f6ff98c9db8ecfe5c11e87d07f182a58cb9696f01086d9e0cdc2e986fab003
print("executorch program: ", exec_result)
print(torch.allclose(quantized_result, exec_result[0], atol=1e-3))

print("End.")
