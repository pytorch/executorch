# Partitioner API

The Core ML partitioner API allows for configuration of the model delegation to Core ML. Passing a `CoreMLPartitioner` instance with no additional parameters will run as much of the model as possible on the Core ML backend with default settings. This is the most common use case. For advanced use cases, the partitioner exposes the following options via the [constructor](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/partition/coreml_partitioner.py#L60):


 - `skip_ops_for_coreml_delegation`: Allows you to skip ops for delegation by Core ML.  By default, all ops that Core ML supports will be delegated.  See [here](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/test/test_coreml_partitioner.py#L42) for an example of skipping an op for delegation.
- `compile_specs`: A list of `CompileSpec`s for the Core ML backend.  These control low-level details of Core ML delegation, such as the compute unit (CPU, GPU, ANE), the iOS deployment target, and the compute precision (FP16, FP32).  These are discussed more below.
- `take_over_mutable_buffer`: A boolean that indicates whether PyTorch mutable buffers in stateful models should be converted to [Core ML `MLState`](https://developer.apple.com/documentation/coreml/mlstate).  If set to `False`, mutable buffers in the PyTorch graph are converted to graph inputs and outputs to the Core ML lowered module under the hood.  Generally, setting `take_over_mutable_buffer` to true will result in better performance, but using `MLState` requires iOS >= 18.0, macOS >= 15.0, and Xcode >= 16.0.
- `take_over_constant_data`: A boolean that indicates whether PyTorch constant data like model weights should be consumed by the Core ML delegate.  If set to False, constant data is passed to the Core ML delegate as inputs.  By default, take_over_constant_data=True.
- `lower_full_graph`: A boolean that indicates whether the entire graph must be lowered to Core ML.  If set to True and Core ML does not support an op, an error is raised during lowering.  If set to False and Core ML does not support an op, the op is executed on the CPU by ExecuTorch.  Although setting `lower_full_graph`=False can allow a model to lower where it would otherwise fail, it can introduce performance overhead in the model when there are unsupported ops.  You will see warnings about unsupported ops during lowering if there are any.  By default, `lower_full_graph`=False.


#### Core ML CompileSpec

A list of `CompileSpec`s is constructed with [`CoreMLBackend.generate_compile_specs`](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/compiler/coreml_preprocess.py#L210).  Below are the available options:
- `compute_unit`: this controls the compute units (CPU, GPU, ANE) that are used by Core ML.  The default value is `coremltools.ComputeUnit.ALL`.  The available options from coremltools are:
    - `coremltools.ComputeUnit.ALL` (uses the CPU, GPU, and ANE)
    - `coremltools.ComputeUnit.CPU_ONLY` (uses the CPU only)
    - `coremltools.ComputeUnit.CPU_AND_GPU` (uses both the CPU and GPU, but not the ANE)
    - `coremltools.ComputeUnit.CPU_AND_NE` (uses both the CPU and ANE, but not the GPU)
- `minimum_deployment_target`: The minimum iOS deployment target (e.g., `coremltools.target.iOS18`).  By default, the smallest deployment target needed to deploy the model is selected.  During export, you will see a warning about the "Core ML specification version" that was used for the model, which maps onto a deployment target as discussed [here](https://apple.github.io/coremltools/mlmodel/Format/Model.html#model).  If you need to control the deployment target, please specify it explicitly.
- `compute_precision`: The compute precision used by Core ML (`coremltools.precision.FLOAT16` or `coremltools.precision.FLOAT32`).  The default value is `coremltools.precision.FLOAT16`.  Note that the compute precision is applied no matter what dtype is specified in the exported PyTorch model.  For example, an FP32 PyTorch model will be converted to FP16 when delegating to the Core ML backend by default.  Also note that the ANE only supports FP16 precision.
- `model_type`: Whether the model should be compiled to the Core ML [mlmodelc format](https://developer.apple.com/documentation/coreml/downloading-and-compiling-a-model-on-the-user-s-device) during .pte creation ([`CoreMLBackend.MODEL_TYPE.COMPILED_MODEL`](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/compiler/coreml_preprocess.py#L71)), or whether it should be compiled to mlmodelc on device ([`CoreMLBackend.MODEL_TYPE.MODEL`](https://github.com/pytorch/executorch/blob/14ff52ff89a89c074fc6c14d3f01683677783dcd/backends/apple/coreml/compiler/coreml_preprocess.py#L70)).  Using `CoreMLBackend.MODEL_TYPE.COMPILED_MODEL` and doing compilation ahead of time should improve the first time on-device model load time.

### Dynamic and Enumerated Shapes in Core ML Export

When exporting an `ExportedProgram` to Core ML, **dynamic shapes** are mapped to [`RangeDim`](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html#set-the-range-for-each-dimension).
This enables Core ML `.pte` files to accept inputs with varying dimensions at runtime.

⚠️ **Note:** The Apple Neural Engine (ANE) does not support true dynamic shapes.    If a model relies on `RangeDim`, Core ML will fall back to scheduling the model on the CPU or GPU instead of the ANE.

---

#### Enumerated Shapes

To enable limited flexibility on the ANE—and often achieve better performance overall—you can export models using **[enumerated shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html#select-from-predetermined-shapes)**.

- Enumerated shapes are *not fully dynamic*.
- Instead, they define a **finite set of valid input shapes** that Core ML can select from at runtime.
- This approach allows some adaptability while still preserving ANE compatibility.

---

#### Specifying Enumerated Shapes

Unlike `RangeDim`, **enumerated shapes are not part of the `ExportedProgram` itself.**
They must be provided through a compile spec.

For reference on how to do this, see:
- The annotated code snippet below, and
- The [end-to-end test in ExecuTorch](https://github.com/pytorch/executorch/blob/main/backends/apple/coreml/test/test_enumerated_shapes.py), which demonstrates how to specify enumerated shapes during export.


```python
class Model(torch.nn.Module):
        def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 5)
                self.linear2 = torch.nn.Linear(11, 5)

        def forward(self, x, y):
            return self.linear1(x).sum() + self.linear2(y)

model = Model()
example_inputs = (
    torch.randn((4, 6, 10)),
    torch.randn((5, 11)),
)

# Specify the enumerated shapes.  Below we specify that:
#
# * x can take shape [1, 5, 10] and y can take shape [3, 11], or
# * x can take shape [4, 6, 10] and y can take shape [5, 11]
#
# Any other input shapes will result in a runtime error.
#
# Note that we must export x and y with dynamic shapes in the ExportedProgram
# because some of their dimensions are dynamic
enumerated_shapes = {"x": [[1, 5, 10], [4, 6, 10]], "y": [[3, 11], [5, 11]]}
dynamic_shapes = [
    {
        0: torch.export.Dim.AUTO(min=1, max=4),
        1: torch.export.Dim.AUTO(min=5, max=6),
    },
    {0: torch.export.Dim.AUTO(min=3, max=5)},
]
ep = torch.export.export(
    model.eval(), example_inputs, dynamic_shapes=dynamic_shapes
)

# If enumerated shapes are specified for multiple inputs, we must export
# for iOS18+
compile_specs = CoreMLBackend.generate_compile_specs(
    minimum_deployment_target=ct.target.iOS18
)
compile_specs.append(
    CoreMLBackend.generate_enumerated_shapes_compile_spec(
        ep,
        enumerated_shapes,
    )
)

# When using an enumerated shape compile spec, you must specify lower_full_graph=True
# in the CoreMLPartitioner.  We do not support using enumerated shapes
# for partially exported models
partitioner = CoreMLPartitioner(
    compile_specs=compile_specs, lower_full_graph=True
)
delegated_program = executorch.exir.to_edge_transform_and_lower(
    ep,
    partitioner=[partitioner],
)
et_prog = delegated_program.to_executorch()
```
