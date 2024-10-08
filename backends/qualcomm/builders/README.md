# Contribution for More Operators
Thank you for contributing to Qualcomm AI Engine Direct delegate for ExecuTorch. Reading and following these guidelines will help you quickly get the essentials of implementing operator builder to unblock yourself and land pull requests more efficiently.

## Sections
* [References](#references)
* [Getting Started](#getting-started)
    * [Identify Unsupported Operator](#identify-unsupported-operator)
    * [Check Operator Spec](#check-operator-spec)
    * [Implementation](#implementation)
    * [Quantizer Annotation](#quantizer-annotation)
* [Issues](#issues)
* [Pull Requests](#pull-requests)

## References
### Qualcomm AI Engine Direct
- [Operator Definitions](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/MasterOpDef.html)
- [Supported Operators in Backends](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/operations.html#backend-supplements)

### PyTorch
- [torch.nn Operator Definitions](https://pytorch.org/docs/stable/nn.html)
- [torch.nn.functional Operator Definitions](https://pytorch.org/docs/stable/nn.functional.html)
- [ATen Operator Definitions](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native)

## Getting Started
### Identify Unsupported Operator
Consider we're enabling following model:
```python
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm([768], eps=1e-6)
        self.linear = torch.nn.Linear(768, 100)

    def forward(self, x):
        return self.linear(self.layer_norm(x))
```
At the time we try to lower it with Qualcomm backend:
```python
from excutorch.examples.qualcomm.utils import build_executorch_binary

build_executorch_binary(
    model=MyModel(),
    inputs=(torch.randn(200, 768),),
    soc_model="SM8650"
    file_name="my_model",
    dataset=None,
)
```
Assume there is no `torch.nn.LayerNorm` support, you should see the following error logs:
```bash
File "/executorch/backends/qualcomm/partition/qnn_partitioner.py", line 77, in is_node_supported
    op_wrapper = self.node_visitors[node.target.__name__].define_node(
KeyError: 'aten.native_layer_norm.default'
```
This log comes straight to the point, there is no suitable conversion for delegating torch operator to Qualcomm AI Engine Direct. Where the `node_visitors` is a dictionary which maps operator target name with its implementation callback. The goal of this tutorial aims for helping you register the missing one.<br/>
The very first step is to locate which operator type are we going to support. Sometimes the target name of operator might be obscure, following snippet could help you trace back by its call stack:
```python
from executorch.backends.qualcomm.utils.utils import capture_program

prog = capture_program(MyModel(), (torch.randn(200, 768),))
for node in prog.exported_program.graph.nodes:
    if node.op == "call_function" and node.target.__name__ == 'aten.native_layer_norm.default':
        print(node.meta["source_fn_stack"])
```
It will provide more hint to the source PyTorch layer where the missing operator maps to:
```bash
[('l__self___layer_norm', <class 'torch.nn.modules.normalization.LayerNorm'>)]
```

### Check Operator Spec
- **Qualcomm AI Engine Direct**:<br/>
    You could collect information of `LayerNorm`'s IO via documents mentioned in [Qualcomm AI Engine Direct Manual](#qualcomm-ai-engine-direct):
    * inputs
        - in[0] - input activation / required
        - in[1] - gamma / optional
        - in[2] - beta / optional
    * parameters
        - "epsilon" / optional
        - "axes" / required
    * outputs
        - out[0] - output activation / required

    The required tensors must be provided for no default values were given inside QNN runtime, The order of IOs (`input activation`, `gamma`, `beta`) matters compared to parameters (`epsilon`, `axes`) who are recognized by literal value:
    ```c
    typedef struct {
        /// A human-readable name for the operation instance.
        const char* name;
        /// The name of the operation package to which this operation's type belongs.
        const char* packageName;
        /// The name of operation type (e.g. Conv2D).
        const char* typeName;
        /// The number of static parameters provided in the params array.
        uint32_t numOfParams;
        /// Array of operation parameters.
        Qnn_Param_t* params;
        /// The number of input tensors.
        uint32_t numOfInputs;
        /// Array of input tensors.
        Qnn_Tensor_t* inputTensors;
        /// The number of output tensors.
        uint32_t numOfOutputs;
        /// Array of output tensors.
        Qnn_Tensor_t* outputTensors;
    } Qnn_OpConfigV1_t;
    ```
    This is a data structure used to check operator validity in QNN SDK. Inside validation process, tensors are retrieved sequentially and passed through a series of spec examinations while parameters are matched by their names:
    ```c
    typedef struct {
        /// Parameter type: scalar or tensor
        Qnn_ParamType_t paramType;
        /// Name of the parameter
        const char* name;

        union UNNAMED {
            /// Scalar parameter specification
            Qnn_Scalar_t scalarParam;
            /// Tensor parameter specification; tensors referred to must be STATIC.
            Qnn_Tensor_t tensorParam;
        };
    } Qnn_Param_t;
    ```
    The name value equals to the parameter name described in [Operator Definitions](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/MasterOpDef.html), there are `epsilon`, `axes` for `LayerNorm` case.<br/>

    If you find it hard to correlate missing operator with documentation, this [table](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/SupportedOps.html) might be helpful for searching. In some cases, an exact match may not exist. Consider seeking for a math equivalent approach or notify maintainer for further analysis.

- **PyTorch**:<br/>
    We could also read the IO spec from [function declaration](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/layer_norm.cpp) mentioned in [PyTorch Documentation](#pytorch):
    * inputs
        - in[0] - input activation / required
        - in[1] - normalized_shape / required
        - in[2] - weight_opt / optional
        - in[3] - bias_opt / optional
        - in[4] - eps / required

    Through comparing the [equation](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html), we could sort out the relevance of arguments (`gamma` / `beta` / `epsilon`) inside Qualcomm manual to PyTorch (`weight_opt` / `bias_opt` / `eps`). The unmatched parameter `axes` will have more discussions in the [implementation](#implementation) part.

### Implementation
Let's start with adding new definition in `qnn_constant.py` for `LayerNorm` operator.
```python
@dataclass(init=False, frozen=True)
class OpHardSwish:
    ...

# please insert it in alphabetically order
@dataclass(init=False, frozen=True)
class OpLayerNorm:
    op_name: str = "LayerNorm"
    param_epsilon = "epsilon"
    param_axes = "axes"


@dataclass(init=False, frozen=True)
class OpLogSoftmax:
    ...
```
The conventions are:
- op_name: string describing the operator
- params_xxx: string for consumed parameters

The content should have exact match with literal values mentioned in [Qualcomm AI Engine Direct Manual](#qualcomm-ai-engine-direct) or `QnnOpDef.h` under `$QNN_SDK_ROOT/include/QNN/`:
```c
#define QNN_OP_LAYER_NORM               "LayerNorm"
#define QNN_OP_LAYER_NORM_PARAM_EPSILON "epsilon"
#define QNN_OP_LAYER_NORM_PARAM_AXES    "axes"
```

Next, create a new file with name in snake case format (e.g. `op_layer_norm.py`) and import required modules (please check comments for getting the ideas of usage):
```python
# pybind interface for invoking QNN APIs
import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
# tensors or other numerics will be shipped in numpy format
import numpy as np
import torch
# common keywords of Qualcomm backend
from executorch.backends.qualcomm.utils.constants import QCOM_DATA
# op builder will inherit NodeVisitor and have its own implementation
# register_node_visitor for book-keeping the dictionary of target name v.s. callback
from .node_visitor import NodeVisitor, register_node_visitor
# the definitions required to build operator in QNN
from .qnn_constants import OpLayerNorm, QNN_OP_PACKAGE_NAME_QTI_AISW
# utility to get parameter value when creating tensor in QNN
from .utils import get_parameter
```
Start with function declaration as:
```python
@register_node_visitor
class LayerNormVisitor(NodeVisitor):
    target = ["aten.native_layer_norm.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
```
It's mandatory to have `target` member in list form, since there would have multiple targets map to the same implementation. e.g. `aten.leaky_relu.default`, `aten.prelu.default` have similar equations but only differ in negative slope.<br/>
The `nodes_to_wrappers` is a dictionary maintaining relationship between graph node and its output tensor. `nodes_to_wrappers` acts as an memo for not creating tensor objects to nodes that have already been traversed.<br/>

Now, we can start to fill in function body step by step:
1. Define input activation tensors:
    ```python
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )
    ```
    Through the information in [Check Operator Spec](#check-operator-spec) section, we could easily extract the desired nodes.<br/>
    The `get_tensor` method is responsible for retrieving torch tensor in correct axis order if `layout_transform` pass happened to apply.<br/>
    The `define_tensor` method is for generating tensor object for QNN API and will be memorized by aforementioned `node_to_wrappers`.<br/>
    And yet, there are arguments worth for addressing more:
    - **node**: current graph node
    - **tensor**: torch tensor emitted by node
    - **tensor_type**: type compatible with QNN SDK, oftenly use `QNN_TENSOR_TYPE_NATIVE` for intermediate outputs and `QNN_TENSOR_TYPE_STATIC` for constant parameters
    - **nodes_to_wrappers**: dictionary of graph node and its output tensor (note: the tensor here is not a torch tensor but a wrapped object for QNN)
    - **is_input_tensor**: flag to tell if current tensor is input activation or parameter, which is important for fixed point mixed-precision to work properly
    - **node_name**: (optional) tensor name for user to specify
    - **wrapper_idx**: (optional) defaults to zero if node is not a tuple, otherwise it acts as an indexer to output tensors. e.g. when slicing input tensor into multiple outputs, `wrapper_idx` is necessary for getting correct wrapped tensor object

2. Define input gamma / beta tensors:
    ```python
        weight_node = node.args[2]
        weight_tensor = get_parameter(weight_node, self.edge_program)
        weight_tensor_wrapper = self.define_tensor(
            weight_node,
            weight_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            is_input_tensor=False,
        )

        bias_node = node.args[3]
        bias_tensor = get_parameter(bias_node, self.edge_program)
        bias_tensor_wrapper = self.define_tensor(
            bias_node,
            bias_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            is_input_tensor=False,
        )
    ```
    The logic should be similar and straightforward. Please carefully set arguments `tensor_type`, `is_input_tensor` according to tensors' property.

3. Define parameters:
    ```python
        normalized_shapes = node.args[1]
        if len(normalized_shapes) != 1:
            print("QNN only supports normalized output with rank 1")
            return

        axes = [len(input_tensor.shape) - 1]
        axes_shape = [len(axes)]
        epsilon = node.args[4]
    ```
    Here you can see the constraint introduced by Qualcomm AI Engine Direct. Unlike PyTorch's LayerNorm operator, QNN can only normalize input into 1-D tensor. Therefore we will have log to remind user and return the program directly, this gesture will be considered as validation failure in partitioner and will fallback this operator to CPU.<br/>
    When passing tensor type parameters via pybind interface, it's also required to ship extra information like tensor shape in list form. e.g. `axes_shape = [len(axes)]`. More details will be provided in coming steps.

4. Define output tensor:
    ```python
        output_tensor = self.get_tensor(node, node, 0)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )
    ```
    Althought the input / output activations might map to the graph IOs (a.k.a. user inputs / outputs) with corresponding type   `QNN_TENSOR_TYPE_APP_READ` / `QNN_TENSOR_TYPE_APP_WRITE`. Users are still expected to have `QNN_TENSOR_TYPE_NATIVE` for all nodes' IOs and leave the  detection logic handled inside `define_tensor` method.

5. Generate operator object in QNN graph:
    ```python
        layer_norm_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpLayerNorm.op_name,
        )
    ```

6. Pass IO tensors to operator object:
    ```python
        layer_norm_op.AddInputTensors(
            [input_tensor_wrapper, weight_tensor_wrapper, bias_tensor_wrapper]
        )
        layer_norm_op.AddOutputTensors([output_tensor_wrapper])
    ```
    The IO tensor objects created before are gathered up and shipped to operator object.

7. Pass parameters to operator object:
    ```python
        layer_norm_op.AddScalarParam(
            OpLayerNorm.param_epsilon,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(epsilon)},
        )
        layer_norm_op.AddTensorParam(
            OpLayerNorm.param_axes,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(axis_shape),
            axis_shape,
            np.array(axis, dtype=np.uint32),
            True,
        )
    ```
    By checking the `Shape` property of parameter in [Qualcomm AI Engine Direct Manual](#qualcomm-ai-engine-direct), it should be clear which API to be used. e.g.:
    - "epsilon" > __Shape__: scalar
    - "axes" > __Shape__: 1D of shape[M]

    The function signature of AddScalarParam is:
    - **name**: string maps to the operator name in Qualcomm AI Engine Direct manual
    - **data_type**: type compatible with QNN SDK, e.g. `QNN_DATATYPE_FLOAT_32`, `QNN_DATATYPE_UINT_32`, etc.
    - **attr**: dictionary for shipping data, currently only `QCOM_DATA` key is used

    The function signature of AddTensorParam is:
    - **name**: string maps to the operator name in Qualcomm AI Engine Direct manual
    - **data_type**: type compatible with QNN SDK, e.g. `QNN_DATATYPE_FLOAT_32`, `QNN_DATATYPE_UINT_32`, etc.
    - **rank**: dimensions of tensor
    - **dims**: shape of tensor
    - **data**: tesnor data
    - **copy_data**: user should specify to True for constant parameters

8. Last, return operator object for partitioner to conduct validation:
    ```python
        return layer_norm_op
    ```
    Also update the `__init__.py` for `register_node_visitor` to work properly:
    ```python
    from . import (
        ...
        op_index_put,
        # please insert codes in alphabetical order
        op_layer_norm,
        op_linear,
        ...
    )

    __all__ = [
        ...
        op_index_put,
        # please insert codes in alphabetical order
        op_layer_norm,
        op_linear,
        ...
    ]
    ```

### Quantizer Annotation
The operator now should be functional for Qualcomm backends. For operator to work in fixed-precision, we should also make `QnnQuantizer` to correctly insert observers for recording calibrated encodings. Please read more on the [Quantization Annotation Tutorial](../quantizer//README.md).

## Issues
Please refer to the [issue section](../README.md#issues) for more information.

## Pull Requests
Please refer to the [PR section](../README.md#pull-requests) for more information.
