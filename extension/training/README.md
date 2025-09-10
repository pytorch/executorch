# ExecuTorch On-device Training

This subtree contains infrastructure to facilitate on-device training using ExecuTorch.
This feature is experimental and under heavy active development, all the APIs are
subject to change and many things may not work out of the box or at all in the
current state.

## Layout
- `examples/` : Example end to end flows from model definition to optimizer.step()
- `module/`: Utility class to provide an improved UX when using ExecuTorch for Training.
- `optimizer/`: Cpp implementations of various optimizers, currently only SGD though Adam is planned.
- `test/`: Tests that cover multiple subdirs.

## Technical Birds Eye view

At a high level ExecuTorch training follows a similar flow to inference with a few extra steps.

Instead of relying on autograd at runtime to dynamically generate the backward graph and then walk it,
we capture the backward graph ahead of time. This lets us be a lot leaner on-device as well as
letting backends have more direct control over more of the model execution. Currently the optimizer is not
captured though this may change over time.

Loss functions must be embedded inside the model definition (and be the first output) this is used during
capture to generate the backwards graph.

Gradients become explicit graph outputs rather then hidden tensor state.

Since the weights now need to be mutable during execution, they are memory planned ahead of time and copied
from the .pte into the HeirarchicalAllocator arenas during Method init.

Integration with backends/delegates is still a work in progress.


## End to End Example

To further understand the features of ExecuTorch Training and how to leverage it,
consider the following end to end example with a neural network learning the XOR function.

### Lowering a joint-graph model to ExecuTorch

After following the [setting up ExecuTorch] guide. You can run

```bash
python3 extension/training/examples/XOR/export_model.py --outdir /tmp/foobar
```
to generate the model file. Below is a walkthrough of how that script works.

First lets define our model.
```python
import torch.nn as nn
from torch.nn import functional as F

from torch.export import export
from torch.export.experimental import _export_forward_backward


# Basic Net for XOR
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 10)
        self.linear2 = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear2(F.sigmoid(self.linear(x)))
```

The first big difference from the normal ExecuTorch flow is that for training we must embed
the loss function into model and return the loss as our first output.

We don't want to modify the original model definition so we will just wrap it.

```python
class TrainingNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label):
        pred = self.net(input)
        return self.loss(pred, label), pred.detach().argmax(dim=1)
```

Now that we have our model we can lower it to ExecuTorch. To do that we just have to follow
a few simple steps.

```python
net = TrainingNet(Net())

# Create our inputs, only the shapes of these matter.
input = torch.randn(1, 2)
label = torch.ones(1, dtype=torch.int64)

# Captures the forward graph. The graph will look similar to the model definition now.
ep = export(net, (input, label))
```

This is what the graph looks like after export
```python
>>>print(ep.graph_module.graph)

graph():
    %p_net_linear_weight : [num_users=1] = placeholder[target=p_net_linear_weight]
    %p_net_linear_bias : [num_users=1] = placeholder[target=p_net_linear_bias]
    %p_net_linear2_weight : [num_users=1] = placeholder[target=p_net_linear2_weight]
    %p_net_linear2_bias : [num_users=1] = placeholder[target=p_net_linear2_bias]
    %input : [num_users=1] = placeholder[target=input]
    %label : [num_users=1] = placeholder[target=label]
    %linear : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%input, %p_net_linear_weight, %p_net_linear_bias), kwargs = {})
    %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%linear,), kwargs = {})
    %linear_1 : [num_users=2] = call_function[target=torch.ops.aten.linear.default](args = (%sigmoid, %p_net_linear2_weight, %p_net_linear2_bias), kwargs = {})
    %cross_entropy_loss : [num_users=1] = call_function[target=torch.ops.aten.cross_entropy_loss.default](args = (%linear_1, %label), kwargs = {})
    %detach : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%linear_1,), kwargs = {})
    %argmax : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%detach, 1), kwargs = {})
    return (cross_entropy_loss, argmax)
```

It should look pretty similar to our model's forward function. Now we need to capture the backwards graph.

```python
ep = _export_forward_backward(ep)
```

and now the graph is

```python
>>>print(ep.graph_module.graph)

graph():
    %p_net_linear_weight : [num_users=1] = placeholder[target=p_net_linear_weight]
    %p_net_linear_bias : [num_users=1] = placeholder[target=p_net_linear_bias]
    %p_net_linear2_weight : [num_users=1] = placeholder[target=p_net_linear2_weight]
    %p_net_linear2_bias : [num_users=1] = placeholder[target=p_net_linear2_bias]
    %input : [num_users=2] = placeholder[target=input]
    %label : [num_users=5] = placeholder[target=label]
    %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%p_net_linear_weight, [1, 0]), kwargs = {})
    %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%p_net_linear_bias, %input, %permute), kwargs = {})
    %sigmoid : [num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm,), kwargs = {})
    %alias : [num_users=1] = call_function[target=torch.ops.aten.alias.default](args = (%sigmoid,), kwargs = {})
    %alias_1 : [num_users=1] = call_function[target=torch.ops.aten.alias.default](args = (%alias,), kwargs = {})
    %permute_1 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%p_net_linear2_weight, [1, 0]), kwargs = {})
    %addmm_1 : [num_users=2] = call_function[target=torch.ops.aten.addmm.default](args = (%p_net_linear2_bias, %sigmoid, %permute_1), kwargs = {})
    %_log_softmax : [num_users=3] = call_function[target=torch.ops.aten._log_softmax.default](args = (%addmm_1, 1, False), kwargs = {})
    %alias_2 : [num_users=1] = call_function[target=torch.ops.aten.alias.default](args = (%_log_softmax,), kwargs = {})
    %alias_3 : [num_users=1] = call_function[target=torch.ops.aten.alias.default](args = (%alias_2,), kwargs = {})
    %ne : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%label, -100), kwargs = {})
    %scalar_tensor : [num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu})
    %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %label, %scalar_tensor), kwargs = {})
    %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 1), kwargs = {})
    %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%_log_softmax, 1, %unsqueeze), kwargs = {})
    %squeeze : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%gather, [1]), kwargs = {})
    %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
    %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%label, -100), kwargs = {})
    %scalar_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu})
    %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %scalar_tensor_1), kwargs = {})
    %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%label, -100), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%ne_2, []), kwargs = {})
    %_to_copy : [num_users=2] = call_function[target=torch.ops.aten._to_copy.default](args = (%sum_1,), kwargs = {dtype: torch.float32, device: cpu})
    %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where_1, []), kwargs = {})
    %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, %_to_copy), kwargs = {})
    %alias_4 : [num_users=1] = call_function[target=torch.ops.aten.alias.default](args = (%addmm_1,), kwargs = {})
    %alias_5 : [num_users=1] = call_function[target=torch.ops.aten.alias.default](args = (%alias_4,), kwargs = {})
    %alias_6 : [num_users=1] = call_function[target=torch.ops.aten.alias.default](args = (%alias_5,), kwargs = {})
    %argmax : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%alias_6, 1), kwargs = {})
    %full_like : [num_users=1] = call_function[target=torch.ops.aten.full_like.default](args = (%div, 1), kwargs = {pin_memory: False, memory_format: torch.preserve_format})
    %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%full_like, %_to_copy), kwargs = {})
    %unsqueeze_1 : [num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%label, 1), kwargs = {})
    %ne_3 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_1, -100), kwargs = {})
    %scalar_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu})
    %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze_1, %scalar_tensor_2), kwargs = {})
    %full_like_1 : [num_users=1] = call_function[target=torch.ops.aten.full_like.default](args = (%_log_softmax, 0), kwargs = {pin_memory: False, memory_format: torch.preserve_format})
    %scatter : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_like_1, 1, %where_2, -1.0), kwargs = {})
    %ne_4 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_1, -100), kwargs = {})
    %scalar_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu})
    %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %div_1, %scalar_tensor_3), kwargs = {})
    %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter, %where_3), kwargs = {})
    %alias_7 : [num_users=1] = call_function[target=torch.ops.aten.alias.default](args = (%alias_3,), kwargs = {})
    %alias_8 : [num_users=1] = call_function[target=torch.ops.aten.alias.default](args = (%alias_7,), kwargs = {})
    %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%alias_8,), kwargs = {})
    %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1], True), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, %sum_3), kwargs = {})
    %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %mul_1), kwargs = {})
    %permute_2 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_1, [1, 0]), kwargs = {})
    %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%sub, %permute_2), kwargs = {})
    %permute_3 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%sub, [1, 0]), kwargs = {})
    %mm_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%permute_3, %sigmoid), kwargs = {})
    %permute_4 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mm_1, [1, 0]), kwargs = {})
    %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sub, [0], True), kwargs = {})
    %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%sum_4, [2]), kwargs = {})
    %permute_5 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_4, [1, 0]), kwargs = {})
    %alias_9 : [num_users=1] = call_function[target=torch.ops.aten.alias.default](args = (%alias_1,), kwargs = {})
    %alias_10 : [num_users=2] = call_function[target=torch.ops.aten.alias.default](args = (%alias_9,), kwargs = {})
    %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %alias_10), kwargs = {})
    %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%alias_10, %sub_1), kwargs = {})
    %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, %mul_2), kwargs = {})
    %permute_6 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_3, [1, 0]), kwargs = {})
    %mm_2 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%permute_6, %input), kwargs = {})
    %permute_7 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mm_2, [1, 0]), kwargs = {})
    %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [0], True), kwargs = {})
    %view_1 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%sum_5, [10]), kwargs = {})
    %permute_8 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_7, [1, 0]), kwargs = {})
    return (div, argmax, permute_8, view_1, permute_5, view)
```

Its a lot bigger! We call this the 'joint graph' or the 'forwards backwards graph'. We have explicitly captured the backwards graph
alongside the forward and now our model returns [Loss, Any other user outputs, Gradients].

From here we can lower the rest of the way to ExecuTorch
```python
ep = to_edge(ep)

# After calling to_executorch the weights themselves are also appended to the model outputs. This is to make
# some downstream passes like memory planning a little easier. A couple of hidden utility functions are also
# embedded in the model __et_training_gradients_index_<method_name>,
# __et_training_parameters_index_<method_name>, __et_training_fqn_<method_name>.
#
# These help us partition the huge list of model outputs into meaningful sections as well as assign names to each weight/gradient.
ep = ep.to_executorch()

with open("xor.pte", "wb") as file:
    ep.write_to_file(file)
```

### Run the model train script with CMAKE
After exporting the model for training, we can now try learning using CMake. We can build and use the train_xor, which is a sample wrapper for the ExecuTorch Runtime, TrainingModule, and SGD optimizer. We first begin by configuring the CMake build like such:
```bash
# cd to the root of executorch repo
cd executorch

# Get a clean cmake-out directory
./install_executorch.sh --clean
mkdir cmake-out

# Configure cmake
cmake \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DPYTHON_EXECUTABLE=python \
    -Bcmake-out .
```
Then you can build the runtime componenets with

```bash
cmake --build cmake-out -j9 --target install --config Release
```

Now you should be able to find the executable built at `./cmake-out/extension/training/train_xor` you can run the executable with the model you generated as such
```bash
./cmake-out/extension/training/train_xor --model_path=./xor.pte
```

## What is missing?/ What is next?
A ton! ExecuTorch training is still quite experimental and under heavy active development. Whats here currently is more of a technical preview.

The _export_forward_backward is not very stable yet and may fail on more complicated model architectures, though we have verified it works for LoRA with LLMs.

The ExecuTorch portable operator lib does not yet have full coverage of ops that might show up in the backwards graphs.

We don't have a way yet to serialize the newly trained weights natively in ExecuTorch (though you can convert them to ATen tensors using extension/aten_util and then serialize them using ATen APIs).

We plan to add a way to update models in place on-device (will be needed for finetuning).

We are looking to integrate with many of the existing delegates/backends on ET enabling accelerated training.

and so much more!

## Help & Improvements
If you have problems or questions, or have suggestions for ways to make
implementation and testing better, please reach out to the PyTorch Edge team or
create an issue on [github](https://www.github.com/pytorch/executorch/issues).
