# NXP eIQ Dim Order Support

The NXP ExecuTorch backend supports two different dim orders (memory formats):

* **contiguous** (default)
* **channels last**

Any other dim order will cause an exception during runtime.

## Using the channels last dim order

If you want to use the NXP backend to accelerate your model, it may be beneficial to use the *channels last* dim order.

The Neutron NPU computes on channels last data, whereas ExecuTorch uses the channels first format. This mismatch
requires the execution of some extra transpositions during runtime, which can noticeably impact the performance. By
forcing the ExecuTorch operators to use the channels last dim order, some of these transpositions can be eliminated.

There are **two** options for using the **channels last** dim order:

1. Maintaining the same model inputs
2. Changing the inputs to **channels last**

### Maintaining the same model inputs

The **first** option requires a change to the *python* definition of your model. The model inputs remain unchanged
(contiguous/channels first), and an extra operator is inserted to change their dim order to **channels last**. This
approach does not change how the model is used later on at all. End-to-end, it behaves as if we did nothing to the dim
order.

Example use-case:

```python
class MyModel(nn.Module):

    def forward(self, x):
        # Transform the inputs to the channels last dim order.
        x = x.to(memory_format=torch.channels_last)

        ...  # Rest of your model definition


...

# Turn the weights to the channels last dim order.
model = MyModel().to(memory_format=torch.channels_last)

...  # Export and lower the model using the NXP backend.
```

Depending on the model, the NXP backend may be able to utilize the channels last dim order to achieve faster inference.
In many cases, there will be no improvement, and it is even possible (though rare) for the performance to get worse. So
you should compare the inference speed of the contiguous and channels last variants.

### Changing the inputs to channels last

The **second** option does not require the model definition to be altered, but it also changes the model inputs.

**Note:** Instead of the original **contiguous/channels_first** (NCHW) format, the **input data must be provided in the
channels last** (NHWC) format during runtime.

This approach may provide **significant speed improvements** by greatly reducing the number
of added transpositions (even to 0).

Example use-case:

```python
# Turn the weights to the channels last dim order.
model = YourModel().to(memory_format=torch.channels_last)

# Turn the example inputs to the channels last dim order.
# This will define the dim order of the inputs and the internal data at runtime.
example_inputs = tuple(
    i.to(memory_format=torch.channels_last) for i in your_example_inputs
)

# Use the channels last example inputs to export the model.
exported_program = torch.export.export(model, example_inputs)

...  # Lower the model using the NXP backend.
```

A full example of this use case can be found in the
[aot_neutron_compile.py](https://github.com/pytorch/executorch/blob/main/examples/nxp/aot_neutron_compile.py). The
following command will create the `cifar10_nxp_delegate.pte` model, which takes channels first inputs, contains no
transpositions, and can be run on the `i.MX RT700` board using the __MCUXpresso SDK 25.06__. For details on the
installation see {doc}`nxp-overview`.

```
python -m examples.nxp.aot_neutron_compile --quantize \
    --delegate --neutron_converter_flavor SDK_25_09 -m cifar10 \
    --use_channels_last_dim_order
```

```{toctree}
:hidden:
:maxdepth: 2

nxp-overview
```