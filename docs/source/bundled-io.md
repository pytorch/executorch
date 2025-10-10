# Bundled Program -- a Tool for ExecuTorch Model Validation

## Introduction
`BundledProgram` is a wrapper around the core ExecuTorch program designed to help users wrapping test cases with the model they deploy. `BundledProgram` is not necessarily a core part of the program and not needed for its execution, but is particularly important for various other use-cases, such as model correctness evaluation, including e2e testing during the model bring-up process.

Overall, the procedure can be broken into two stages, and in each stage we are supporting:

* **Emit stage**: Bundling the test I/O cases along with the ExecuTorch program, serializing into flatbuffer.
* **Runtime stage**: Accessing, executing, and verifying the bundled test cases during runtime.

## Emit stage
This stage mainly focuses on the creation of a `BundledProgram` and dumping it out to the disk as a flatbuffer file. The main procedure is as follow:
1. Create a model and emit its ExecuTorch program.
2. Construct a `List[MethodTestSuite]` to record all test cases that needs to be bundled.
3. Generate `BundledProgram` by using the emited model and `List[MethodTestSuite]`.
4. Serialize the `BundledProgram` and dump it out to the disk.

### Step 1: Create a Model and Emit its ExecuTorch Program.

ExecuTorch Program can be emitted from user's model by using ExecuTorch APIs. Follow the [Generate and emit sample ExecuTorch program](getting-started.md#exporting) or [Exporting to ExecuTorch tutorial](tutorials/export-to-executorch-tutorial) <!-- @lint-ignore -->.

### Step 2: Construct `List[MethodTestSuite]` to hold test info

In `BundledProgram`, we create two new classes, `MethodTestCase` and `MethodTestSuite`, to hold essential info for ExecuTorch program verification.

`MethodTestCase` represents a single testcase. Each `MethodTestCase` contains inputs and expected outputs for a single execution.

:::{dropdown} `MethodTestCase`

```{eval-rst}
.. autofunction:: executorch.devtools.bundled_program.config.MethodTestCase.__init__
    :noindex:
```
:::

`MethodTestSuite` contains all testing info for single method, including a str representing method name, and a `List[MethodTestCase]` for all testcases:

:::{dropdown} `MethodTestSuite`

```{eval-rst}
.. autofunction:: executorch.devtools.bundled_program.config.MethodTestSuite
    :noindex:
```
:::

Since each model may have multiple inference methods, we need to generate `List[MethodTestSuite]` to hold all essential infos.


### Step 3: Generate `BundledProgram`

We provide `BundledProgram` class under `executorch/devtools/bundled_program/core.py` to bundled the `ExecutorchProgram`-like variable, including
                            `ExecutorchProgram`, `MultiMethodExecutorchProgram` or `ExecutorchProgramManager`, with the `List[MethodTestSuite]`:

:::{dropdown} `BundledProgram`

```{eval-rst}
.. autofunction:: executorch.devtools.bundled_program.core.BundledProgram.__init__
    :noindex:
```
:::

Construtor of `BundledProgram `will do sannity check internally to see if the given `List[MethodTestSuite]` matches the given Program's requirements. Specifically:
1. The method_names of each `MethodTestSuite` in `List[MethodTestSuite]` for should be also in program. Please notice that it is no need to set testcases for every method in the Program.
2. The metadata of each testcase should meet the requirement of the coresponding inference methods input.

### Step 4: Serialize `BundledProgram` to Flatbuffer.

To serialize `BundledProgram` to make runtime APIs use it, we provide two APIs, both under `executorch/devtools/bundled_program/serialize/__init__.py`.

:::{dropdown} Serialize and Deserialize

```{eval-rst}
.. currentmodule:: executorch.devtools.bundled_program.serialize
.. autofunction:: serialize_from_bundled_program_to_flatbuffer
    :noindex:
```

```{eval-rst}
.. currentmodule:: executorch.devtools.bundled_program.serialize
.. autofunction:: deserialize_from_flatbuffer_to_bundled_program
    :noindex:
```
:::

### Emit Example

Here is a flow highlighting how to generate a `BundledProgram` given a PyTorch model and the representative inputs we want to test it along with.

```python
import torch

from executorch.exir import to_edge_transform_and_lower
from executorch.devtools import BundledProgram

from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from torch.export import export


# Step 1: ExecuTorch Program Export
class SampleModel(torch.nn.Module):
    """An example model with multi-methods. Each method has multiple input and single output"""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('a', 3 * torch.ones(2, 2, dtype=torch.int32))
        self.register_buffer('b', 2 * torch.ones(2, 2, dtype=torch.int32))

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        z = x.clone()
        torch.mul(self.a, x, out=z)
        y = x.clone()
        torch.add(z, self.b, out=y)
        torch.add(y, q, out=y)
        return y


# Inference method name of SampleModel we want to bundle testcases to.
# Notices that we do not need to bundle testcases for every inference methods.
method_name = "forward"
model = SampleModel()

# Inputs for graph capture.
capture_input = (
    (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
    (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
)

# Export method's FX Graph.
method_graph = export(
    export(model, capture_input).module(),
    capture_input,
)


# Emit the traced method into ET Program.
et_program = to_edge_transform_and_lower(method_graph).to_executorch()

# Step 2: Construct MethodTestSuite for Each Method

# Prepare the Test Inputs.

# Number of input sets to be verified
n_input = 10

# Input sets to be verified.
inputs = [
    # Each list below is a individual input set.
    # The number of inputs, dtype and size of each input follow Program's spec.
    [
        (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
        (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
    ]
    for _ in range(n_input)
]

# Generate Test Suites
method_test_suites = [
    MethodTestSuite(
        method_name=method_name,
        test_cases=[
            MethodTestCase(
                inputs=input,
                expected_outputs=(getattr(model, method_name)(*input), ),
            )
            for input in inputs
        ],
    ),
]

# Step 3: Generate BundledProgram
bundled_program = BundledProgram(et_program, method_test_suites)

# Step 4: Serialize BundledProgram to flatbuffer.
serialized_bundled_program = serialize_from_bundled_program_to_flatbuffer(
    bundled_program
)
save_path = "bundled_program.bpte"
with open(save_path, "wb") as f:
    f.write(serialized_bundled_program)
```

We can also regenerate `BundledProgram` from flatbuffer file if needed:

```python
from executorch.devtools.bundled_program.serialize import deserialize_from_flatbuffer_to_bundled_program
save_path = "bundled_program.bpte"
with open(save_path, "rb") as f:
    serialized_bundled_program = f.read()

regenerate_bundled_program = deserialize_from_flatbuffer_to_bundled_program(serialized_bundled_program)
```

## Runtime Stage
This stage mainly focuses on executing the model with the bundled inputs and comparing the model's output with the bundled expected output. We provide multiple APIs to handle the key parts of it.


### Get ExecuTorch Program Pointer from `BundledProgram` Buffer
We need the pointer to ExecuTorch program to do the execution. To unify the process of loading and executing `BundledProgram` and Program flatbuffer, we create an API for this
`executorch::bundled_program::get_program_data`. Check out an [example usage](https://github.com/pytorch/executorch/blob/release/0.6/examples/devtools/example_runner/example_runner.cpp#L128-L137) of this API.

### Load Bundled Input to Method
To execute the program on the bundled input, we need to load the bundled input into the method. Here we provided an API called `executorch::bundled_program::load_bundled_input`.  Check out an [example usage](https://github.com/pytorch/executorch/blob/release/0.6/examples/devtools/example_runner/example_runner.cpp#L253-L259) of this API.

### Verify the Method's Output.
We call `executorch::bundled_program::verify_method_outputs` to verify the method's output with bundled expected outputs. Check out an [example usage](https://github.com/pytorch/executorch/blob/release/0.6/examples/devtools/example_runner/example_runner.cpp#L300-L311) of this API.

### Runtime Example

Please checkout our [example runner](https://github.com/pytorch/executorch/blob/release/0.6/examples/devtools/README.md#bundledprogram) for a bundled program. You could run these commands to test with the BundledProgram binary (`.bpte`) file you generated in the previous step:

```bash
cd executorch
   ./examples/devtools/build_example_runner.sh
   ./cmake-out/examples/devtools/example_runner --bundled_program_path {your-bpte-file} --output_verification
```

It is expected to see no output from running the above mentioned snippet.

For a detailed example of how the runner should be like, please refer to our [example runner](https://github.com/pytorch/executorch/blob/release/0.6/examples/devtools/example_runner/example_runner.cpp).

## Common Errors

Errors will be raised if `List[MethodTestSuites]` doesn't match the `Program`. Here're two common situations:

### Test input doesn't match model's requirement.

Each inference method of PyTorch model has its own requirement for the inputs, like number of input, the dtype of each input, etc. `BundledProgram` will raise error if test input not meet the requirement.

Here's the example of the dtype of test input not meet model's requirement:

```python
import torch

from executorch.exir import to_edge
from executorch.devtools import BundledProgram

from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from torch.export import export


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 3 * torch.ones(2, 2, dtype=torch.float)
        self.b = 2 * torch.ones(2, 2, dtype=torch.float)

    def forward(self, x):
        out_1 = torch.ones(2, 2, dtype=torch.float)
        out_2 = torch.ones(2, 2, dtype=torch.float)
        torch.mul(self.a, x, out=out_1)
        torch.add(out_1, self.b, out=out_2)
        return out_2


model = Module()
method_names = ["forward"]

inputs = (torch.ones(2, 2, dtype=torch.float), )

# Find each method of model needs to be traced my its name, export its FX Graph.
method_graph = export(
    export(model, inputs).module(),
    inputs,
)

# Emit the traced methods into ET Program.
et_program = to_edge(method_graph).to_executorch()

# number of input sets to be verified
n_input = 10

# Input sets to be verified for each inference methods.
# To simplify, here we create same inputs for all methods.
inputs = {
    # Inference method name corresponding to its test cases.
    m_name: [
        # NOTE: executorch program needs torch.float, but here is torch.int
        [
            torch.randint(-5, 5, (2, 2), dtype=torch.int),
        ]
        for _ in range(n_input)
    ]
    for m_name in method_names
}

# Generate Test Suites
method_test_suites = [
    MethodTestSuite(
        method_name=m_name,
        test_cases=[
            MethodTestCase(
                inputs=input,
                expected_outputs=(getattr(model, m_name)(*input),),
            )
            for input in inputs[m_name]
        ],
    )
    for m_name in method_names
]

# Generate BundledProgram

bundled_program = BundledProgram(et_program, method_test_suites)
```

:::{dropdown} Raised Error

```
The input tensor tensor([[-2,  0],
        [-2, -1]], dtype=torch.int32) dtype shall be torch.float32, but now is torch.int32
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
Cell In[1], line 72
     56 method_test_suites = [
     57     MethodTestSuite(
     58         method_name=m_name,
   (...)
     67     for m_name in method_names
     68 ]
     70 # Step 3: Generate BundledProgram
---> 72 bundled_program = create_bundled_program(program, method_test_suites)
File /executorch/devtools/bundled_program/core.py:276, in create_bundled_program(program, method_test_suites)
    264 """Create bp_schema.BundledProgram by bundling the given program and method_test_suites together.
    265
    266 Args:
   (...)
    271     The `BundledProgram` variable contains given ExecuTorch program and test cases.
    272 """
    274 method_test_suites = sorted(method_test_suites, key=lambda x: x.method_name)
--> 276 assert_valid_bundle(program, method_test_suites)
    278 bundled_method_test_suites: List[bp_schema.BundledMethodTestSuite] = []
    280 # Emit data and metadata of bundled tensor
File /executorch/devtools/bundled_program/core.py:219, in assert_valid_bundle(program, method_test_suites)
    215 # type of tensor input should match execution plan
    216 if type(cur_plan_test_inputs[j]) == torch.Tensor:
    217     # pyre-fixme[16]: Undefined attribute [16]: Item `bool` of `typing.Union[bool, float, int, torch._tensor.Tensor]`
    218     # has no attribute `dtype`.
--> 219     assert cur_plan_test_inputs[j].dtype == get_input_dtype(
    220         program, program_plan_id, j
    221     ), "The input tensor {} dtype shall be {}, but now is {}".format(
    222         cur_plan_test_inputs[j],
    223         get_input_dtype(program, program_plan_id, j),
    224         cur_plan_test_inputs[j].dtype,
    225     )
    226 elif type(cur_plan_test_inputs[j]) in (
    227     int,
    228     bool,
    229     float,
    230 ):
    231     assert type(cur_plan_test_inputs[j]) == get_input_type(
    232         program, program_plan_id, j
    233     ), "The input primitive dtype shall be {}, but now is {}".format(
    234         get_input_type(program, program_plan_id, j),
    235         type(cur_plan_test_inputs[j]),
    236     )
AssertionError: The input tensor tensor([[-2,  0],
        [-2, -1]], dtype=torch.int32) dtype shall be torch.float32, but now is torch.int32

```

:::

### Method name in `BundleConfig` does not exist.

Another common error would be the method name in any `MethodTestSuite` does not exist in Model. `BundledProgram` will raise error and show the non-exist method name:

```python
import torch

from executorch.exir import to_edge
from executorch.devtools import BundledProgram

from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from torch.export import export


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 3 * torch.ones(2, 2, dtype=torch.float)
        self.b = 2 * torch.ones(2, 2, dtype=torch.float)

    def forward(self, x):
        out_1 = torch.ones(2, 2, dtype=torch.float)
        out_2 = torch.ones(2, 2, dtype=torch.float)
        torch.mul(self.a, x, out=out_1)
        torch.add(out_1, self.b, out=out_2)
        return out_2


model = Module()
method_names = ["forward"]

inputs = (torch.ones(2, 2, dtype=torch.float),)

# Find each method of model needs to be traced my its name, export its FX Graph.
method_graph = export(
    export(model, inputs).module(),
    inputs,
)

# Emit the traced methods into ET Program.
et_program = to_edge(method_graph).to_executorch()

# number of input sets to be verified
n_input = 10

# Input sets to be verified for each inference methods.
# To simplify, here we create same inputs for all methods.
inputs = {
    # Inference method name corresponding to its test cases.
    m_name: [
        [
            torch.randint(-5, 5, (2, 2), dtype=torch.float),
        ]
        for _ in range(n_input)
    ]
    for m_name in method_names
}

# Generate Test Suites
method_test_suites = [
    MethodTestSuite(
        method_name=m_name,
        test_cases=[
            MethodTestCase(
                inputs=input,
                expected_outputs=(getattr(model, m_name)(*input),),
            )
            for input in inputs[m_name]
        ],
    )
    for m_name in method_names
]

# NOTE: MISSING_METHOD_NAME is not an inference method in the above model.
method_test_suites[0].method_name = "MISSING_METHOD_NAME"

# Generate BundledProgram
bundled_program = BundledProgram(et_program, method_test_suites)

```

:::{dropdown} Raised Error

```
All method names in bundled config should be found in program.execution_plan,          but {'MISSING_METHOD_NAME'} does not include.
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
Cell In[3], line 73
     70 method_test_suites[0].method_name = "MISSING_METHOD_NAME"
     72 # Generate BundledProgram
---> 73 bundled_program = create_bundled_program(program, method_test_suites)
File /executorch/devtools/bundled_program/core.py:276, in create_bundled_program(program, method_test_suites)
    264 """Create bp_schema.BundledProgram by bundling the given program and method_test_suites together.
    265
    266 Args:
   (...)
    271     The `BundledProgram` variable contains given ExecuTorch program and test cases.
    272 """
    274 method_test_suites = sorted(method_test_suites, key=lambda x: x.method_name)
--> 276 assert_valid_bundle(program, method_test_suites)
    278 bundled_method_test_suites: List[bp_schema.BundledMethodTestSuite] = []
    280 # Emit data and metadata of bundled tensor
File /executorch/devtools/bundled_program/core.py:141, in assert_valid_bundle(program, method_test_suites)
    138 method_name_of_program = {e.name for e in program.execution_plan}
    139 method_name_of_test_suites = {t.method_name for t in method_test_suites}
--> 141 assert method_name_of_test_suites.issubset(
    142     method_name_of_program
    143 ), f"All method names in bundled config should be found in program.execution_plan, \
    144      but {str(method_name_of_test_suites - method_name_of_program)} does not include."
    146 # check if method_tesdt_suites has been sorted in ascending alphabetical order of method name.
    147 for test_suite_id in range(1, len(method_test_suites)):
AssertionError: All method names in bundled config should be found in program.execution_plan,          but {'MISSING_METHOD_NAME'} does not include.
```
:::
