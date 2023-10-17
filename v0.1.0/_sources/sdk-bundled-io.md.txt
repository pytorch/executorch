# Bundled Program -- a Tool for ExecuTorch Model Validation

## Introduction
BundledProgram is a wrapper around the core ExecuTorch program designed to help users wrapping test cases with the model they deploy. BundledProgram is not necessarily a core part of the program and not needed for its execution, but is particularly important for various other use-cases, such as model correctness evaluation, including e2e testing during the model bring-up process.

Overall, the procedure can be broken into two stages, and in each stage we are supporting:

* **Emit stage**: Bundling the test I/O cases along with the ExecuTorch program, serializing into flatbuffer.
* **Runtime stage**: Accessing, executing, and verifying the bundled test cases during runtime.

## Emit stage
This stage mainly focuses on the creation of a `BundledProgram` and dumping it out to the disk as a flatbuffer file. The main procedure is as follow:
1. Create a model and emit its ExecuTorch program.
2. Construct a `BundledConfig` to record all info that needs to be bundled.
3. Generate `BundledProgram` by using the emited model and `BundledConfig`.
4. Serialize the `BundledProgram` and dump it out to the disk.

### Step 1: Create a Model and Emit its ExecuTorch Program.

ExecuTorch Program can be emitted from user's model by using ExecuTorch APIs. Follow the [Generate Sample ExecuTorch program](./getting-started-setup.md) or [Exporting to ExecuTorch tutorial](./tutorials/export-to-executorch-tutorial).

### Step 2: Construct `BundledConfig`


`BundledConfig` is a class under `executorch/bundled_program/config.py` that contains all information to be bundled for model verification. Here's the constructor api to create `BundledConfig`:

:::{dropdown} `BundledConfig`

```{eval-rst}
.. autofunction:: bundled_program.config.BundledConfig.__init__
    :noindex:
```
:::


### Step 3: Generate `BundledProgram`

We provide `create_bundled_program` API under `executorch/bundled_program/core.py` to generate `BundledProgram` by bundling the emitted ExecuTorch program with the bundled_config:

:::{dropdown} `BundledProgram`

```{eval-rst}
.. currentmodule:: bundled_program.core
.. autofunction:: create_bundled_program
    :noindex:
```
:::

`create_bundled_program` will do sannity check internally to see if the given BundledConfig matches the given Program's requirements. Specifically:
1. The name of methods we create BundledConfig for should be also in program. Please notice that it is no need to set testcases for every method in the Program.
2. The metadata of each testcase should meet the requirement of the coresponding inference methods input.

### Step 4: Serialize `BundledProgram` to Flatbuffer.

To serialize `BundledProgram` to make runtime APIs use it, we provide two APIs, both under `executorch/bundled_program/serialize/__init__.py`.

:::{dropdown} Serialize and Deserialize

```{eval-rst}
.. currentmodule:: bundled_program.serialize
.. autofunction:: serialize_from_bundled_program_to_flatbuffer
    :noindex:
```

```{eval-rst}
.. currentmodule:: bundled_program.serialize
.. autofunction:: deserialize_from_flatbuffer_to_bundled_program
    :noindex:
```
:::

### Emit Example

Here is a flow highlighting how to generate a `BundledProgram` given a PyTorch model and the representative inputs we want to test it along with.

```python

import torch
from torch.export import export

from executorch.bundled_program.config import BundledConfig
from executorch.bundled_program.core import create_bundled_program
from executorch.bundled_program.serialize import serialize_from_bundled_program_to_flatbuffer

from executorch.exir import to_edge


# Step 1: ExecuTorch Program Export

class SampleModel(torch.nn.Module):
    """An example model with multi-methods. Each method has multiple input and single output"""

    def __init__(self) -> None:
        super().__init__()
        self.a: torch.Tensor = 3 * torch.ones(2, 2, dtype=torch.int32)
        self.b: torch.Tensor = 2 * torch.ones(2, 2, dtype=torch.int32)

    def encode(
        self, x: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        z = x.clone()
        torch.mul(self.a, x, out=z)
        y = x.clone()
        torch.add(z, self.b, out=y)
        torch.add(y, q, out=y)
        return y

    def decode(
        self, x: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        y = x * q
        torch.add(y, self.b, out=y)
        return y

# Inference method names of SampleModel we want to bundle testcases to.
# Notices that we do not need to bundle testcases for every inference methods.
method_names = ["encode", "decode"]
model = SampleModel()

capture_inputs = {
    m_name: (
        (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
        (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
    )
    for m_name in method_names
}

# Find each method of model needs to be traced my its name, export its FX Graph.
method_graphs = {
    m_name: export(getattr(model, m_name), capture_inputs[m_name])
    for m_name in method_names
}

# Emit the traced methods into ET Program.
program = to_edge(method_graphs).to_executorch().executorch_program

# Step 2: Construct BundledConfig

# number of input sets to be verified
n_input = 10

# Input sets to be verified for each inference methods.
inputs = [
    # The below list is all inputs for a single inference method.
    [
        # Each list below is a individual input set.
        # The number of inputs, dtype and size of each input follow Program's spec.
        [
            (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
            (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
        ]
        for _ in range(n_input)
    ]
    for _ in method_names
]

# Expected outputs align with inputs.
expected_outputs = [
    [[getattr(model, m_name)(*x)] for x in inputs[i]]
    for i, m_name in enumerate(method_names)
]

# Create BundledConfig
bundled_config = BundledConfig(
    method_names, inputs, expected_outputs
)


# Step 3: Generate BundledProgram

bundled_program = create_bundled_program(program, bundled_config)

# Step 4: Serialize BundledProgram to flatbuffer.
serialized_bundled_program = serialize_from_bundled_program_to_flatbuffer(bundled_program)
save_path = "bundled_program.bp"
with open(save_path, "wb") as f:
    f.write(serialized_bundled_program)

```

We can also regenerate `BundledProgram` from flatbuffer file if needed:

```python
from executorch.bundled_program.serialize import deserialize_from_flatbuffer_to_bundled_program
save_path = "bundled_program.bp"
with open(save_path, "rb") as f:
    serialized_bundled_program = f.read()

regenerate_bundled_program = deserialize_from_flatbuffer_to_bundled_program(serialized_bundled_program)
```

## Runtime Stage
This stage mainly focuses on executing the model with the bundled inputs and and comparing the model's output with the bundled expected output. We provide multiple APIs to handle the key parts of it.


### Get ExecuTorch Program Pointer from `BundledProgram` Buffer
We need the pointer to ExecuTorch program to do the execution. To unify the process of loading and executing `BundledProgram` and Program flatbuffer, we create an API:

:::{dropdown} `GetProgramData`

```{eval-rst}
.. doxygenfunction:: torch::executor::util::GetProgramData
```
:::

Here's an example of how to use the `GetProgramData` API:
```c++
std::shared_ptr<char> buff_ptr;
size_t buff_len;

// FILE_PATH here can be either BundledProgram or Program flatbuffer file.
Error status = torch::executor::util::read_file_content(
    FILE_PATH, &buff_ptr, &buff_len);
ET_CHECK_MSG(
    status == Error::Ok,
    "read_file_content() failed with status 0x%" PRIx32,
    status);

const void* program_ptr;
size_t program_len;
status = torch::executor::util::GetProgramData(
    buff_ptr.get(), buff_len, &program_ptr, &program_len);
ET_CHECK_MSG(
    status == Error::Ok,
    "GetProgramData() failed with status 0x%" PRIx32,
    status);
```

### Load Bundled Input to Method
To execute the program on the bundled input, we need to load the bundled input into the method. Here we provided an API called `torch::executor::util::LoadBundledInput`:

:::{dropdown} `LoadBundledInput`

```{eval-rst}
.. doxygenfunction:: torch::executor::util::LoadBundledInput
```
:::

### Verify the Method's Output.
We call `torch::executor::util::VerifyResultWithBundledExpectedOutput` to verify the method's output with bundled expected outputs. Here's the details of this API:

:::{dropdown} `VerifyResultWithBundledExpectedOutput`

```{eval-rst}
.. doxygenfunction:: torch::executor::util::VerifyResultWithBundledExpectedOutput
```
:::


### Runtime Example

Here we provide an example about how to run the bundled program step by step. Most of the code is borrowed from [executor_runner](https://github.com/pytorch/executorch/blob/main/examples/sdk/sdk_example_runner/sdk_example_runner.cpp), and please review that file if you need more info and context:

```c++
// method_name is the name for the method we want to test
// memory_manager is the executor::MemoryManager variable for executor memory allocation.
// program is the ExecuTorch program.
Result<Method> method = program->load_method(method_name, &memory_manager);

ET_CHECK_MSG(
    method.ok(),
    "load_method() failed with status 0x%" PRIx32,
    method.error());

// Load testset_idx-th input in the buffer to plan
status = torch::executor::util::LoadBundledInput(
        *method,
        program_data.bundled_program_data(),
        &bundled_input_allocator,
        method_name,
        FLAGS_testset_idx);
ET_CHECK_MSG(
    status == Error::Ok,
    "LoadBundledInput failed with status 0x%" PRIx32,
    status);

// Execute the plan
status = method->execute();
ET_CHECK_MSG(
    status == Error::Ok,
    "method->execute() failed with status 0x%" PRIx32,
    status);

// Verify the result.
status = torch::executor::util::VerifyResultWithBundledExpectedOutput(
        *method,
        program_data.bundled_program_data(),
        &bundled_input_allocator,
        method_name,
        FLAGS_testset_idx,
        FLAGS_rtol,
        FLAGS_atol);
ET_CHECK_MSG(
    status == Error::Ok,
    "Bundle verification failed with status 0x%" PRIx32,
    status);

```

## Common Errors

Errors will be raised if `BundledConfig` doesn't match the `Program`. Here're two common situations:

### Test input doesn't match model's requirement.

Each inference method of PyTorch model has its own requirement for the inputs, like number of input, the dtype of each input, etc. `BundledProgram` will raise error if test input not meet the requirement.

Here's the example of the dtype of test input not meet model's requirement:

```python
import torch
from torch.export import export

from executorch.bundled_program.config import BundledConfig
from executorch.bundled_program.core import create_bundled_program

from executorch.exir import to_edge


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
method_names = ['forward']

inputs = torch.ones(2, 2, dtype=torch.float)
print(model(inputs))

# Find each method of model needs to be traced my its name, export its FX Graph.
method_graphs = {
    m_name: export(getattr(model, m_name), (inputs, ))
    for m_name in method_names
}

# Emit the traced methods into ET Program.
program = to_edge(method_graphs).to_executorch().executorch_program


# number of input sets to be verified
n_input = 10

# All Input sets to be verified.
inputs = [
    [
        # NOTE: executorch program needs torch.float, but here is torch.int
        [
            torch.randint(-5, 5, (2, 2), dtype=torch.int),
        ]
        for _ in range(n_input)
    ]
]

# Expected outputs align with inputs.
expected_outpus = [
    [[model(*x)] for x in inputs[0]]
]

bundled_config = BundledConfig(method_names, inputs, expected_outpus)

bundled_program = create_bundled_program(program, bundled_config)
```

:::{dropdown} Raised Error

```
The input tensor tensor([[ 0,  3],
        [-3, -3]], dtype=torch.int32) dtype shall be torch.float32, but now is torch.int32
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
     57 expected_outpus = [
     58     [[model(*x)] for x in inputs[0]]
     59 ]
     61 bundled_config = BundledConfig(method_names, inputs, expected_outpus)
---> 63 bundled_program = create_bundled_program(program, bundled_config)
File /executorch/bundled_program/core.py:270, in create_bundled_program(program, bundled_config)
    259 def create_bundled_program(
    260     program: Program,
    261     bundled_config: BundledConfig,
    262 ) -> BundledProgram:
    263     """Create BundledProgram by bundling the given program and bundled_config together.
    264
    265     Args:
    266         program: The program to be bundled.
    267         bundled_config: The config to be bundled.
    268     """
--> 270     assert_valid_bundle(program, bundled_config)
    272     execution_plan_tests: List[BundledExecutionPlanTest] = []
    274     # Emit data and metadata of bundled tensor
File /executorch/bundled_program/core.py:224, in assert_valid_bundle(program, bundled_config)
    220 # type of tensor input should match execution plan
    221 if type(cur_plan_test_inputs[j]) == torch.Tensor:
    222     # pyre-fixme[16]: Undefined attribute [16]: Item `bool` of `typing.Union[bool, float, int, torch._tensor.Tensor]`
    223     # has no attribute `dtype`.
--> 224     assert cur_plan_test_inputs[j].dtype == get_input_dtype(
    225         program, program_plan_id, j
    226     ), "The input tensor {} dtype shall be {}, but now is {}".format(
    227         cur_plan_test_inputs[j],
    228         get_input_dtype(program, program_plan_id, j),
    229         cur_plan_test_inputs[j].dtype,
    230     )
    231 elif type(cur_plan_test_inputs[j]) in (
    232     int,
    233     bool,
    234     float,
    235 ):
    236     assert type(cur_plan_test_inputs[j]) == get_input_type(
    237         program, program_plan_id, j
    238     ), "The input primitive dtype shall be {}, but now is {}".format(
    239         get_input_type(program, program_plan_id, j),
    240         type(cur_plan_test_inputs[j]),
    241     )
AssertionError: The input tensor tensor([[ 0,  3],
        [-3, -3]], dtype=torch.int32) dtype shall be torch.float32, but now is torch.int32

```

:::

### Method name in `BundleConfig` does not exist.

Another common error would be the method name in `BundledConfig` does not exist in Model. `BundledProgram` will raise error and show the non-exist method name:

```python
import torch
from torch.export import export

from executorch.bundled_program.config import BundledConfig
from executorch.bundled_program.core import create_bundled_program

from executorch.exir import to_edge



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

method_names = ['forward']

inputs = torch.ones(2, 2, dtype=torch.float)

# Find each method of model needs to be traced my its name, export its FX Graph.
method_graphs = {
    m_name: export(getattr(model, m_name), (inputs, ))
    for m_name in method_names
}

# Emit the traced methods into ET Program.
program = to_edge(method_graphs).to_executorch().executorch_program

# Number of input sets to be verified
n_input = 10

# All Input sets to be verified.
inputs = [
    [
        [
            torch.randint(-5, 5, (2, 2), dtype=torch.float),
        ]
        for _ in range(n_input)
    ]
]

# Expected outputs align with inputs.
expected_outpus = [
    [[model(*x)] for x in inputs[0]]
]


# NOTE: MISSING_METHOD_NAME is not an inference method in the above model.
wrong_method_names = ['MISSING_METHOD_NAME']

bundled_config = BundledConfig(wrong_method_names, inputs, expected_outpus)

bundled_program = create_bundled_program(program, bundled_config)

```

:::{dropdown} Raised Error

```
All method names in bundled config should be found in program.execution_plan,          but {'wrong_forward'} does not include.
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
     58 expected_outpus = [
     59     [[model(*x)] for x in inputs[0]]
     60 ]
     62 bundled_config = BundledConfig(method_names, inputs, expected_outpus)
---> 64 bundled_program = create_bundled_program(program, bundled_config)
File /executorch/bundled_program/core.py:270, in create_bundled_program(program, bundled_config)
    259 def create_bundled_program(
    260     program: Program,
    261     bundled_config: BundledConfig,
    262 ) -> BundledProgram:
    263     """Create BundledProgram by bundling the given program and bundled_config together.
    264
    265     Args:
    266         program: The program to be bundled.
    267         bundled_config: The config to be bundled.
    268     """
--> 270     assert_valid_bundle(program, bundled_config)
    272     execution_plan_tests: List[BundledExecutionPlanTest] = []
    274     # Emit data and metadata of bundled tensor
File /executorch/bundled_program/core.py:147, in assert_valid_bundle(program, bundled_config)
    142 method_name_of_program = {e.name for e in program.execution_plan}
    143 method_name_of_bundled_config = {
    144     t.method_name for t in bundled_config.execution_plan_tests
    145 }
--> 147 assert method_name_of_bundled_config.issubset(
    148     method_name_of_program
    149 ), f"All method names in bundled config should be found in program.execution_plan, \
    150      but {str(method_name_of_bundled_config - method_name_of_program)} does not include."
    152 # check if  has been sorted in ascending alphabetical order of method name.
    153 for bp_plan_id in range(1, len(bundled_config.execution_plan_tests)):
AssertionError: All method names in bundled config should be found in program.execution_plan,          but {'MISSING_METHOD_NAME'} does not include.
```
:::
