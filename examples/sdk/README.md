# SDK Examples
This directory contains examples of BundledProgram and ETDump generation.

## Directory structure
```bash
examples/sdk
├── scripts                           # Python scripts to illustrate export workflow of bundled program.
├── sdk_executor_runner               # Contains an example for both BundledProgram to verify ExecuTorch model, and generate ETDump for runtime results.
└── README.md                         # Current file
```

## BundledProgram

We will use an example model (in `torch.nn.Module`) and its representative inputs, both from [`models/`](../models) directory, to generate a [BundledProgram(`.bp`)](../../docs/source/sdk-bundled-io.md) file using the [script](scripts/export_bundled_program.py). Then we will use [sdk_example_runner](sdk_example_runner/sdk_example_runner.cpp) to execute the `.bp` model on the ExecuTorch runtime and verify the model on BundledProgram API.


1. Sets up the basic development environment for ExecuTorch by [Setting up ExecuTorch from GitHub](../../docs/source/getting-started-setup.md).

2. Using the [script](scripts/export_bundled_program.py) to generate a BundledProgram binary file by retreiving a `torch.nn.Module` model and its representative inputs from the list of available models in the [`models/`](../models) dir。

```bash
cd executorch # To the top level dir

# To get a list of example models
python3 -m examples.sdk.scripts.export_bundled_program -h

# To generate a specific `.bp` model
python3 -m examples.sdk.scripts.export_bundled_program -m mv2 # for MobileNetv2

# This should generate ./mv2_bundled.bp file, if successful.
```

3. Once we have the BundledProgram binary (`.bp`) file, then let's run and verify it with ExecuTorch runtime and BundledProgram APIs using the [sdk_example_runner](sdk_example_runner/sdk_example_runner.cpp).

```bash
buck2 run examples/sdk/sdk_example_runner:sdk_example_runner -- --bundled_program_path mv2_bundled.bp --output_verification
```


## Generate ETDump

Next step is to generate an ``ETDump``. ``ETDump`` contains runtime results
from executing the model. To generate, users have two options:

**Option 1:**

Use Buck::
```bash
   python3 -m examples.sdk.scripts.export_bundled_program -m mv2
   buck2 run -c executorch.event_tracer_enabled=true examples/sdk/sdk_example_runner:sdk_example_runner -- --bundled_program_path mv2_bundled.bp
```
 **Option 2:**

 Use CMake::
```bash
   cd executorch
   rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake -DBUCK2=buck2 -DEXECUTORCH_BUILD_SDK=1 -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=1 ..
   cd ..
   cmake --build cmake-out -j8 -t sdk_example_runner
   ./cmake-out/examples/sdk/sdk_example_runner --bundled_program_path mv2_bundled.bp
   ```
