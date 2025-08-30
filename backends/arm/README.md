# ExecuTorch Arm&reg; Delegate for TOSA devices

This subtree contains the Arm(R) Delegate implementation for ExecuTorch.

This delegate is structured to, over time, support a number of different Arm devices
through an AoT flow which targets multiple Arm IP using the TOSA standard.

For more information on TOSA see https://www.mlplatform.org/tosa/tosa_spec.html

**The expected flows are:**
* torch.nn.module -> TOSA for development and validation of model export
* torch.nn.module -> TOSA/VGF for flows supporting a JiT compilation step.
* torch.nn.module -> TOSA -> command_stream for fully AoT flows e.g. embedded.

**Currently device support is for:**
* TOSA to Ethos&trade;-U55/65/85 via the ethos-u-vela compilation stack.
  * This is cross-compiled to the appropriate target CPU
  * There is a separate arm_executor_runner for bare-metal platforms
* TOSA to VGF via the model-converter for devices supporting the ML SDK for Vulkan&reg;
  * The VGF graph represents TOSA directly in a SPIR-V&trade; standardized form.
  * As the VGF delegate runs on Vulkan, it's required to be built with the Vulkan delegate also present.

**Currently supported development platforms are:**
* For ahead of time tooling
  * Linux aarch64
  * Linux x86_64
  * macOS with Apple silicon
* Bare metal builds For the Ethos-U target and Cortex-M targets
  * Full testing is available in tree for the Corstone&trade; FVPs
  * This is a reference implementation for porting to silicon targets
* Linux target support For VGF capable targets
  * This flow re-uses the common executor_runner

## Layout of key components

Export:
* `tosa_backend.py` - The TOSA conversion flow all other backends rely on.
* `ethosu/backend.py` - Main entrypoint for the EthosUBackend.
* `vgf_backend.py` - Main entrypoint for VgfBackend.
  * For more information see the section on [Arm Backend Architecture](#arm-backend-architecture).
* `scripts` - For the core scripts which prepare AoT dependencies such as backend compilers.

Passes (which prepare the partitioned graphs for TOSA conversion):
* `_passes\arm_pass_manager.py` - Pass manager. Will decide which passes need to be applied depending on the compile_spec.
* `_passes\*_pass.py` - Compiler passes derived from ExportPass

Operators (which handle mapping of operators to TOSA):
* `operators/node_visitor.py` - Base class for edge operator lowering
* `operators/op_*.py` - Edge operator lowering/serialization to TOSA

Quantization:
* `quantizer/arm_quantizer.py` - Quantizers for Arm backend.
  * Contains the EthosUQuantizer which inherits from the TOSAQuantizer
  * Contains the VgfQuantizer which inherits from the TOSAQuantizer
* `arm_quantizer_utils.py` - Utilities for quantization

Runtime:
- `runtime/ArmEthosUBackend.cpp` - The Arm delegate for Ethos-U targets
- `runtime/VGFBackend.cpp` - The Arm delegate for VGF capable targets
- `CMakeLists.txt` - the build configuration for both targets

Other:
- `third-party/` - Dependencies for runtime builds
- `test/` - Unit test and test support functions


## Testing

The tests and related support scripts will test TOSA, Ethos-U and VGF behaviour based on the installed tools. It is expected that the relevant environment preparation has been performed as outlined in ./examples/arm/README.md.

After setup you can run unit tests with the test_arm_baremetal.sh script.

To run the pytests suite run

```
backends/arm/test/test_arm_baremetal.sh test_pytest
```

To run the unit test suite with Corstone3x0 FVP simulator support use

```
backends/arm/test/test_arm_baremetal.sh test_pytest_ethosu_fvp
```

You can test to run some models with the full fvp test flow

```
backends/arm/test/test_arm_baremetal.sh test_full_ethosu_fvp
```

To run the unit test suite with VKML use the following. Note Vulkan SDK need to be installed.
Have a look at install_vulkan_sdk() in .ci/scripts/setup-vulkan-linux-deps.sh on how to install Vulkan SDK.

```
backends/arm/test/test_arm_baremetal.sh test_pytest_vkml
```

You can test to run some models with the full VKML flow

```
backends/arm/test/test_arm_baremetal.sh test_full_vkml
```

## Unit tests

This is the structure of the test directory

```
test                            #  Root test folder
├── misc                        #  Testing of debug features
├── models                      #  Full model tests
├── ops                         #  Single op tests
├── passes                      #  Compiler passes tests
├── tester                      #  Arm Tester class
├── tosautil                    #  Utility functions for TOSA artifacts
├ common.py                     #  Common functions and definitions used by many tests
├ setup_testing.sh              #  Script to prepare testing for using the Corstone 3x0 FVP
├ setup_testing_vkml.sh         #  Script to prepare testing for using the VKML
├ test_arm_baremetal.sh         #  Help script to trigger testing
```

Some example commands to run these tests follow. Run a single test:

```
pytest -c /dev/null -v -n auto backends/arm/test/ops/test_add.py -k test_add2_tosa_BI
```

Or discover and run many tests:

```
pytest -c /dev/null -v -n auto backends/arm/test/ops/
```


You can run tests using Corstone3x0 simulators to see how it would work on something more target like
first you need to build and prepare some used target libs

```
examples/arm/run.sh --model_name=add --build_only
backends/arm/test/setup_testing.sh and/or backends/arm/test/setup_testing_vkml.sh
```

The you can run the tests with

```
pytest -c /dev/null -v -n auto backends/arm/test
```

### Model test dependencies
Some model tests in Arm backend require third-party libraries or packages. To run these tests, you need to install the required dependencies by running the script `examples/arm/setup.sh` with the flag `--setup-test-dependency`.

Please note that installing model test dependencies is a standalone process. When using the `--setup-test-dependency` flag, the script will install only the necessary dependencies for model tests, skipping all other setup procedures.

List of models with specific dependencies:
- Stable Diffusion: [diffusers](https://github.com/huggingface/diffusers/tree/main)


There are currently a number of ways we unit test our code:
1. TOSA FP. These tests are using non-quantized data and ops. Edge IR representation of the module is lowered to a TOSA flatbuffer, which is tested for numerical correcteness using the ```tosa_reference_model``` tool.
2. TOSA INT. Same as above, but data and ops integer, and represent a quantized domain.
3. Ethos-U. These tests use quantized data and ops (aka TOSA base inference). Edge IR is lowered to a TOSA flatbuffer, which is fed into the Vela compiler. Theses tests are functional tests and do not test numerical correctness, since that should be guaranteed by TOSA.
4. VGF. These tests enable both FP and INT testing for the VGF/SPIR-V representation of TOSA.

In order to distinguise between general, and more targeted tests, you will find suffixes with FP, INT, U55, VGF, etc.

## Help & Improvements
If you have problems or questions, or have suggestions for ways to make
implementation and testing better, please reach out to the Arm team developing this delegate, or
create an issue on [github](https://www.github.com/pytorch/executorch/issues) and add the "Partner: Arm" label.

# Arm Backend Architecture

The broad principle with the Arm backend implemention for ExecuTorch is to support multiple Arm devices and device configurations through a largely Homogeneous flow with maximal sharing of class logic.
The EthosUBackend and VgfBackend are the user facing targets available for the the Ethos-U55 and Ethos-U85 hardware IP, and VGF targets. It is using the TOSABackend under the hood to share compiler passes and legalisation, along with other code and functionality, but also to enable separate testing for the TOSA flow itself.

In practice for compilation, this means that the flow goes via [Arm TOSA](https://www.mlplatform.org/tosa/tosa_spec.html) to produce a common IR and quantization behaviour compatible with our various IP, and typically, device-specific backends to further lower to a device specific binary which can happen ahead of time (within the Python development flow) or at runtime (during a JIT compilation stage).


## Arm Backend Status and Maturity

The Arm EthosU Backend should be considered reasonable quality at this point, supporting a large number of operators and major networks.
The Arm VGF Backend should be considered of Alpha quality, likely subject to significant change and improvement, and with a limited coverage of functionality.
We are actively developing the codebase for both targets.

## Current flows

The Arm backends have a two stage process,
1. Compile to TOSA to by applying FX passes and legalizing the graph into supported TOSA profiles. Currently this is to v1.0 TOSA INT/FP, this is via calls into the TOSABackend.
1. Lower via the target compilation flow which takes TOSA v1.0 as an input and produces a lower level format for the hardware
  * For Ethos-U this is a hardware command stream that is possible to directly execute on hardware
  * For VGF this is a SPIR-V representation of TOSA to enable JiT compilation on the target platform

All targets provide a partitioner to enable the standard partially delegated flow offered by ExecuTorch.

There is also a generic TOSABackend with accompanying TOSAPartitioner and TOSAQuantizer, these can be used directly to verify the lowering to the TOSA representation of the model (refer to the unit tests in backends/arm/test which uses the TOSA backend in the test suites).

### Controlling compilation

It is possible to control the compilation flow to aid in development and debug of both networks and the code itself.

Configuration of the export flow is controlled by CompileSpec information (essentially used as compilation flags) to determine which of these outputs is produced. In particular this allows for compilation flags, capturing intermediate forms during lowering, and use of the tosa_reference_model to run intermediate output to check for correctness and quantization accuracy without a full loop via hardware implemntation.

## Model specific and optional passes
The current TOSA version does not support int64. However, int64 is commonly used in many models. In order to lower the operators with int64 inputs and/or outputs to TOSA, a few passes have been developed to handle the int64-related issues. The main idea behind these passes is to replace the uses of int64 with int32 where feasible.
- For floating-point models, these passes need to run very early in the lowering process and can be passed in to the to_edge_transform_and_lower() function call as an optional parameter.
- For quantized models, these transformations will be automatically handled during annotation before the export stage.

List of model specific and optional passes:
- InsertCastForOpsWithInt64InputPass
    - Functionality:
        - For LLMs such as LLama, some opeartors like aten.embedding have int64 input. In order to lower these operators to TOSA, this pass will insert a casting node that converts the input from int64 to int32.
        - Example usage: backends/arm/test/models/test_llama.py
    - Supported Ops:
        - aten.embedding.default, aten.slice_copy.Tensor
