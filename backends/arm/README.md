# ExecuTorch Arm/TOSA Delegate

This subtree contains the Arm(R) Delegate implementation for ExecuTorch.

This delegate is structured to, over time, support a number of different Arm devices
through an AoT flow which targets multiple Arm IP using the TOSA standard.

The expected flow is:
 * torch.nn.module -> TOSA -> command_stream for fully AoT flows e.g. embedded.
 * torch.nn.module -> TOSA for flows supporting a JiT compilation step.

Current backend support is being developed for TOSA to Ethos(TM)-U55/65/85 via the
ethos-u-vela compilation stack. which follows the fully AoT flow.

## Layout

Export:
- `ethosu_backend.py` - Main entrypoint for the EthosUBackend. For more information see the section on
[Arm Backend Architecture](#arm-backend-architecture). For examples of use see `executorch/examples/arm`.
- `tosa_mapping.py` - utilities for mapping edge dialect to TOSA
- `tosa_quant_utils.py` - utilities for mapping quantization information to TOSA encoding

Operators:
- `node_visitor.py` - Base class for edge operator lowering
- `op_*.py` - Edge operator lowering/serialization to TOSA

Passes:
- `arm_pass_manager.py` - Pass manager. Will decide which passes need to be applied depending on the compile_spec.
- `*_pass.py` - Compiler passes derived from ExportPass

Quantization:
- `arm_quantizer.py` - Quantizers for Arm backend. Contains the EthosUQuantizer which inherits from the TOSAQuantizer
- `arm_quantizer_utils.py` - Utilities for quantization

Runtime:
- `runtime/ArmEthosUBackend.cpp` - The Arm backend implementation of the ExecuTorch runtime backend (BackendInterface) for Ethos-U

Other:
- `third-party/` - Dependencies on other code - in particular the TOSA serialization_lib for compiling to TOSA and the ethos-u-core-driver for the bare-metal backend supporting Ethos-U
- `test/` - Unit test and test support functions

## Testing

After a setup you can run unit tests with the test_arm_baremetal.sh script.

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
├ test_arm_baremetal.sh         #  Help script to trigger testing
```

Some example commands to run these tests follow. Run a single test:

```
python -m unittest backends.arm.test.ops.test_add.TestSimpleAdd -k test_add2_tosa_BI
```

or with pytest

```
pytest -c /dev/null -v -n auto backends/arm/test/ops/test_add.py -k test_add2_tosa_BI
```

Or all tests in "TestSimpleAdd":

```
python -m unittest backends.arm.test.ops.test_add.TestSimpleAdd
```

Or discover and run many tests:

```
python -m unittest discover -s backends/arm/test/ops/
```

or with pytest

```
pytest -c /dev/null -v -n auto backends/arm/test/ops/
```


You can run tests using Corstone3x0 simulators to see how it would work on something more target like
first you need to build and prepare some used target libs

```
examples/arm/run.sh --model_name=add --build_only
backends/arm/test/setup_testing.sh
```

The you can run the tests with

```
pytest -c /dev/null -v -n auto backends/arm/test --arm_run_corstoneFVP
```

## Passes

With the default passes in the Arm Ethos-U backend, assuming the model lowers fully to the
Ethos-U, the exported program is composed of a Quantize node, Ethos-U custom delegate
and a Dequantize node. In some circumstances, you may want to feed quantized input to the Neural
Network straight away, e.g. if you have a camera sensor outputting (u)int8 data and keep all the
arithmetic of the application in the int8 domain. For these cases, you can apply the
`exir/passes/quantize_io_pass.py`. See the unit test in `executorch/backends/arm/
test/passes/test_ioquantization_pass.py`for an example how to feed quantized inputs and
obtain quantized outputs.


### Code coverage

To get code coverage:

```
coverage run --source=<SRC> --rcfile=backends/arm/test/.coveragerc -m pytest \
--config-file=/dev/null backends/arm/test/
```

All files in `SRC` and its child directories will be analysed for code coverage,
unless explicitly exluded in the .coveragerc file. If using venv this might be
under `env/lib/python<VERSION_NUMBER>/site-packages/executorch/`. To get the
absolute path, run:

```
python -c "import executorch; print(executorch.__path__)"
```

This contains a list of paths where the source directory is located. Pick the
one that is located in `env/lib`. If that does not work try the others. Add
`backends/arm` to the path in `--source` to only get code coverage for the Arm
backend.

### A note on unit tests

There are currently 3 ways we unit test our code.
1. TOSA main inference. These tests are using non-quantized data and ops. Edge IR representation of the module is lowered to a TOSA flatbuffer, which is tested for numerical correcteness using the ```tosa_reference_model``` tool.
2. TOSA base inference. Same as above, but data and ops are quantized.
3. Ethos-U55. These tests use quantized data and ops (aka TOSA base inference). Edge IR is lowered to a TOSA flatbuffer, which is fed into the Vela compiler. Theses tests are functional tests and do not test numerical correctness, since that should be guaranteed by TOSA.

In order to distinguise between the different tests, the following suffixes have been added to the respective test case.
* ```_MI``` for main inference
* ```_BI``` for base inference
* ```_U55_BI``` for base inference on U55

## Help & Improvements
If you have problems or questions, or have suggestions for ways to make
implementation and testing better, please reach out to the Arm team developing this delegate, or
create an issue on [github](https://www.github.com/pytorch/executorch/issues).

# Arm Backend Architecture

The broad principle with the Arm backend implemention for ExecuTorch is to support multiple Arm devices and device configurations through a largely Homogeneous flow with maximal sharing of class logic.
The EthosUBackend is currently the one user facing API that target the Ethos-U55 and Ethos-U85 hardware IP. It is using the TOSABackend under the hood to share code and functionality, but also to separate testing possibilities to the TOSA flow itself.

In practice for compilation, this means that the flow goes via [Arm TOSA](https://www.mlplatform.org/tosa/tosa_spec.html) to produce a common IR and quantization behaviour compatible with our various IP, and typically, device-specific backends to further lower to a device specific binary which can happen ahead of time (within the Python development flow) or at runtime (during a JIT compilation stage).

In practice for the runtime, this means we will share common runtime backend functionality, with the aim for features like debugging to be available through common tooling.


## Arm Backend Status and Maturity

The Arm EthosU Backend should be considered a prototype quality at this point, likely subject to significant change and improvement, and with a limited coverage of functionality. We are actively developing this codebase.

## Current flows

The EthosUBackend has a two stage process,
- Compile to TOSA to rationalise the graph into known hardware support profiles. Currently this is to v0.80 TOSA BI with specific concern to a subset which gives support on Ethos-U55 and Ethos-U85, the target of the initial prototype efforts. This calls into the TOSABackend.
- Lower via the ethos-u-vela compilation flow which takes TOSA v0.80 as an input and produces a low level commandstream for the hardware which is then passed via the delegate to the ethos-u-core-driver for direct execution.

The EthosUPartitioner is currenly used to ensure the operations converted are Ethos-U compatible, but will be extended to offer spec-correct TOSA Base inference and TOSA Main Inference generation in future.

There is also a generic TOSABackend with accompanying TOSAPartitioner and TOSAQuantizer, which are used by the EthosUBackend and friends. The Arm TOSA Backend can be used by it's own to verify the lowering to the TOSA representation of the model (refer to the unit tests in backends/arm/test which uses the TOSA backend in the test suites).

### Controlling compilation

It is possible to control the compilation flow to aid in development and debug of both networks and the code itself.

Configuration of the EthosUBackend export flow is controlled by CompileSpec information (essentially used as compilation flags) to determine which of these outputs is produced. In particular this allows for use of the tosa_reference_model to run intermediate output to check for correctness and quantization accuracy without a full loop via hardware implemntation.

As this is in active development see the EthosUBackend for accurate information on [compilation flags](https://github.com/pytorch/executorch/blob/29f6dc9353e90951ed3fae3c57ae416de0520067/backends/arm/arm_backend.py#L319-L324)
