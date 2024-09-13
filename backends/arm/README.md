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
- `arm_backend.py` - Main entrypoint for the ArmPartitioner and ArmBackend. For more information see the section on
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
- `arm_quantizer.py` - Quantizer for Arm backend
- `arm_quantizer_utils.py` - Utilities for quantization

Runtime:
- `runtime/ArmBackendEthosU.cpp` - The Arm backend implementation of the ExecuTorch runtime backend (BackendInterface) for Ethos-U

Other:
- `third-party/` - Dependencies on other code - in particular the TOSA serialization_lib for compiling to TOSA and the ethos-u-core-driver for the bare-metal backend supporting Ethos-U
- `test/` - Unit test and test support functions

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
```

Some example commands to run these tests follow. Run a single test:

```
python -m unittest backends.arm.test.ops.test_add.TestSimpleAdd -k test_add2_tosa_BI
```

Or all tests in "TestSimpleAdd":

```
python -m unittest backends.arm.test.ops.test_add.TestSimpleAdd
```

Or discover and run many tests:

```
python -m unittest discover -s backends/arm/test/ops/
```

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

In practice for compilation, this means that the flow goes via [Arm TOSA](https://www.mlplatform.org/tosa/tosa_spec.html) to produce a common IR and quantization behaviour compatible with our various IP, and typically, device-specific backends to further lower to a device specific binary which can happen ahead of time (within the Python development flow) or at runtime (during a JIT compilation stage).

In practice for the runtime, this means we will share common runtime backend functionality, with the aim for features like debugging to be available through common tooling.


## Arm Backend Status and Maturity

The Arm Backend should be considered a prototype quality at this point, likely subject to significant change and improvement, and with a limited coverage of functionality. We are actively developing this codebase.

## Current flows

The ArmBackend has a two stage process,
- Compile to TOSA to rationalise the graph into known hardware support profiles. Currently this is to v0.80.0 TOSA BI with specific concern to a subset which gives support on Ethos-U55, the target of the initial prototype efforts.
- Lower via the ethos-u-vela compilation flow which takes TOSA v0.80.0 as an input and produces a low level commandstream for the hardware which is then passed via the delegate to the ethos-u-core-driver for direct execution.

The ArmPartitioner is currenly used to ensure the operations converted are Ethos-U compatible, but will be extended to offer spec-correct TOSA Base inference and TOSA Main Inference generation in future.

### Controlling compilation

It is possible to control the compilation flow to aid in development and debug of both networks and the code itself.

Configuration of the ArmBackend export flow is controlled by CompileSpec information (essentially used as compilation flags) to determine which of these outputs is produced. In particular this allows for use of the tosa_reference_model to run intermediate output to check for correctness and quantization accuracy without a full loop via hardware implemntation.

As this is in active development see the ArmBackend for accurate information on [compilation flags](https://github.com/pytorch/executorch/blob/29f6dc9353e90951ed3fae3c57ae416de0520067/backends/arm/arm_backend.py#L319-L324)

You can also refer to the [example TOSA end-to-end code](/examples/arm/arm_tosa_e2e.py)
