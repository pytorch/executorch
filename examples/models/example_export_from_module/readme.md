# Example of an external model for the ARM AOT Compiler
This is an example of an external Python file to be used as a module by the `run.sh` (and the `aot_arm_compiler.py`) scripts in `examples/arm` directory. Just pass the path of the `model.py` file as `--model_name`.

`ModelUnderTest` should be a `torch.nn.module` instance.

`ModelInputs` should be a tuple of inputs to the forward function.
