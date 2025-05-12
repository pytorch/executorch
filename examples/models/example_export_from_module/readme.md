# Example of an external model for the ARM AOT Compiler
This is an example of an external Python file to be used as a module by the ```aot_arm_compiler.py``` (and the ```run.sh```) scripts in ```examples/arm``` directory. 

```ModelUnderTest``` should be a ```torch.nn.module``` instance.

```ModelInputs``` should be a tuple of inputs to the forward function.
