# Executorch Arm/TOSA Delegate

This subtree contains the Arm Delegate implementation for Executorch.

This delegate is structured to, over time, support a number of different Arm devices
through an AoT flow which targets multiple Arm IP using the TOSA standard.

The expected flow is:
 * torch.nn.module -> TOSA -> command_stream for fully AoT flows e.g. embedded.
 * torch.nn.module -> TOSA for flows supporting a JiT compilation step.

Current backend support is being developed for TOSA to Ethos-U55/65 via the
ethos-u-vela compilation stack. which follows the fully AoT flow.

## Layout
- `arm_backend.py` - AoT Partitioner which maps to a subset of Base Inference and Main Inference TOSA profiles, where the subset may be further constrained for early support devices like Ethos-U55. AoT Backend which implements the preprocess step which converts to TOSA and can emit files for ethos-u-vela as shown in `executorch/examples/arm/`
- `test/` - unit test and test support functions
- `third-party/` - source dependencies - currently just on TOSA serialization_lib
- `tosa_mapping.py` - helper functions for mapping edge dialect to TOSA

## Help & Improvements
If you have problems or questions, or have suggestions for ways to make
implementation and testing better, please reach out to the Arm team developing this delegate, or
create an issue on [github](https://www.github.com/pytorch/executorch/issues).
