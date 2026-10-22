# MobileSAM Runtime

MobileSAM uses the standard ExecuTorch Arm executor runner. The example has no
model-specific C++ runtime or CMake project.

The top-level [`run.sh`](../run.sh) embeds the exported `.pte` in the runner,
builds it for Ethos-U85-256, and launches it on Corstone-320. Semihosting passes
the raw input and output tensors between the host and FVP.

After inference, [`visualize_fvp_output.py`](visualize_fvp_output.py) thresholds
the raw output tensor, checks it against the host quantized mask, and writes:

`arm_test/mobilesam/result/fvp_comparison.png`

The runtime stage passes when the runner reports successful execution and the
FVP/reference mask IoU is at least `0.9`.
