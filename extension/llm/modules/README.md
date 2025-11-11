## Export-friendly Modules

Modules in this directory:
* Extend `torch.nn.Module`.
* Are guaranteed to work out of the box with `torch.export.export()`.
* Should work out of the box with `torch.aot_compile()`.
* Should be able to workt with ExecuTorch.

All modules should be covered by unit tests to make sure they are:
1. Give the output as the reference eager model in PyTorch or TorrchTune
2. Export-friendly

Additionally, we aim to make these modules:
3. AOTI-friendly
4. ExecuTorch-friendly

These modules are subject to change (may upstream to TorchTune) so proceed with caution.
