## Export Friendly Modules

Modules in this directory are:
* Extending `torch.nn.Module`.
* Guranteed to work out of the box with `torch.export.export()` and `torch.aot_compile()`.
* Guranteed to be able to work with ExecuTorch.

All modules should be covered by unit tests to make sure they are:
1. giving the same output as the reference implementation in PyTorch or torchtune
2. export friendly
3. AOTI friendly
4. ExecuTorch friendly

Notice that these modules are subject to change (may upstream to torchtune) so proceed with caution.
