# NXP eIQ Neutron Kernel Selective Kernel Registration

The NXP ExecuTorch backend supports selective Neutron kernel registration for `Neutron-C` targets, which decreases the
size of the Neutron Firmware. During the backend's conversion to the Neutron representation by the Neutron Converter,
microcode for the Neutron accelerator is generated.
The microcode consists of kernel calls executed by the Neutron Driver. The code for kernel call functions is
distributed in Neutron Firmware. 

The `eiq_neutron_sdk.neutron_converter` optionally generates the `*_kernel_selection.c` file, registering 
only kernels that are required for a particular model or in the case of ExecuTorch, a delegated subgraph. This 
`*_kernel_selection.c`, when used during the application linking, takes precedence over the default list of registered 
kernels in the Neutron Firmware, and allows the linker to include only the necessary Neutron kernels.
This software is required for deployment on an edge device (e.g. `i.MXRT700`) and is
distributed via the MCUXpresso SDK. The MCUXpresso SDK enables building of a final application that is then flashed on 
the edge device. For more details about this process, see
[eIQ ExecuTorch Library User Guide](https://mcuxpresso.nxp.com/mcuxsdk/latest/html/middleware/eiq/executorch/docs/nxp/ugindex.html).

By default, for Neutron-C targets like `i.MXRT700`, all kernel implementations are present in the Neutron Firmware, which
is linked to the final application. This enables an easy build process for any model, but increases the size of the
final application with unused code. In the case of limited RAM, you can link only kernels that are used in the set of
models deployed. This way you can reduce the size of the final app by linking only selected kernels, used in one or
multiple models.

The feature works as follows: The Neutron Converter with the appropriate flag exports a kernel selection file for each 
converted subgraph, the kernel selection files are then merged and ready to be included in the MCUXpresso SDK to use for
a selection-only build.

> **Note:** This feature applies only to `Neutron-C` targets. `Neutron-S` has a different implementation and links only
> kernel calls included in the model's microcode by default.

## Export kernel selection file

To turn on this feature on the side of NXP ExecuTorch backend, use the parameter `--dump_kernel_selection_code` in 
`aot_neutron_compile.py`. An example with the CifarNet model:

```commandline
python -m examples.nxp.aot_neutron_compile --quantize \
    --delegate -m cifar10 \
    --dump_kernel_selection_code
```

This command will create a `*_kernel_selection.c` file alongside the converted PTE file in the working directory.

## Kernel Registration for Multiple Models

If you want to use or experiment with multiple models in one application while having reduced kernel set, you can
create one kernel selection file with the script `merge_kernel_selection_code.py`:

```commandline
python -m eiq_neutron_sdk.neutron_library_utils.merge_kernel_selection_code \
    -input-files model1_kernel_selection.c model2_kernel_selection.c [modelX_kernel_selection.c ...] \ 
    -output-file merged_kernel_selection.c
```

Each particular model must be converted by the same Neutron converter version, so the `*_kernel_selection.c` files
share the same version.

## MCUXpresso SDK build with kernel selection

Copy your model PTE file and kernel selection file into the MCUXpresso SDK and follow its 
[documentation](https://mcuxpresso.nxp.com/mcuxsdk/latest/html/middleware/eiq/executorch/docs/nxp/ugindex.html).
