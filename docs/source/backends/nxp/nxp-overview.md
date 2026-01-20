# NXP eIQ Neutron Backend

This manual page is dedicated to introduction NXP eIQ Neutron backend.
NXP offers accelerated machine learning models inference on edge devices.
To learn more about NXP's machine learning acceleration platform, please refer to [the official NXP website](https://www.nxp.com/applications/technologies/ai-and-machine-learning:MACHINE-LEARNING).

<div class="admonition tip">
For up-to-date status about running ExecuTorch on Neutron backend please visit the <a href="https://github.com/pytorch/executorch/blob/main/backends/nxp/README.md">manual page</a>.
</div>

## Features


ExecuTorch v1.0 supports running machine learning models on selected NXP chips (for now only i.MXRT700).
Among currently supported machine learning models are:
- Convolution-based neutral networks
- Full support for MobileNetV2 and CifarNet

## Target Requirements

- Hardware with NXP's [i.MXRT700](https://www.nxp.com/products/i.MX-RT700) chip or a evaluation board like MIMXRT700-EVK. 

## Development Requirements

- [MCUXpresso IDE](https://www.nxp.com/design/design-center/software/development-software/mcuxpresso-software-and-tools-/mcuxpresso-integrated-development-environment-ide:MCUXpresso-IDE) or [MCUXpresso Visual Studio Code extension](https://www.nxp.com/design/design-center/software/development-software/mcuxpresso-software-and-tools-/mcuxpresso-for-visual-studio-code:MCUXPRESSO-VSC)
- [MCUXpresso SDK 25.12](https://mcuxpresso.nxp.com/mcuxsdk/25.12.00/html/index.html)
- eIQ Neutron Converter for MCUXPresso SDK 25.12, what you can download from eIQ PyPI:

```commandline
$ pip install --index-url https://eiq.nxp.com/repository neutron_converter_SDK_25_12
```

Instead of manually installing requirements, except MCUXpresso IDE and SDK, you can use the setup script: 
```commandline
$ ./examples/nxp/setup.sh
```

## Using NXP eIQ Backend

To test converting a neural network model for inference on NXP eIQ Neutron backend, you can use our example script:

```shell
# cd to the root of executorch repository
./examples/nxp/aot_neutron_compile.sh [model (cifar10 or mobilenetv2)]
```

For a quick overview how to convert a custom PyTorch model, take a look at our [example python script](https://github.com/pytorch/executorch/tree/release/1.0/examples/nxp/aot_neutron_compile.py).


## Runtime Integration

To learn how to run the converted model on the NXP hardware, use one of our example projects on using ExecuTorch runtime from MCUXpresso IDE example projects list.
For more finegrained tutorial, visit [this manual page](https://mcuxpresso.nxp.com/mcuxsdk/latest/html/middleware/eiq/executorch/docs/nxp/topics/example_applications.html).

## Reference

**→{doc}`nxp-partitioner` — Partitioner options.**

**→{doc}`nxp-quantization` — Supported quantization schemes.**

**→{doc}`tutorials/nxp-tutorials` — Tutorials.**

**→{doc}`nxp-dim-order` — Dim order support (channels last inputs).**

```{toctree}
:maxdepth: 2
:hidden:
:caption: NXP Backend

nxp-partitioner
nxp-quantization
tutorials/nxp-tutorials
nxp-dim-order
```
