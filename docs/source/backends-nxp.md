# NXP eIQ Neutron Backend

This manual page is dedicated to introduction of using the ExecuTorch with NXP eIQ Neutron Backend.
NXP offers accelerated machine learning models inference on edge devices.
To learn more about NXP's machine learning acceleration platform, please refer to [the official NXP website](https://www.nxp.com/applications/technologies/ai-and-machine-learning:MACHINE-LEARNING).

<div class="admonition tip">
For up-to-date status about running ExecuTorch on Neutron Backend please visit the <a href="https://github.com/pytorch/executorch/blob/main/backends/nxp/README.md">manual page</a>.
</div>

## Features

Executorch v1.0 supports running machine learning models on selected NXP chips (for now only i.MXRT700).
Among currently supported machine learning models are:
- Convolution-based neutral networks
- Full support for MobileNetv2 and CifarNet

## Prerequisites (Hardware and Software)

In order to succesfully build executorch project and convert models for NXP eIQ Neutron Backend you will need a computer running Windows or Linux.

If you want to test the runtime, you'll also need:
- Hardware with NXP's [i.MXRT700](https://www.nxp.com/products/i.MX-RT700) chip or a testing board like MIMXRT700-AVK
- [MCUXpresso IDE](https://www.nxp.com/design/design-center/software/development-software/mcuxpresso-software-and-tools-/mcuxpresso-integrated-development-environment-ide:MCUXpresso-IDE) or [MCUXpresso Visual Studio Code extension](https://www.nxp.com/design/design-center/software/development-software/mcuxpresso-software-and-tools-/mcuxpresso-for-visual-studio-code:MCUXPRESSO-VSC)

## Using NXP backend 

To test converting a neural network model for inference on NXP eIQ Neutron Backend, you can use our example script:

```shell
# cd to the root of executorch repository
./examples/nxp/aot_neutron_compile.sh [model (cifar10 or mobilenetv2)]
```

For a quick overview how to convert a custom PyTorch model, take a look at our [exmple python script](https://github.com/pytorch/executorch/tree/release/1.0/examples/nxp/aot_neutron_compile.py).

## Runtime Integration

To learn how to run the converted model on the NXP hardware, use one of our example projects on using executorch runtime from MCUXpresso IDE example projects list.
For more finegrained tutorial, visit [this manual page](https://mcuxpresso.nxp.com/mcuxsdk/latest/html/middleware/eiq/executorch/docs/nxp/topics/example_applications.html).
