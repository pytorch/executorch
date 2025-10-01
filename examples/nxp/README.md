# PyTorch Model Delegation to Neutron Backend

In this guide we will show how to use the ExecuTorch AoT flow to convert a PyTorch model to ExecuTorch format and delegate the model computation to eIQ Neutron NPU using the eIQ Neutron Backend.

First we will start with an example script converting the model. This example show the CifarNet model preparation. It is the same model which is part of the `example_cifarnet`

The steps are expected to be executed from the executorch root folder.
1. Run the setup.sh script to install the neutron-converter:
```commandline
$ examples/nxp/setup.sh
```

2. Now run the `aot_neutron_compile.py` example with the `cifar10` model 
```commandline
$ python -m examples.nxp.aot_neutron_compile --quantize \
    --delegate --neutron_converter_flavor SDK_25_09 -m cifar10 
```

3. It will generate you `cifar10_nxp_delegate.pte` file which can be used with the MXUXpresso SDK `cifarnet_example` project, presented [here](https://mcuxpresso.nxp.com/mcuxsdk/latest/html/middleware/eiq/executorch/docs/nxp/topics/example_applications.html#how-to-build-and-run-executorch-cifarnet-example).
To get the MCUXpresso SDK follow this [guide](https://mcuxpresso.nxp.com/mcuxsdk/latest/html/middleware/eiq/executorch/docs/nxp/topics/getting_mcuxpresso.html), use the MCUXpresso SDK v25.03.00. 