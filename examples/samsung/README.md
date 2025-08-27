# Exynos backend Examples

This directory contains examples for some AI models.

Please make sure you have built the library and executable before
you start, if you have no idea how to build, please refer to [backend README](../../backends/samsung/README.md).

## Environment
We set up `PYTHONPATH` because it's easier to develop and import executorch Python APIs.
Users might also build and install executorch package as usual python package.
```bash
export PYTHONPATH=${EXECUTORCH_ROOT}/..
```

Note: Since we set `PYTHONPATH`, we may have issue with finding `program.fbs` and `scalar_type.fbs`
when we export a model. A workaround is that we copy them to directory `${EXECUTORCH_ROOT}/exir/_serialize/`.
We can find the files in `${EXECUTORCH_ROOT}/schema` or in
`${EXECUTORCH_ROOT}/pip-out/lib.linux-x86_64-cpython-310/executorch/exir/_serialize`.

## Device
Prepare an android phone with samsung exynos chip. Use `adb` to connect with the mobile phone.

Check the chip's version, when lower the model, set the corresponding chip version.

## Lowering

Before running an example, please copy python artifacts to target directory `PYTHON_TARGET_DIR`.
If you use `build.sh` script to compile samsung backend, please skip the copy step.

Set up `PYTHON_TARGET_DIR` to `${EXECUTORCH_ROOT}/backends/samsung/python`.
```bash
cp -rf ${EXECUTORCH_ROOT}/build_x86_64/backends/samsung/Py*.so ${PYTHON_TARGET_DIR}
cp -rf ${EXYNOS_AI_LITECORE_PATH}/python/snc_py_api*.so ${PYTHON_TARGET_DIR}
```

Take `EXECUTORCH_ROOT` as work directory and here is an example for ic3.
```bash
python -m executorch.examples.samsung.aot_compiler --chipset E9955 -m ic3 --output_dir ic3_artifact
```

## Execution

After lowering, we could get a pte model and then run it on mobile phone.

#### Step 1: Push required ENN libraries and executor runner to device
```bash
DEVICE_DIR=/data/local/tmp/executorch
adb shell mkdir ${DEVICE_DIR}
adb push ${EXECUTORCH_ROOT}/cmake-android-out/backends/samsung/enn_executor_runner ${DEVICE_DIR}
```

#### Step 2: Indicate dynamic linkers and execute model
```bash
adb push ./ic3_exynos_fp32.pte ${DEVICE_DIR}
adb shell "cd ${DEVICE_DIR} \
           && export LD_LIBRARY_PATH=${DEVICE_DIR} \
           && ./enn_executor_runner -model ./ic3_exynos_fp32.pte -input ./ic3_input_0.bin --output_path ."
```

`enn_executor_runner` has more usages, please refer to the help message.
