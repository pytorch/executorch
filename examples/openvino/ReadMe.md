# TODO: Delete and reformat later

## Build Executorch

```bash
git clone -b openvino_backend https://github.com/ynimmaga/executorch
cd executorch
git submodule update --init –recursive
./install_requirements.sh
(If not successful) pkill -f buck && ./install_requirements.sh
```

## Build OpenVINO and source environment variables:

```bash
git clone -b executorch_ov_backend https://github.com/ynimmaga/openvino
cd openvino
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON
make -j<N>
cd wheels
pip install <openvino_wheel>

cd ../..
cmake --install build --prefix <your_preferred_install_location>
cd <your_preferred_install_location>
source setupvars.sh
```

## Build gflags:

```bash
cd third-party/gflags
mkdir build
cd build
cmake ..
make -j12
```

## Build OpenVINO example:

```bash
cd ../../../examples/openvino
./openvino_build.sh
```

### AOT step:
```bash
cd aot
python aot_openvino_compiler.py --suite torchvision --model resnet50 --input_shape "(1, 3, 256, 256)" --device CPU
```

### Update the model.pte in executorch example and rebuild
```bash
cd <executorch_root>
cd examples/openvino/executor_runner
Update the path of model.pte in openvino_executor_runner.cpp at https://github.com/ynimmaga/executorch/blob/openvino_backend/examples/openvino/executor_runner/openvino_executor_runner.cpp#L20

Rebuild the example using “./openvino_build.sh”
The executable is in <executorch_root>/cmake-openvino-out/examples/openvino
./openvino_executor_runner
```
