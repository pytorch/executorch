## Summary
In this exmaple, we export a model ([phi-3-mini](https://github.com/pytorch/executorch/tree/main/examples/models/phi-3-mini)) appended with additional LoRA layers to ExecuTorch.

## Instructions
### Step 1: [Optional] Install ExecuTorch dependencies
`./install_requirements.sh` in ExecuTorch root directory.

### Step 2: Install TorchTune nightly
The LoRA model used is recent and is not yet officially released on `TorchTune`. To be able to run this example, you will need to run the following to install TorchTune nighly:
- `./examples/models/llava_encoder/install_requirements.sh`'

### Step 3: Export and run the model
1. Export the model to ExecuTorch.
```
python export_model.py
```

2. Run the model using an example runtime. For more detailed steps on this, check out [Build & Run](https://pytorch.org/executorch/stable/getting-started-setup.html#build-run).
```
# Clean and configure the CMake build system. Compiled programs will appear in the executorch/cmake-out directory we create here.
(rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake ..)

# Build the executor_runner target
cmake --build cmake-out --target executor_runner -j9

./cmake-out/executor_runner --model_path mini_phi3_lora.pte
```
