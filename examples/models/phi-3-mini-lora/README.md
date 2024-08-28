## Summary
In this example, we export to ExecuTorch a model ([phi-3-mini](https://github.com/pytorch/executorch/tree/main/examples/models/phi-3-mini)) appended with attention and mlp LoRA layers. The model is exported to ExecuTorch for both inference and training. Note: the exported training model can only train at the moment.

## Instructions
### Step 1: [Optional] Install ExecuTorch dependencies
`./install_requirements.sh` in ExecuTorch root directory.

### Step 2: Install Requirements
- `./examples/models/phi-3-mini-lora/install_requirements.sh`

### Step 3: Export and run the model
1. Export the inferenace and training models to ExecuTorch.
```
python export_model.py
```

2. Run the inference model using an example runtime. For more detailed steps on this, check out [Build & Run](https://pytorch.org/executorch/stable/getting-started-setup.html#build-run).
```
# Clean and configure the CMake build system. Compiled programs will appear in the executorch/cmake-out directory we create here.
(rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake ..)

# Build the executor_runner target
cmake --build cmake-out --target executor_runner -j9

# Run the model for inference.
./cmake-out/executor_runner --model_path phi3_mini_lora.pte
```
