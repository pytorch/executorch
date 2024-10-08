## Summary
In this example, we showcase how to export a model ([phi-3-mini](https://github.com/pytorch/executorch/tree/main/examples/models/phi-3-mini)) appended with LoRA layers to ExecuTorch. The model is exported to ExecuTorch for both inference and training.

To see how you can use the model exported for training in a fully involved finetuning loop, please see our example on [LLM PTE Fintetuning](https://github.com/pytorch/executorch/tree/main/examples/llm_pte_finetuning).

## Instructions
### Step 1: [Optional] Install ExecuTorch dependencies
`./install_requirements.sh` in ExecuTorch root directory.

### Step 2: Install Requirements
- `./examples/models/phi-3-mini-lora/install_requirements.sh`

### Step 3: Export and run the model
1. Export the inference and training models to ExecuTorch.
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
