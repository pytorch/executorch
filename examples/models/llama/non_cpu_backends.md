
# Running Llama 3/3.1 8B on non-CPU backends

### QNN
Please follow [the instructions](https://pytorch.org/executorch/main/llm/build-run-llama3-qualcomm-ai-engine-direct-backend) to deploy Llama 3 8B to an Android smartphone with Qualcomm SoCs.

### MPS
Export:
```
python -m examples.models.llama2.export_llama --checkpoint llama3.pt --params params.json -kv --disable_dynamic_shape --mps --use_sdpa_with_kv_cache -d fp32 -qmode 8da4w -G 32 --embedding-quantize 4,32
```

After exporting the MPS model .pte file, the [iOS LLAMA](https://pytorch.org/executorch/main/llm/llama-demo-ios) app can support running the model. ` --embedding-quantize 4,32` is an optional args for quantizing embedding to reduce the model size.

### CoreML
Export:
```
python -m examples.models.llama2.export_llama --checkpoint llama3.pt --params params.json -kv --disable_dynamic_shape --coreml --coreml-ios 18 --coreml-quantize b4w
```

After exporting the CoreML model .pte file, please [follow the instruction to build llama runner](https://github.com/pytorch/executorch/tree/main/examples/models/llama#step-3-run-on-your-computer-to-validate) with CoreML flags enabled as the instruction described.

### MTK
Please [follow the instructions](https://github.com/pytorch/executorch/tree/main/examples/mediatek#llama-example-instructions) to deploy llama3 8b to an Android phones with MediaTek chip
