## Summary
In this example, we initiate the process of running multi modality through ExecuTorch.
- Demonstrate how to export the image encoder model in the [LLava](https://github.com/haotian-liu/LLaVA) multimodal model.
- Provide TODO steps on how to use the exported .pte file and the existing [exported Llama2 model](https://github.com/pytorch/executorch/tree/main/examples/models/llama2), to build the multimodal pipeline.

## Instructions
Note that this folder does not host the pretrained LLava model. 
- To have Llava available, follow the [Install instructions](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install) in the LLava github. Follow the licence in the specific repo when using L
- Since the pytorch model version may not be updated, `cd executorch`, run `./install_requirements.sh`.
- Run `python3 -m examples.portable.scripts.export --model_name="llava_encoder"`. The llava_encoder.pte file will be generated. 

## TODO
- Write the pipeline in cpp
  - Have image and text prompts as inputs.
  - Call image processing functions to preprocess the image tensor.
  - Load the llava_encoder.pte model, run it using the image tensor.
  - The output of the encoder can be combined with the prompt, as inputs to the llama model. Call functions in llama_runner.cpp to run the llama model and get outputs. 