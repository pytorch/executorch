'''
Example to convert .gguf files into .pte format.

1. Load our model using transformers/gguf
2. Torch export
3. Executorch lowering and export to .pte
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from torch.export import export
import torch

model_id = "bartowski/SmolLM2-135M-Instruct-GGUF" # Here we would have our HF model in GGUF form we wish to convert
filename = "SmolLM2-135M-Instruct-Q8_0.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
print(f"Model weights dtype: {model.dtype}")
model.eval()

# Generate some sample input for our torch export
sample_inputs = tokenizer("Plants create energy through a process known as", return_tensors="pt",)
print(sample_inputs)
print(sample_inputs["input_ids"].shape)
print(sample_inputs["attention_mask"].shape)

sample_inputs = (sample_inputs["input_ids"], sample_inputs["attention_mask"],)

# Torch export followed by ET lowering and export
exported_program = export(model, sample_inputs)
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner = [XnnpackPartitioner()]
).to_executorch()

with open("model.pte", "wb") as file:
    file.write(executorch_program.buffer)
