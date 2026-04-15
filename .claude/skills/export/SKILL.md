---
name: export
description: Export a PyTorch model to .pte format for ExecuTorch. Use when converting models, lowering to edge, or generating .pte files.
---

# Export

## Basic pattern
```python
from executorch.exir import to_edge_transform_and_lower
from torch.export import export

exported = export(model.eval(), example_inputs)
edge = to_edge_transform_and_lower(exported)
with open("model.pte", "wb") as f:
    f.write(edge.to_executorch().buffer)
```

## Model-specific scripts
| Model | Script |
|-------|--------|
| Llama | `examples/models/llama/export_llama.py` |
| Whisper | `examples/models/whisper/` |
| Parakeet | `examples/models/parakeet/export_parakeet_tdt.py` |

## Debugging
- Draft export: `export(model, inputs, strict=False)`
- tlparse: `TORCH_LOGS="+dynamo,+export" python script.py 2>&1 | tlparse`

## Guides
- [torch.export basics](guides/torch-export.md) — `torch.export.export()`, dynamic shapes, strict mode, debugging
