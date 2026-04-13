# Troubleshooting

This page describes common issues when using the MLX backend and how to debug them.

## Debug Logging

### AOT (export/compilation) debugging

Set `ET_MLX_DEBUG=1` during export to see detailed debug logging from the partitioner and preprocessor — including ops-to-not-decompose lists, graph dumps, per-node support decisions, and serialization details:

```bash
ET_MLX_DEBUG=1 python my_export_script.py
```

### Runtime per-op logging

Per-op logging prints each MLX instruction as it executes, showing op names and tensor IDs. This is useful for correlating runtime behavior with the compiled graph.

When using **pybindings** (i.e. `python install_executorch.py`), per-op logging is compiled in by default. Just set the environment variable:

```bash
ET_MLX_ENABLE_OP_LOGGING=1 python my_inference_script.py
```

For **C++ builds**, you need to build with the debug preset first (which compiles in the logging code), then set the environment variable:

```bash
# Build with debug preset
cmake --workflow --preset mlx-debug

# Run with per-op logging enabled
ET_MLX_ENABLE_OP_LOGGING=1 ./cmake-out/my_app model.pte
```

The release preset (`mlx-release`) strips the logging code for performance.

## Inspecting `.pte` Files

The MLX backend includes a `.pte` inspector for debugging exported models. It can parse the ExecuTorch program structure, extract and decode the MLX delegate payload, and display instructions, tensor metadata, and I/O maps.

### Basic usage

Dump the full PTE structure as JSON:

```bash
python -m executorch.backends.mlx.pte_inspector model.pte
```

### MLX summary

Show a high-level summary of the MLX delegate (tensor counts, I/O maps, mutable buffers):

```bash
python -m executorch.backends.mlx.pte_inspector model.pte --mlx-summary
```

### MLX instructions

Show every instruction in the compiled graph with operands and parameters. This is useful for verifying quantization, inspecting fused patterns, and debugging incorrect outputs:

```bash
python -m executorch.backends.mlx.pte_inspector model.pte --mlx-instructions
```

### Extract delegate payload

Extract the raw MLX delegate payload to a binary file:

```bash
python -m executorch.backends.mlx.pte_inspector model.pte --extract-delegate mlx -o delegate.bin
```

Parse and dump the extracted payload as JSON:

```bash
python -m executorch.backends.mlx.pte_inspector model.pte --extract-delegate mlx --parse-mlx -o mlx_graph.json
```

### All options

| Flag | Description |
|------|-------------|
| `--mlx-summary` | High-level summary (tensor counts, I/O maps) |
| `--mlx-instructions` | Detailed instruction list with operands |
| `--extract-delegate ID` | Extract raw delegate payload by ID |
| `--parse-mlx` | Parse extracted MLX payload to JSON (use with `--extract-delegate mlx`) |
| `--delegate-index N` | Index of delegate to extract (0-based, default: first match) |
| `--format json`/`summary` | Output format (default: json) |
| `-o FILE` | Write output to file instead of stdout |

## Common Issues

### Metal compiler not found

**Error:** `xcrun -sdk macosx --find metal` fails.

**Solution:** Install the full Xcode application (not just Command Line Tools). The Metal compiler ships with Xcode. If Xcode is installed but not selected:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

### `MLXPartitioner must be used with to_edge_transform_and_lower()`

**Error:** RuntimeError when using the legacy `to_edge()` + `to_backend()` workflow.

**Solution:** Use `to_edge_transform_and_lower()` instead:

```python
import torch
from executorch.backends.mlx import MLXPartitioner
from executorch.exir import to_edge_transform_and_lower

et_program = to_edge_transform_and_lower(
    torch.export.export(model, example_inputs),
    partitioner=[MLXPartitioner()],
).to_executorch()
```

### Unsupported ops falling back to CPU

If some ops in your model are not supported by the MLX delegate, they will automatically fall back to ExecuTorch's portable CPU runtime. This is expected behavior but may impact performance.

To see which ops are unsupported, enable debug logging:

```bash
ET_MLX_DEBUG=1 python my_export_script.py
```

The partitioner logs a summary of unsupported ops with reasons during partitioning. You can also check the [supported operators](mlx-op-support.md) page.

### Dynamic shapes not preserved

If dynamic shapes are being lost during export, ensure you are using `to_edge_transform_and_lower()` (not the legacy workflow). The MLX partitioner's `ops_to_not_decompose()` mechanism preserves higher-level ops that carry shape information, and it pulls `sym_size` nodes into the delegate partition to keep shapes dynamic at runtime.
