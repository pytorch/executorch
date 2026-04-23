---
name: profile
description: Profile ExecuTorch model execution. Use when measuring performance, analyzing operator timing, or debugging slow models.
---

# Profile

## 1. Enable ETDump when loading
```python
program = runtime.load_program("model.pte", enable_etdump=True, debug_buffer_size=int(1e7))
```

## 2. Execute and save
```python
outputs = program.load_method("forward").execute(inputs)
program.write_etdump_result_to_file("etdump.etdp", "debug.bin")
```

## 3. Analyze with Inspector
```python
from executorch.devtools import Inspector
inspector = Inspector(etrecord="model.etrecord", etdump_path="etdump.etdp")
inspector.print_data_tabular()
```
