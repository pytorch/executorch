# Runtime API

## executorch.runtime (preferred)
```python
from executorch.runtime import Runtime, Program, Method
runtime = Runtime.get()
program = runtime.load_program(Path("model.pte"))
outputs = program.load_method("forward").execute(inputs)
```

## portable_lib (low-level)
```python
from executorch.extension.pybindings.portable_lib import _load_for_executorch
module = _load_for_executorch("model.pte")
outputs = module.forward(inputs)
```

## Missing kernel fixes

If runtime shows missing kernel errors, import the kernel module before loading:

```python
# Missing quantized kernels (e.g., quantized_decomposed::embedding_byte.out)
from executorch.kernels import quantized

# Missing LLM custom ops (e.g., llama::custom_sdpa.out, llama::update_cache.out)
from executorch.extension.llm.custom_ops import custom_ops
```
