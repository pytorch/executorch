# Partitioner API

The `EnnPartitioner` API is the primary entrypoint when exporting a model to the Samsung
Exynos backend. The partitioner is responsible for determining which parts of the model
should be lowered to the backend and also provides an interface for configuring the
behaviour of the backend.

Currently, the configuration options for `EnnPartitioner` can be generated automatically
using the `gen_samsung_backend_compile_spec` API. For instance,

```python
from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)

from executorch.exir import to_edge_transform_and_lower

chipset = "E9955"
compile_specs = [gen_samsung_backend_compile_spec(chipset)]

et_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[EnnPartitioner(compile_specs)],
).to_executorch()
```

At the moment, only `"E9955"` is supported as a valid chipset name, which corresponds to
the Exynose 2500 SoC. Support for additional chipsets will be added in the future.
