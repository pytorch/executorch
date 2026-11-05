# Quantization

The Core AI backend compresses models with [coreai-opt](https://pypi.org/project/coreai-opt/), which is installed with the backend's pinned requirements. ExecuTorch wraps it in two classes:

- `CoreAIQuantizer` for linear quantization, which stores weights (and optionally activations) in a lower-precision integer or floating-point format. It follows the PT2E prepare, calibrate and convert flow.
- `CoreAIPalettizer` for palettization, which replaces each weight with a small lookup table of values plus, for each element, an index into that table.

Both run on your model before the final `torch.export` for lowering. You then export and lower the result as usual.

## Supported Quantization Schemes

| Scheme | Front-end | Configuration | Granularity |
|---|---|---|---|
| INT8 weights and activations | `CoreAIQuantizer` | `QuantizerConfig()` (the default) | Weights per channel; activations per tensor, static |
| INT8 weight-only | `CoreAIQuantizer` | `QuantizerConfig.presets.w8()` | Per channel |
| INT4 weight-only | `CoreAIQuantizer` | `QuantizerConfig.presets.w4()` | Per channel |
| INT4 weight-only, per block | `CoreAIQuantizer` | `QuantizerConfig.presets.w4_per_block()` | Blocks of 32 elements |
| INT2, FP8 or FP4 weight-only | `CoreAIQuantizer` | A custom configuration. See [Other weight formats](#other-weight-formats). | Per channel or per block |
| 4- or 6-bit palettization | `CoreAIPalettizer` | `KMeansPalettizerConfig.presets.w4()` or `.w6()` | One lookup table per group of 16 output channels |
| 8-bit palettization | `CoreAIPalettizer` | `KMeansPalettizerConfig.presets.w8()` | One lookup table per weight |

The integer presets are all symmetric. Activation quantization is static: calibration fixes each activation's scale, which is then stored in the model, rather than computing it at runtime. `w4_per_block()` gives each block of 32 weights its own scale instead of `w4()`'s single scale per output channel. The finer scales can preserve more accuracy, at the cost of storing more of them.

`QuantizerConfig` comes from `coreai_opt.quantization` and `KMeansPalettizerConfig` from `coreai_opt.palettization`. The presets cover common cases; construct a configuration yourself for finer control.

## Linear Quantization with `CoreAIQuantizer`

`prepare` exports the model and inserts fake quantization. You then run calibration data through the prepared model, and `convert` produces the quantized model:

```python
import torch
from coreai_opt.quantization import QuantizerConfig
from executorch.backends.apple.coreai import (
    CoreAIPartitioner,
    get_default_compile_config,
    get_default_passes,
)
from executorch.backends.apple.coreai.quantizer import CoreAIQuantizer
from executorch.exir import to_edge_transform_and_lower

quantizer = CoreAIQuantizer(model, QuantizerConfig())
prepared = quantizer.prepare(example_inputs)

with quantizer.calibration_mode():
    for inputs in calibration_data:
        prepared(*inputs)

quantized = quantizer.convert()

et_program = to_edge_transform_and_lower(
    torch.export.export(quantized, example_inputs),
    transform_passes=get_default_passes(),
    partitioner=[CoreAIPartitioner()],
    compile_config=get_default_compile_config(),
).to_executorch()
```

- **Calibration data:** the default configuration quantizes activations, so calibrate with inputs representative of real use. The weight-only presets don't need calibration.
- **Dynamic shapes:** pass `dynamic_shapes` to `prepare` if the model has dynamic inputs.
- **Output:** unlike standard PT2E, which produces generic quantize and dequantize ops, `convert` returns a model that already uses Core AI's quantized ops, ready to export.

For quantization-aware training, train the prepared model inside `quantizer.training_mode()` instead of calibrating it, then call `convert`.

### Other weight formats

The presets cover INT8 and INT4. For other weight formats, build a weight-only configuration that sets a `QuantizationSpec` for each layer's weight and leaves activations unquantized:

```python
import torch
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    QuantizationSpec,
    QuantizerConfig,
)

def weight_only(dtype, scale_dtype=None, granularity=None):
    spec = QuantizationSpec(
        dtype=dtype,
        scale_dtype=scale_dtype,
        granularity=granularity or {"type": "per_channel", "axis": 0},
    )
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": spec},
            op_input_spec=None,
            op_output_spec=None,
        )
    )

config = weight_only(torch.float8_e4m3fn)
```

Then pass the configuration to `CoreAIQuantizer` as usual. These formats run on Core AI:

| Format | `dtype` | `scale_dtype` | `granularity` |
|---|---|---|---|
| INT2 | `torch.int2` | default | per channel, or per block such as `{"type": "per_block", "block_size": (1, 16)}` |
| FP8 | `torch.float8_e4m3fn` or `torch.float8_e5m2` | default | per channel |
| FP8 microscaling | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` | per block of 32: `{"type": "per_block", "block_size": (1, 32)}` |
| FP4 microscaling | `torch.float4_e2m1fn_x2` | default (the scale is always `float8_e8m0fnu`) | per block of 32 |

INT2 keeps only four levels per weight, so check your model's accuracy carefully. Per-block scales make the model larger than per-channel scales.

## Palettization with `CoreAIPalettizer`

`CoreAIPalettizer` uses k-means clustering to build each weight's lookup table. It needs no calibration data:

```python
from coreai_opt.palettization import KMeansPalettizerConfig
from executorch.backends.apple.coreai.palettizer import CoreAIPalettizer

palettizer = CoreAIPalettizer(model, KMeansPalettizerConfig.presets.w4())
palettizer.prepare(example_inputs)
palettized = palettizer.finalize()

et_program = to_edge_transform_and_lower(
    torch.export.export(palettized, example_inputs),
    transform_passes=get_default_passes(),
    partitioner=[CoreAIPartitioner()],
    compile_config=get_default_compile_config(),
).to_executorch()
```

`finalize` returns a regular `nn.Module`, ready to export. Palettization only compresses weights; activations stay in floating point.

Optional settings:

- **Sensitivity-weighted clustering:** after `prepare`, run data through `with palettizer.calibration_mode(loss_fn=..., sensitivity_path=...) as calibrator:`, calling `calibrator.step(output, target)` for each batch. This saves per-weight sensitivities to `sensitivity_path`; pass that path to a later `prepare` so clustering favors the weights that matter most.
- **Large models:** pass `num_workers` to `prepare` to parallelize clustering, and `mmap_dir` to `finalize` to memory-map large lookup tables rather than holding them in memory.

`coreai-opt` builds its k-means extension on first use, which requires `ninja` on your `PATH` (`pip install ninja`).
