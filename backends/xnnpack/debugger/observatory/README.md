# XNNPack Observatory CLI

XNNPack-specific Observatory CLI that wraps `devtools/observatory` with XNNPack backend patches and accuracy lenses.

## Usage

### Collection mode (default)

```bash
python -m executorch.backends.xnnpack.debugger.observatory \
    [--output-html PATH] [--output-json PATH] SCRIPT [SCRIPT_ARGS...]
```

### With accuracy debugging

```bash
python -m executorch.backends.xnnpack.debugger.observatory \
    --lense_recipe=accuracy \
    [--output-html PATH] [--output-json PATH] \
    SCRIPT [SCRIPT_ARGS...]
```

### Visualize mode (JSON → HTML, no re-execution)

```bash
python -m executorch.backends.xnnpack.debugger.observatory visualize \
    --input-json report.json --output-html report.html
```

## XNNPack examples

`examples/xnnpack/aot_compiler.py` uses relative imports (`from . import MODEL_NAME_TO_OPTIONS`,
`from ..models import ...`) and must be executed as a Python module. The CLI handles this
automatically: when the supplied path ends in `.py` and its directory contains `__init__.py`, it
uses `runpy.run_module` instead of `runpy.run_path`.

### File path (auto-detected as module)

```bash
python -m executorch.backends.xnnpack.debugger.observatory \
    --output-html /tmp/mv2/report.html \
    --lense_recipe=accuracy \
    examples/xnnpack/aot_compiler.py \
    --model_name=mv2 --delegate --quantize --output_dir /tmp/mv2
```

### Dotted module name (explicit)

```bash
python -m executorch.backends.xnnpack.debugger.observatory \
    --output-html /tmp/mv2/report.html \
    --lense_recipe=accuracy \
    examples.xnnpack.aot_compiler \
    --model_name=mv2 --delegate --quantize --output_dir /tmp/mv2
```

### Available model names

Pass `--model_name` with any of the models defined in `examples/xnnpack/__init__.py`:

| Model | Notes |
|---|---|
| `mv2` | MobileNetV2 — fast, quantizable |
| `mv3` | MobileNetV3 |
| `resnet18` | ResNet-18 |
| `resnet50` | ResNet-50 |
| `vit` | Vision Transformer |
| `ic3` | InceptionV3 |
| `ic4` | InceptionV4 |
| `dl3` | DeepLabV3 |
| `edsr` | Super-resolution |
| `mobilebert` | MobileBERT |
| `w2l` | Wav2Letter |
| `linear` | Linear baseline |
| `add` / `add_mul` | Arithmetic baselines |
| `llama2` | Llama 2 (requires HuggingFace token) |
| `emformer_join` / `emformer_transcribe` | Speech |

Common flags: `--delegate` (XNNPACK delegation, on by default), `--quantize` (8-bit PTQ),
`--output_dir` (where the `.pte` is written).

## Accuracy lenses (`--lense_recipe=accuracy`)

Registers `AccuracyLens` and `PerLayerAccuracyLens` on top of the default
`PipelineGraphCollectorLens`. These produce:

- Per-stage accuracy metrics (PSNR, cosine similarity, MSE, top-k)
- Per-layer accuracy heat-map overlaid on the graph
- Cross-stage diff labels in the left panel of the HTML report

## Two-step workflow

Collect in one environment, visualize in another:

```bash
# Step 1 — collect
python -m executorch.backends.xnnpack.debugger.observatory \
    --output-html /tmp/mv2/report.html \
    --output-json /tmp/mv2/report.json \
    examples/xnnpack/aot_compiler.py \
    --model_name=mv2 --delegate --quantize --output_dir /tmp/mv2

# Step 2 — re-generate HTML from JSON (e.g., after lens code update)
python -m executorch.backends.xnnpack.debugger.observatory visualize \
    --input-json /tmp/mv2/report.json \
    --output-html /tmp/mv2/report_v2.html
```

## Backend patches

`lenses/xnnpack_patches.py` installs XNNPack-specific monkey-patches so the
`PipelineGraphCollectorLens` can intercept XNNPack-specific lowering steps. These patches are
active only while the Observatory context is open and are removed when it closes.

## See also

- `devtools/observatory/README.md` — framework overview, Python API, custom lens guide
- `devtools/observatory/USAGE.md` — full CLI reference
- `devtools/observatory/lenses/LENSES.md` — built-in lens details
