# Qualcomm Observatory CLI

Qualcomm-specific Observatory CLI that wraps `devtools/observatory` with QNN backend patches and
accuracy lenses. Requires a QNN SDK environment (source `$QNN_SDK_ROOT/bin/envsetup.sh` before
running on-device jobs).

## Usage

### Collection mode (default)

```bash
python -m executorch.backends.qualcomm.debugger.observatory \
    [--output-html PATH] [--output-json PATH] SCRIPT [SCRIPT_ARGS...]
```

### With one or more lens recipes

Repeat the flag or comma-separate the values:

```bash
python -m executorch.backends.qualcomm.debugger.observatory \
    --lens-recipe adb --lens-recipe accuracy \
    [--output-html PATH] [--output-json PATH] \
    SCRIPT [SCRIPT_ARGS...]

python -m executorch.backends.qualcomm.debugger.observatory \
    --lens-recipe adb,accuracy \
    SCRIPT [SCRIPT_ARGS...]
```

Available recipes:

- `accuracy` — `AccuracyLens` + `PerLayerAccuracyLens` (per-stage and
  per-layer accuracy on the graph).
- `adb` — `AdbLens` (device info, transfer summary, inference command +
  logs with logcat/dmesg). See the *ADB lens* section below.

### Visualize mode (JSON → HTML, no re-execution)

```bash
python -m executorch.backends.qualcomm.debugger.observatory visualize \
    --input-json report.json --output-html report.html
```

## Qualcomm examples

Qualcomm example scripts use only absolute imports and live in directories without `__init__.py`,
so the Observatory CLI runs them as plain scripts via `runpy.run_path` (no special invocation
needed).

### Vision model (ImageNet)

```bash
source $QNN_SDK_ROOT/bin/envsetup.sh

python -m executorch.backends.qualcomm.debugger.observatory \
    --output-html /tmp/obs_vit/report.html \
    --output-json /tmp/obs_vit/report.json \
    --lens-recipe accuracy \
    examples/qualcomm/scripts/torchvision_vit.py \
    -m SM8650 -b ./build-android \
    --dataset imagenet-mini-val/ \
    -H mlgtw-linux -s <device_serial> \
    -a /tmp/obs_vit --seed 1126 --compile_only
```

### NLP model (Wikipedia sentences)

```bash
python -m executorch.backends.qualcomm.debugger.observatory \
    --output-html /tmp/obs_roberta/report.html \
    --lens-recipe accuracy \
    examples/qualcomm/oss_scripts/roberta.py \
    -m SM8650 -b ./build-android \
    -H mlgtw-linux -s <device_serial> \
    -a /tmp/obs_roberta --compile_only
```

### Compile-only (no device required)

Add `--compile_only` to any Qualcomm script to export and lower without pushing to device.
This is useful for inspecting the compilation pipeline in CI or on a dev machine.

## Available example scripts

### `examples/qualcomm/scripts/` — vision models

| Script | Model |
|---|---|
| `torchvision_vit.py` | Vision Transformer |
| `mobilenet_v2.py` | MobileNetV2 |
| `mobilenet_v3.py` | MobileNetV3 |
| `inception_v3.py` | InceptionV3 |
| `inception_v4.py` | InceptionV4 |

Dataset: ImageNet (pass with `--dataset <path>` or `-d <path>`).

### `examples/qualcomm/oss_scripts/` — NLP/open-source models

| Script | Model |
|---|---|
| `roberta.py` | RoBERTa |
| `bert.py` | BERT |
| `albert.py` | ALBERT |
| `distilbert.py` | DistilBERT |
| `eurobert.py` | EuroBERT |

Dataset: Wikipedia sentences (`wikisent2.txt`). Pass with `-d <path>`.

Common flags: `-m <SOC_MODEL>` (e.g. `SM8650`), `-b <build_folder>`, `-H <host>`,
`-s <device_serial>`, `-a <artifact_dir>`, `--compile_only`.

## Accuracy lenses (`--lens-recipe accuracy`)

Registers `AccuracyLens` and `PerLayerAccuracyLens` (with QNN dataset patches) on top of the
default `PipelineGraphCollectorLens`. These produce:

- Per-stage accuracy metrics (PSNR, cosine similarity, MSE, top-k)
- Per-layer accuracy heat-map overlaid on the graph
- Cross-stage diff labels in the left panel of the HTML report

QNN dataset patches (`lenses/qnn_dataset_patches.py`) wire the on-device inference output back
into the accuracy lens so metrics reflect true QNN outputs, not emulated CPU results.

## ADB lens (`--lens-recipe adb`)

Captures the device-side life of a QNN run by patching `SimpleADB`:
every `push`, `pull`, and the inference invocation is recorded with
exit code, duration, and (for the inference) full stdout. The HTML
report gains:

- A **Device Info** dashboard block (serial, host, soc, htp arch,
  workspace, build path).
- A compact **Transfers** dashboard block (one row per push/pull
  group) so 20+ file transfers do not crowd out the inference detail.
- A left-panel **`adb.execute`** record per inference call, showing
  the full `qnn_executor_runner ...` command with one-click copy, a
  scrollable log with errors highlighted in red, and collapsible
  `logcat -d` / `adb shell dmesg` panels fetched after the run.

Config (override via `Observatory.enable_context(config={"adb": {...}})`):

| Key | Default | Effect |
|---|---|---|
| `enabled` | `True` | Master switch. `False` disables patching. |
| `forward_to_stdout` | `True` | Tee captured inference stdout to terminal. |
| `max_stdout_bytes` | `4 * 1024 * 1024` | Cap on captured streams. |
| `fetch_logcat` | `"auto"` | `"auto"`: only after `execute()`; `True`/`False` force. |
| `fetch_dmesg` | `"auto"` | Same shape as `fetch_logcat`. |

Example combining both recipes:

```bash
python -m executorch.backends.qualcomm.debugger.observatory \
    --output-html /tmp/obs/report.html \
    --lens-recipe accuracy --lens-recipe adb \
    examples/qualcomm/oss_scripts/swin_transformer.py \
    -m SM8850 -b ./build-android \
    --dataset imagenet-mini-val/ \
    -H weilhuan-linux -s <device_serial>
```

## Two-step workflow

Collect on-device in CI, visualize locally without re-running:

```bash
# Step 1 — collect (e.g., in CI with device attached)
python -m executorch.backends.qualcomm.debugger.observatory \
    --output-html /tmp/obs/report.html \
    --output-json /tmp/obs/report.json \
    examples/qualcomm/scripts/torchvision_vit.py \
    -m SM8650 -b ./build-android -d imagenet-mini-val/ \
    -H mlgtw-linux -s <device_serial> -a /tmp/obs

# Step 2 — re-generate HTML from JSON (e.g., locally after lens update)
python -m executorch.backends.qualcomm.debugger.observatory visualize \
    --input-json /tmp/obs/report.json \
    --output-html /tmp/obs/report_v2.html
```

## Backend patches

`lenses/qnn_patches.py` installs a monkey-patch on `ptq_calibrate` so the
`PipelineGraphCollectorLens` can intercept the QNN quantization calibration stage and capture
the graph at that point. The patch is active only while the Observatory context is open.

`lenses/qnn_dataset_patches.py` wires on-device inference results into `AccuracyLens` so that
accuracy metrics use real QNN outputs.

## See also

- `backends/qualcomm/debugger/README.md` — broader Qualcomm debugger overview (QAIRT visualizer,
  intermediate output debugger)
- `devtools/observatory/README.md` — framework overview, Python API, custom lens guide
- `devtools/observatory/USAGE.md` — full CLI reference
- `devtools/observatory/lenses/LENSES.md` — built-in lens details
