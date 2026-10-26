# QNN/HTP joint-graph training example: 4-layer toy model

A minimal, self-contained example of on-device training on the QNN/HTP
(Hexagon) backend: export a forward+backward "joint graph" for a plain
4-layer toy model, lower it to the QNN delegate, and run a real host-driven
training loop against the compiled `.pte`.

- **No quantization.** Compiled fp16 for the HTP.
- **No pretrained weights.** `ToyModel`'s weights are whatever `nn.Linear`'s
  default init produces; pass `--seed` for reproducibility.

This example builds the joint graph with the lower-level
`aot_export_joint_with_descriptors` API (see `joint_graph.py`) and lowers it
to the QNN/HTP delegate with `to_edge_transform_and_lower_to_qnn` (see
`export_toy_model.py`). Because the resulting joint `GraphModule` has no
`nn.Parameter` attributes -- every weight arrives as an anonymous `primals_N`
placeholder -- re-exporting it inside `to_edge_transform_and_lower_to_qnn`
classifies every weight as a plain input automatically.

## Minimal decomposition table

`joint_graph.py`'s `build_decompositions()` starts from
`core_aten_decompositions()`, removes everything QNN supports natively via
`get_skip_decomp_table()`, and registers only the one decomp a bias-free
ToyModel + MSE-loss graph actually needs:

- `mean.default` -> `mean.dim` (`mse_loss`'s `reduction="mean"` produces a
  full-tensor mean; QNN has no `mean.default` builder, only `mean.dim`)

The `nn.Linear` layers have no bias, so they lower straight to `mm` (for which QNN builder is present).

## One .pte per weight-sharing setting

`export_toy_model.py` compiles two methods -- `forward` (inference) and
`forward_backward` (the joint graph, used for training) -- named from the
`name` config.json field. Both methods take weights as explicit `USER_INPUT`s,
so trained weights can be fed directly at inference time without re-exporting.
Which variant gets built is controlled by the `weight_sharing` config.json
field (or `--weight_sharing`):

- `false` (default): QNN weight-sharing disabled. Compiles `forward` and
  `forward_backward` into **separate** `.pte` files (`<name>_fwd.pte` and
  `<name>_fwd_bwd.pte`) under `<output_dir>/no_ws/`. The inference runner
  uses `_fwd.pte` and calls `"forward"` directly with trained weights.
- `true`: QNN weight-sharing enabled. Compiles both methods into a **single**
  `.pte` (`<name>.pte`) under `<output_dir>/ws/`, one shared compile context
  for both methods. Weight-sharing is a hardware-only feature -- requires real Hexagon/HTP
  hardware. The inference runner uses this single `.pte` and also calls
  `"forward"` with trained weights.

`manifest.json` (written next to the `.pte`(s), in the same `ws`/`no_ws`
subdir) records `pte_name`, `fwd_pte_name` (no_ws only), `weight_sharing`,
`param_names`, `x_shape`, and `target_shape`. `export_toy_model.py` also saves
the initial weights next to it as `init_weights.safetensors`, keyed by
`param_names`.

## On-device C++ training runner

`on-device/trainer.cpp` (built as `qnn_toy_model_trainer`, wired into
`examples/qualcomm/CMakeLists.txt` behind the `BUILD_QNN_TRAINING_EXAMPLE`
CMake option, OFF by default -- pass `-DBUILD_QNN_TRAINING_EXAMPLE=ON` when
configuring `examples/qualcomm` to build it),
for exercising the training loop mechanics on real Hexagon/HTP hardware --
including the `weight_sharing=true` variant. It uses the plain
`executorch::extension::Module` API (`method_meta` + `execute`), joint graph's weights
are plain method I/O (anonymous `primals_N` placeholders), not
`nn.Parameter`-like mutable buffers.

Usage:

```bash
qnn_toy_model_trainer  <config.json> <manifest.json>  # train; writes trained_model.safetensors
qnn_toy_model_inference <config.json> <manifest.json>  # infer from trained_model.safetensors
```

The `.pte` and `init_weights.safetensors` are expected alongside `manifest.json`
(`export_toy_model.py`'s layout) -- resolved from `manifest.json`'s own
directory, not a separate flag. Initial weights are reordered to
`manifest.json`'s `param_names`; a single `x`/`target` batch is generated
on-device once via `N(0,1)`, seeded from `config.json`'s `seed` if present,
and reused every step -- the model memorizes it, so loss decreases
monotonically. `steps`/`lr`/`log_every` come from `config.json`.

To run on a connected Android device, use `run_toy_model.py` (requires
`export_toy_model.py` to have been run first):

```bash
# push + train + infer (default)
python run_toy_model.py --device <serial> --host <adb-host> \
    --soc_model SM8850 --build_folder /path/to/build-android

# push only
python run_toy_model.py ... --push_only

# skip push, train only (re-run without re-pushing)
python run_toy_model.py ... --skip_push --skip_inference

# skip push, infer only (after training already done)
python run_toy_model.py ... --skip_push --skip_train
```

Use `--weight_sharing true` to select the `ws/` variant (default: `no_ws/`).
`QNN_SDK_ROOT` must be set (same requirement as all other Qualcomm examples).

This lands `qnn_toy_model_trainer`, `qnn_toy_model_inference`, and the QNN/HTP
shared libraries into `bin/`, and `config.json`/`manifest.json`/the `.pte`(s)/
`init_weights.safetensors` into `artifacts/` -- both under
`/data/local/tmp/$(whoami)/qnn_toy_model_trainer/` on the device.

## Configuration

All scripts read defaults from `config.json` at the `training/toy_model/` root (override the path with
`--config`, or any individual field with its CLI flag). Key fields:

- `name`: base filename for the exported artifacts (`<name>_fwd.pte`, `<name>_fwd_bwd.pte`).
- `output_dir`: base artifacts directory (default `outputs/artifacts`).
  `export_toy_model.py` writes into `<output_dir>/ws/` or `<output_dir>/no_ws/`
  depending on `weight_sharing`;
- `seed`: seeds `torch.manual_seed` at export time (reproducible initial
  weights) and the on-device RNG used to generate the `x`/`target` batch.
- `steps`, `lr`, `log_every`: training-loop hyperparameters.

```bash
cd examples/qualcomm/training/toy_model

# Export: writes *.pte's + manifest.json to outputs/artifacts/no_ws/
python export/export_toy_model.py
```