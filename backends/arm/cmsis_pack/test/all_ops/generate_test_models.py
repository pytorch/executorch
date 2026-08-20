#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Export a self-contained test .pte for every operator recipe.

For each operator component the pack ships (op_guards.discover_components), this
looks up its recipe in op_recipes and exports a tiny model to a .pte. Each .pte
embeds its own test data as constant methods -- nb_inputs, nb_outputs, atol,
rtol, channel_last, input_<i>, output_<i> -- so the all-ops consumer firmware
can run every test from the model file alone; manifest.json only maps ops to
.pte names.

Coverage is reconciled up front: every discovered component must have either a
recipe or an explicit skip reason, otherwise the run fails (no silent gaps).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from executorch.exir import ExecutorchProgramManager

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parents[1] / "scripts"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SCRIPTS))

import op_recipes  # type: ignore[import-not-found]  # noqa: E402
from op_guards import (  # type: ignore[import-not-found]  # noqa: E402
    discover_components,
)


def _assert_executorch_from_source(source_dir: Path) -> None:
    """Fail fast if the imported ``executorch`` is not the one under
    source_dir.

    The pack ships C++ kernels copied straight from the repo tree, but the test
    models are exported by importing ``executorch`` as a Python package. If that
    package resolves to a stale site-packages install rather than this checkout,
    the exported .pte carries OLD op signatures (e.g. quantized_add without the
    fused activation_min/max args, quantized_conv2d without the AoT scratch
    tensor) while the pack registers the CURRENT arity -- so the op links but
    fails at runtime with a KernelCall arity mismatch. Catch that skew here
    instead of on the FVP.
    """
    import executorch  # noqa: PLC0415

    paths = list(getattr(executorch, "__path__", []) or [])
    file = getattr(executorch, "__file__", None)
    if file:
        paths.append(str(Path(file).parent))
    source_dir = source_dir.resolve()
    if any(
        source_dir in Path(p).resolve().parents or Path(p).resolve() == source_dir
        for p in paths
    ):
        return
    raise SystemExit(
        "executorch is imported from outside the source tree, so the exported "
        "models would not match the pack built from it:\n"
        f"    imported executorch: {paths or '<unknown>'}\n"
        f"    --source-dir:        {source_dir}\n"
        "Reinstall executorch from this checkout (./install_executorch.sh or "
        "pip install -e . --no-build-isolation) before exporting, or set "
        "ALLOW_STALE_EXECUTORCH=1 to bypass (models may fail on device)."
    )


def _get_number_of_outputs(outputs) -> int:
    if isinstance(outputs, torch.Tensor):
        return 1
    if isinstance(outputs, (tuple, list)):
        return len(outputs)
    raise TypeError(f"unsupported output type {type(outputs)}")


def _mk_metadata(
    inputs: tuple, outputs: tuple | torch.Tensor, atol: float, rtol: float
) -> dict:
    metadata: dict[str, Any] = {
        "nb_inputs": len(inputs),
        "nb_outputs": _get_number_of_outputs(outputs),
        "atol": atol,
        "rtol": rtol,
    }

    # Store detached leaf clones: the Ethos-U partitioner path deepcopies the
    # constant_methods dict (to_backend copies self._config_methods), and a
    # non-leaf graph tensor -- e.g. a quantized conv's output -- cannot be
    # deepcopied. detach().clone() severs that history without changing values.
    def _leaf(t: torch.Tensor) -> torch.Tensor:
        return t.detach().clone()

    # We assume that when one input tensor is channel_last, all others are too.
    # This assumption holds for the operators tested.
    channel_last = False
    for i, t in enumerate(inputs):
        metadata[f"input_{i}"] = _leaf(t)
        if t.is_contiguous(memory_format=torch.channels_last):
            channel_last = True
    if isinstance(outputs, torch.Tensor):
        metadata["output_0"] = _leaf(outputs)
    else:
        for i, t in enumerate(outputs):
            metadata[f"output_{i}"] = _leaf(t)

    # Inputs/outputs are exported channel-first: to_edge_transform_and_lower
    # does not preserve the memory format of tensors exported as
    # constant_methods (looks like an upstream bug). Record the intended
    # memory format so consumers that compare raw storage can permute; the
    # in-tree main.cpp copies/compares via per-tensor strides and does not
    # need it.
    metadata["channel_last"] = channel_last

    return metadata


def _export_portable(
    model: torch.nn.Module,
    inputs: tuple,
    recipe: op_recipes.Recipe,
    display_quantized_values: bool = False,
    display_metadata: bool = False,
    target_core: str = "m55",  # uniform exporter signature; Cortex-M-only
) -> ExecutorchProgramManager:
    from executorch.exir import EdgeCompileConfig, to_edge

    model = model.eval()
    exported = torch.export.export(model, inputs, strict=True)

    expected = model(*inputs)
    metadata = _mk_metadata(inputs, expected, recipe.atol, recipe.rtol)
    if display_metadata:
        print(metadata)
    # The pack ships the full portable op set, including ops outside the Core
    # ATen opset (bitwise shifts, unfold, var_mean.correction, ...), so skip the
    # core-ATen IR validity gate; to_executorch still lowers to the .out kernels.
    edge = to_edge(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        constant_methods=metadata,
    )
    return edge.to_executorch()


def _shapes(outputs):
    if isinstance(outputs, (tuple, list)):
        return [t.shape for t in outputs]
    return outputs.shape


def _compute_test_threshold(actual, expected):
    """Worst-case abs/rel error over every output leaf.

    Outputs are not always a single float tensor: var_mean, topk, split and
    the other multi-output recipes return tuples, and the comparison and
    logical recipes return bool tensors, which cannot be subtracted. Both are
    compared leaf-wise in float space, so an exact match yields 0.0/0.0.
    """
    a_leaves = actual if isinstance(actual, (tuple, list)) else (actual,)
    e_leaves = expected if isinstance(expected, (tuple, list)) else (expected,)

    atol = 0.0
    rtol = 0.0
    for a, e in zip(a_leaves, e_leaves):
        a = a.to(torch.float32)
        e = e.to(torch.float32)
        abs_err = (a - e).abs()
        atol = max(atol, abs_err.max().item())

        # Avoid division by zero
        mask = e.abs() > 1e-12
        if mask.any():
            rtol = max(rtol, (abs_err[mask] / e.abs()[mask]).max().item())
    return atol, rtol


def _export_cortex_m(
    model: torch.nn.Module,
    inputs: tuple,
    recipe: op_recipes.Recipe,
    display_quantized_values: bool = False,
    display_metadata: bool = False,
    target_core: str = "m55",
) -> ExecutorchProgramManager:
    from executorch.backends.cortex_m.edge_compile_config import (
        cortex_m_edge_compile_config,
    )
    from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
        CortexMPassManager,
    )
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
    from executorch.exir import to_edge_transform_and_lower
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    # The pass manager sizes each op's AoT scratch buffer for this core's
    # CMSIS-NN path (M55 -> MVE, M7/M4 -> DSP, M0+ -> SCALAR). Sizing for the
    # actual deployment core avoids the runtime scratch top-up: a DSP-class core
    # needs more scratch than MVE for the depthwise/avg_pool danger shapes.
    target_config = CortexMTargetConfig(cpu=CortexM[target_core.upper()])

    model = model.eval()
    expected = model(*inputs)

    captured = torch.export.export(model, inputs, strict=True).module()
    prepared = prepare_pt2e(captured, CortexMQuantizer())
    prepared(*inputs)  # calibrate
    quantized = convert_pt2e(prepared)
    move_exported_model_to_eval(quantized)
    actual = quantized(*inputs)
    if display_quantized_values:
        print("=== quantized values ===")
        print("Expected:", expected, _shapes(expected))
        print("Actual:", actual, _shapes(actual))
    atol, rtol = _compute_test_threshold(actual, expected)

    metadata = _mk_metadata(inputs, expected, atol, rtol)
    if display_metadata:
        print(metadata)

    exported = torch.export.export(quantized, inputs, strict=True)

    edge = to_edge_transform_and_lower(
        exported,
        compile_config=cortex_m_edge_compile_config(),
        constant_methods=metadata,
    )
    edge._edge_programs["forward"] = CortexMPassManager(
        edge.exported_program(), target_config=target_config
    ).transform()
    return edge.to_executorch()


def _export_ethos_u(
    model: torch.nn.Module,
    inputs: tuple,
    recipe: op_recipes.Recipe,
    display_quantized_values: bool = False,
    display_metadata: bool = False,
    target_core: str = "m55",  # uniform exporter signature; Cortex-M-only
) -> ExecutorchProgramManager:
    """Lower a recipe for the Corstone-320 Ethos-U85 variant.

    Every op is quantized and offered to the EthosUPartitioner: whatever Vela
    can take runs on the NPU, the rest stays on the host core (the partitioner
    leaves un-partitionable nodes on the CPU). Delegation is not all-or-nothing
    here: a recipe that Vela cannot take is not an error, it simply falls back.
    Operators in op_recipes.SKIPS are still skipped -- those have no recipe at
    all, so they cannot be exported for any target -- as are the few in
    op_recipes.ETHOS_U_SKIPS, which this flow cannot lower for the documented
    backend reasons but which still run on the CPU variants.
    """
    from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config,
    )
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    # ethos-u85-256 must match the FVP's mps4_board.subsystem.ethosu.num_macs=256
    # (run.sh). The memory mode governs which AXI port Vela expects const
    # (weights) and arena (scratch) on; it must match how the firmware lays out
    # memory. This all-ops image is entirely DDR-resident with no dedicated NPU
    # SRAM, so Sram_Only (const+arena+cache all on one port, weights co-located
    # with the DDR-loaded blob) is correct -- Shared_Sram/Dedicated_Sram put
    # weights on a separate AXI port the runtime does not populate, so the NPU
    # fetches the wrong region and rejects the stream (invalid_weight_stream).
    # Overridable via ETHOS_MEMORY_MODE / ETHOS_SYSTEM_CONFIG for experiments.
    compile_spec = EthosUCompileSpec(
        target="ethos-u85-256",
        system_config=os.environ.get("ETHOS_SYSTEM_CONFIG", "Ethos_U85_SYS_DRAM_Mid"),
        memory_mode=os.environ.get("ETHOS_MEMORY_MODE", "Sram_Only"),
    )

    model = model.eval()
    expected = model(*inputs)

    captured = torch.export.export(model, inputs, strict=True).module()
    _strip_guards_fn(captured)
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(captured, quantizer)
    prepared(*inputs)  # calibrate
    quantized = convert_pt2e(prepared)
    actual = quantized(*inputs)
    if display_quantized_values:
        print("=== quantized values ===")
        print("Expected:", expected, _shapes(expected))
        print("Actual:", actual, _shapes(actual))
    atol, rtol = _compute_test_threshold(actual, expected)

    metadata = _mk_metadata(inputs, expected, atol, rtol)
    if display_metadata:
        print(metadata)

    exported = torch.export.export(quantized, inputs, strict=True)
    _strip_guards_fn(exported.graph_module)
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[EthosUPartitioner(compile_spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        constant_methods=metadata,
    )
    return edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )


def _strip_guards_fn(gm) -> None:
    """Drop the dead ``_guards_fn`` call_module node recent torch export emits.

    The Arm annotation/decomposition passes iterate over the graph and reject
    unexpected ``call_module`` nodes (``DecomposeSelectScatterPass: call_module
    is not supported``), so remove it before lowering to the Ethos-U delegate.
    """
    changed = False
    for node in list(gm.graph.nodes):
        if node.op == "call_module" and str(node.target) == "_guards_fn":
            gm.graph.erase_node(node)
            changed = True
    if changed:
        gm.graph.eliminate_dead_code()
        gm.recompile()


def _is_delegated(program: ExecutorchProgramManager) -> bool:
    """True if the forward graph contains an Ethos-U delegate call, i.e. Vela
    took at least part of the graph onto the NPU (the rest, if any, stays on
    the host core)."""
    return any(
        node.op == "call_function" and "executorch_call_delegate" in str(node.target)
        for node in program.exported_program("forward").graph_module.graph.nodes
    )


def _assert_kernel_present(
    program: ExecutorchProgramManager, category: str, op_name: str
) -> None:
    """Fail the export when the graph silently fell back off the named kernel.

    A Cortex-M recipe whose quantization is rejected still exports and often
    still passes numerically -- as float aten:: kernels plus the boundary
    quantize/dequantize (this happened for the conv recipes when their inputs
    were channel-first). Require the op's own cortex_m:: kernel in the final
    forward graph so such fallbacks fail loudly at export time.
    """
    if category != "Cortex-M":
        return
    wanted = f"cortex_m.{op_name}"
    targets = [
        str(node.target).replace("::", ".")
        for node in program.exported_program("forward").graph_module.graph.nodes
        if node.op == "call_function"
    ]
    if not any(wanted in target for target in targets):
        cortex_m_targets = sorted({t for t in targets if "cortex_m" in t})
        raise RuntimeError(
            f"forward graph contains no {wanted} kernel; cortex_m nodes "
            f"present: {cortex_m_targets or 'none'} (recipe fell back to "
            "float aten kernels?)"
        )


def _schema_arity(source_dir: Path, op_name: str) -> int | None:
    """Positional-arg count of ``cortex_m::<op_name>.out`` from the shipped
    operators.yaml, or None if not found. Args after ``*,`` (the out kwarg) are
    excluded -- the exported call node carries only positional args.
    """
    yaml_path = source_dir / "backends" / "cortex_m" / "ops" / "operators.yaml"
    if not yaml_path.is_file():
        return None
    prefix = f"cortex_m::{op_name}.out("
    for line in yaml_path.read_text().splitlines():
        idx = line.find(prefix)
        if idx == -1:
            continue
        sig = line[idx + len(prefix) :]
        sig = sig[: sig.find(") ->")] if ") ->" in sig else sig
        positional = sig.split(", *,")[0] if ", *," in sig else sig.split(",*,")[0]
        return len([t for t in positional.split(",") if t.strip()])
    return None


def _assert_arity_matches_schema(
    program: ExecutorchProgramManager,
    source_dir: Path,
    category: str,
    op_name: str,
) -> None:
    """Fail if the exported cortex_m call has fewer positional args than the
    shipped schema -- the signature-skew symptom of exporting with a stale
    executorch (see _assert_executorch_from_source).

    Such a .pte links against the pack but fails at runtime with a
    KernelCall arity mismatch.
    """
    if category != "Cortex-M":
        return
    want = _schema_arity(source_dir, op_name)
    if want is None:
        return
    wanted = f"cortex_m.{op_name}"
    for node in program.exported_program("forward").graph_module.graph.nodes:
        if node.op == "call_function" and wanted in str(node.target).replace("::", "."):
            got = len(node.args)
            if got != want:
                raise RuntimeError(
                    f"{wanted} exported with {got} positional args but the "
                    f"shipped schema declares {want}; the .pte would fail at "
                    "runtime (KernelCall arity mismatch). This is the stale-"
                    "executorch signature skew -- reinstall executorch from "
                    "this checkout before exporting."
                )
            return


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Export a self-contained test .pte for every operator recipe"
    )
    parser.add_argument(
        "--source-dir", "-s", required=True, help="repo root / staged pack tree"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="where to write models/ + manifest.json",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "keep sweeping after a failed export (full diagnostics at the end) "
            "instead of aborting at the first one; the exit status still "
            "reflects failures either way"
        ),
    )
    parser.add_argument(
        "--ethos-u",
        action="store_true",
        help=(
            "Export the Corstone-320 Ethos-U85 variant: every op with a recipe "
            "is lowered through Vela; whatever partitions runs on the NPU, the "
            "rest falls back to the host core. Coverage only -- the delegated "
            "path has known numerical deviations (see SKIPPED_OPS.md)."
        ),
    )
    parser.add_argument(
        "--target-core",
        default="m55",
        help=(
            "Cortex-M core the Cortex-M ops' AoT scratch is sized for "
            "(m55 -> MVE, m7/m4 -> DSP, m0plus -> SCALAR). Default m55. Size for "
            "the deployment core to avoid the runtime scratch top-up."
        ),
    )
    parser.add_argument(
        "--display-quantized-values",
        action="store_true",
        help="display quantized values during export",
    )
    parser.add_argument(
        "--display-metadata",
        action="store_true",
        help="display metadata during export",
    )
    # Model explorer (with pte extension) or Netron are not able to display
    # all the .pte in the right way.
    # So, another way to check is to print the final graph.
    parser.add_argument(
        "--display-graph",
        action="store_true",
        help="display graph during export",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guard against exporting with a stale installed executorch (see the
    # function docstring); the resulting .pte would mismatch the pack's kernel
    # arity and fail on device. ALLOW_STALE_EXECUTORCH=1 bypasses for debugging.
    if os.environ.get("ALLOW_STALE_EXECUTORCH") != "1":
        _assert_executorch_from_source(source_dir)

    components = discover_components(source_dir)
    keys = {(c.category, c.name) for c in components}

    # No-silent-gap reconciliation: every component must have a recipe or a skip.
    gaps = sorted(keys - op_recipes.all_keys())
    if gaps:
        raise SystemExit(
            "Operators with no recipe and no skip reason (add either to op_recipes.py):\n"
            + "\n".join(f"    [{cat}] {name}" for cat, name in gaps)
        )

    exporters = {
        "Portable": _export_portable,
        "Quantized": _export_portable,
        "Cortex-M": _export_cortex_m,
    }
    manifest: list = []
    skipped: list = []
    failed: list = []

    for component in components:
        key = (component.category, component.name)
        if key in op_recipes.SKIPS:
            skipped.append((component.category, component.name, op_recipes.SKIPS[key]))
            continue
        if args.ethos_u and key in op_recipes.ETHOS_U_SKIPS:
            skipped.append(
                (component.category, component.name, op_recipes.ETHOS_U_SKIPS[key])
            )
            continue
        recipe = op_recipes.RECIPES[key]
        # Ethos-U variant: route every recipe through the delegated exporter.
        # Whatever Vela partitions runs on the NPU, the rest falls back to the
        # host core, so the per-category CPU exporters do not apply here.
        exporter = _export_ethos_u if args.ethos_u else exporters[component.category]
        cat_id = component.category.lower().replace("-", "_")
        op_name = out_dir / f"{cat_id}__{component.name}"
        op_pte = op_name.with_suffix(".pte")
        try:
            if args.display_quantized_values or args.display_metadata:
                print(f"=== exporting {component.category}/{component.name} ===")
            model, inputs = recipe.make()

            pte = exporter(
                model,
                inputs,
                recipe,
                display_quantized_values=args.display_quantized_values,
                display_metadata=args.display_metadata,
                target_core=args.target_core,
            )
            if args.display_graph:
                _ = pte.exported_program("forward").graph_module.print_readable()
            # The Cortex-M kernel-presence + arity checks are about the CPU
            # quantized path and are meaningless once ops are delegated to the
            # NPU (the delegate subsumes the cortex_m call).
            if not args.ethos_u:
                _assert_kernel_present(pte, component.category, component.name)
                _assert_arity_matches_schema(
                    pte, source_dir, component.category, component.name
                )

            op_pte.write_bytes(bytes(pte.buffer))

            entry = {
                "op": component.name,
                "category": component.category,
                "name": op_name.name,
            }
            if args.ethos_u:
                entry["delegated"] = _is_delegated(pte)
            manifest.append(entry)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
            failed.append(
                (component.category, component.name, f"{type(exc).__name__}: {exc}")
            )
            if not args.continue_on_error:
                traceback.print_exc()

    # Variant shapes: extra test models for existing components (op_recipes.
    # VARIANTS). Same export flow; kernel-presence/arity checks validate
    # against the variant's BASE component name.
    n_variants = 0
    for (category, variant_name), (base_name, recipe) in op_recipes.VARIANTS.items():
        exporter = _export_ethos_u if args.ethos_u else exporters[category]
        cat_id = category.lower().replace("-", "_")
        op_name = out_dir / f"{cat_id}__{variant_name}"
        op_pte = op_name.with_suffix(".pte")
        try:
            model, inputs = recipe.make()
            pte = exporter(
                model,
                inputs,
                recipe,
                display_quantized_values=args.display_quantized_values,
                display_metadata=args.display_metadata,
                target_core=args.target_core,
            )
            if not args.ethos_u:
                _assert_kernel_present(pte, category, base_name)
                _assert_arity_matches_schema(pte, source_dir, category, base_name)

            op_pte.write_bytes(bytes(pte.buffer))

            entry = {
                "op": variant_name,
                "category": category,
                "name": op_name.name,
            }
            if args.ethos_u:
                entry["delegated"] = _is_delegated(pte)
            manifest.append(entry)
            n_variants += 1
        except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
            failed.append((category, variant_name, f"{type(exc).__name__}: {exc}"))
            if not args.continue_on_error:
                traceback.print_exc()

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = len(keys)
    print(
        f"\n=== coverage: {len(manifest)} exported ({n_variants} variants), "
        f"{len(skipped)} skipped, {len(failed)} failed of {total} components ==="
    )
    if args.ethos_u:
        delegated = [m for m in manifest if m.get("delegated")]
        fallback = [m for m in manifest if not m.get("delegated")]
        print(
            f"    Ethos-U: {len(delegated)} delegated to NPU, "
            f"{len(fallback)} host-core fallback"
        )
        if fallback:
            print("    host-core fallback (Vela did not partition):")
            for m in fallback:
                print(f"        [{m['category']}] {m['op']}")
    if skipped:
        print("\nskipped (build/link covered, not executed):")
        for cat, name, reason in skipped:
            print(f"    [{cat}] {name}: {reason}")
    if failed:
        print("\nFAILED to export:")
        for cat, name, reason in failed:
            print(f"    [{cat}] {name}: {reason}")

    # Always fail on export errors, even with --continue-on-error: a missing
    # export would otherwise silently shrink the embedded test set and the
    # firmware would report a full pass over fewer ops.
    if failed:
        raise SystemExit(f"{len(failed)} operator(s) failed to export")


if __name__ == "__main__":
    main()
