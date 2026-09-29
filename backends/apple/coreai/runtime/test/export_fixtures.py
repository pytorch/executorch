# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export source or AOT-compiled fixtures and reference vectors for OS 27 inference.

Run with the Core AI conda environment and --output-dir pointing to a fresh
folder. Each fixture contains model.pte, vectors.json, and delegate.json.
Sidecar fixtures also contain the hash-keyed asset beside model.pte; pass that
fixture directory as the runtime's absolute base_dir option.
Use --aot-platform to compile all fixtures, with optional repeated --aot-architecture
flags. Compilation does not execute the SDK; inference requires OS 27.
"""

import argparse
import contextlib
import hashlib
import json
import operator
import struct
from pathlib import Path

import torch
from executorch.backends.apple.coreai import (
    get_default_compile_config,
    get_default_passes,
)
from executorch.backends.apple.coreai.compiler.preprocess import (
    AOTCompileConfig,
    coreai_sidecar_dir,
)
from executorch.backends.apple.coreai.partition.partitioner import CoreAIPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import (
    executorch_call_delegate,
    get_lowered_backend_modules,
)


class OrderedArithmetic(torch.nn.Module):
    def forward(self, z, a):
        difference = z - a
        scaled = z + (a + a)
        return scaled, difference


def _inputs(rows, dtype, case=0):
    values = torch.arange(rows * 4, dtype=dtype).reshape(rows, 4)
    return values + 7 + case, 2 - values - 2 * case


def _tensor_record(tensor):
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "values": tensor.flatten().tolist(),
    }


def _validate_asset_metadata(manifest, directory, store):
    """Independently check fixture metadata against the delivered bytes."""
    prefix = f"coreai/{manifest['hash']}/"
    if store is None:
        root = directory / manifest["hash"]
        payloads = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
    else:
        if any(not key.startswith(prefix) for key in store.pte_data):
            raise RuntimeError("fixture has unexpected named-data keys")
        payloads = {
            key[len(prefix) :]: store.buffers[entry.buffer_index]
            for key, entry in store.pte_data.items()
        }
    sizes = {name: len(payload) for name, payload in payloads.items()}
    bundles = (
        {Path(path).name for path in manifest["archs"].values()}
        if "archs" in manifest
        else {"model.aimodel"}
    )
    digests = {}
    for bundle in bundles:
        names = sorted(
            (name.encode("utf-8") for name in payloads if name.startswith(bundle + "/"))
        )
        if not names:
            raise RuntimeError(f"fixture bundle contains no files: {bundle}")
        digest = hashlib.sha256(b"ExecuTorch.CoreAI.raw-assets.sha256.v1\0")
        digest.update(struct.pack(">Q", len(names)))
        for name in names:
            payload = payloads[name.decode("utf-8")]
            digest.update(struct.pack(">Q", len(name)))
            digest.update(name)
            digest.update(struct.pack(">Q", len(payload)))
            digest.update(payload)
        digests[bundle] = digest.hexdigest()
    if (
        manifest.get("version") != 2
        or manifest.get("files") != sizes
        or not all(
            type(size) is int and size >= 0 for size in manifest["files"].values()
        )
        or {name.split("/", 1)[0] for name in payloads} != bundles
        or manifest.get("bundle_digests") != digests
    ):
        raise RuntimeError("fixture asset metadata does not match delivered bytes")


def export_fixtures(
    output_dir: Path,
    *,
    include_dynamic: bool = True,
    aot_compile_config: AOTCompileConfig | None = None,
) -> list[dict]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"fixture output directory must be empty: {output_dir}")
    records = []
    for dtype_name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        for dynamic in (False, True) if include_dynamic else (False,):
            for packaging in ("inline", "sidecar"):
                name = f"{dtype_name}_{'dynamic' if dynamic else 'static'}_{packaging}"
                directory = output_dir / name
                directory.mkdir()
                model = OrderedArithmetic().eval()
                dynamic_shapes = None
                if dynamic:
                    rows = torch.export.Dim("rows", min=1, max=4)
                    dynamic_shapes = ({0: rows}, {0: rows})
                ep = torch.export.export(
                    model, _inputs(2, dtype), dynamic_shapes=dynamic_shapes
                )
                uses_sidecar = packaging == "sidecar"
                delivery = (
                    coreai_sidecar_dir(str(directory))
                    if uses_sidecar
                    else contextlib.nullcontext()
                )
                with delivery:
                    lowered = to_edge_transform_and_lower(
                        ep,
                        partitioner=[
                            CoreAIPartitioner(
                                uses_sidecar=uses_sidecar,
                                min_deployment_version="27.0",
                                aot_compile_config=aot_compile_config,
                            )
                        ],
                        transform_passes=get_default_passes(),
                        compile_config=get_default_compile_config(),
                    )
                graph = lowered.exported_program().graph_module
                delegates = get_lowered_backend_modules(graph)
                if len(delegates) != 1:
                    raise RuntimeError(f"{name}: expected exactly one delegate")
                calls = [
                    n
                    for n in graph.graph.nodes
                    if n.op == "call_function" and n.target is executorch_call_delegate
                ]
                placeholders = [n for n in graph.graph.nodes if n.op == "placeholder"]
                if len(calls) != 1 or list(calls[0].args[1:]) != placeholders:
                    raise RuntimeError(
                        f"{name}: delegate inputs do not match model boundary order"
                    )
                model_outputs = graph.graph.output_node().args[0]
                if any(
                    n.op != "call_function"
                    or n.target is not operator.getitem
                    or n.args[0] is not calls[0]
                    for n in model_outputs
                ):
                    raise RuntimeError(
                        f"{name}: outputs must come directly from the delegate"
                    )
                output_indices = [n.args[1] for n in model_outputs]
                if sorted(output_indices) != [0, 1]:
                    raise RuntimeError(f"{name}: expected both delegate outputs")
                manifest = json.loads(bytes(delegates[0].processed_bytes))
                expected_bindings = {
                    "version": 2,
                    "function": "main",
                    "input_names": ["input_0", "input_1"],
                    "output_names": ["output_0", "output_1"],
                }
                if any(manifest.get(k) != v for k, v in expected_bindings.items()):
                    raise RuntimeError(
                        f"{name}: unexpected delegate bindings: {manifest}"
                    )
                _validate_asset_metadata(
                    manifest, directory, delegates[0].named_data_store_output
                )
                (directory / "model.pte").write_bytes(
                    bytes(lowered.to_executorch().buffer)
                )
                (directory / "delegate.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                cases = []
                # Revisit smaller shapes after capacity-sized outputs on one session.
                for case, rows in enumerate([2, 4, 1, 3, 2] if dynamic else [2, 2]):
                    inputs = _inputs(rows, dtype, case)
                    outputs = model(*inputs)
                    cases.append(
                        {
                            "inputs": [_tensor_record(x) for x in inputs],
                            "outputs": [_tensor_record(x) for x in outputs],
                        }
                    )
                vectors = {
                    "method": "forward",
                    "input_names": ["z", "a"],
                    "output_expressions": ["z + 2 * a", "z - a"],
                    "model_output_delegate_indices": output_indices,
                    "dynamic_rows": {"min": 1, "max": 4} if dynamic else None,
                    "cases": cases,
                }
                (directory / "vectors.json").write_text(
                    json.dumps(vectors, indent=2) + "\n"
                )
                records.append(
                    {
                        "name": name,
                        "pte": f"{name}/model.pte",
                        "vectors": f"{name}/vectors.json",
                        "packaging": manifest["packaging"],
                        "base_dir": name if uses_sidecar else None,
                    }
                )
    (output_dir / "fixtures.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--static-only", action="store_true", help="Omit bounded dynamic fixtures"
    )
    parser.add_argument(
        "--aot-platform",
        choices=("iOS", "macOS", "watchOS", "visionOS", "tvOS"),
        help="Export AOT-compiled .aimodelc fixtures for this platform",
    )
    parser.add_argument(
        "--aot-architecture",
        action="append",
        help="Target SDK architecture (repeatable; default: all supported)",
    )
    args = parser.parse_args()
    if args.aot_architecture and not args.aot_platform:
        parser.error("--aot-architecture requires --aot-platform")
    config = (
        AOTCompileConfig(
            platform=args.aot_platform, architectures=args.aot_architecture
        )
        if args.aot_platform
        else None
    )
    records = export_fixtures(
        args.output_dir,
        include_dynamic=not args.static_only,
        aot_compile_config=config,
    )
    print(f"Exported {len(records)} fixtures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
