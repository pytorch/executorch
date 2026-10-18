# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Benchmark the ESP-NN backend on a connected Espressif board.

For each model in the zoo this script quantizes and lowers it with
aot_espnn_compiler.py, packs the results into bundles that fit the detected
chip's flash, builds and flashes examples/espressif/espnn_runner, and records
correctness (bit-exact or SQNR against the host) and latency (min/mean/max over
timed iterations). Results go to <output>/<target>.json and <target>.md.

Run inside an activated ESP-IDF shell with a Python that has executorch:
    python examples/espressif/scripts/benchmark.py --port /dev/ttyACM0 \
        --images ~/data/imagenette2-160 --dashboard

Models that need more RAM than the board has are reported as "did not fit".
ExecuTorch-mode models need ExecuTorch cross-compiled for the target
(build_executorch_esp.sh) and enough flash for the runtime (about 6 MB with
all portable kernels); pass --skip-executorch on 4 MB boards.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

import espnn_board as board  # noqa: E402
from espnn_bundle import read_bundle, write_bundle  # noqa: E402

ZOO = f"{SCRIPTS_DIR / 'bench_models.py'}"
IMAGENET = (["--mean", "0.485", "0.456", "0.406"], ["--std", "0.229", "0.224", "0.225"])
FIRMWARE_ESTIMATE = {"standalone": 600 * 1024, "executorch": 7 * 1024 * 1024}


@dataclass
class BenchModel:
    name: str
    model: str
    shapes: list[str]
    mode: str
    images: bool
    description: str
    extra: tuple[str, ...] = ()


MODELS = [
    BenchModel(
        "ds_cnn_kws",
        f"{ZOO}:ds_cnn_kws",
        ["1,1,49,10"],
        "standalone",
        False,
        "Keyword spotting DS-CNN, 49x10 MFCC",
    ),
    BenchModel(
        "tiny_resnet_64",
        f"{ZOO}:tiny_resnet",
        ["1,3,64,64"],
        "standalone",
        True,
        "Small residual CNN, 64x64",
    ),
    BenchModel(
        "mobilenet_v3_small_96",
        "mobilenet_v3_small",
        ["1,3,96,96"],
        "standalone",
        True,
        "MobileNetV3-Small, ImageNet, 96x96",
        ("--resize", "110"),
    ),
    BenchModel(
        "mobilenet_v3_small_224",
        "mobilenet_v3_small",
        ["1,3,224,224"],
        "standalone",
        True,
        "MobileNetV3-Small, ImageNet, 224x224",
    ),
    BenchModel(
        "mobilenet_v2_224",
        "mobilenet_v2",
        ["1,3,224,224"],
        "standalone",
        True,
        "MobileNetV2, ImageNet, 224x224",
    ),
    BenchModel(
        "mobilenet_v3_large_224",
        "mobilenet_v3_large",
        ["1,3,224,224"],
        "standalone",
        True,
        "MobileNetV3-Large, ImageNet, 224x224",
    ),
    BenchModel(
        "efficientnet_b0_224",
        "efficientnet_b0",
        ["1,3,224,224"],
        "standalone",
        True,
        "EfficientNet-B0, ImageNet, 224x224",
    ),
    BenchModel(
        "lraspp_mobilenet_v3_seg",
        f"{ZOO}:lraspp_mobilenet_v3",
        ["1,3,224,224"],
        "executorch",
        True,
        "LR-ASPP MobileNetV3 segmentation, 21 classes, 224x224",
        ("--crc-outputs",),
    ),
    BenchModel(
        "ssdlite320_mobilenet_v3_det",
        f"{ZOO}:ssdlite320_mobilenet_v3",
        ["1,3,320,320"],
        "executorch",
        True,
        "SSDLite320 MobileNetV3 detection heads, COCO, 320x320",
        ("--crc-outputs", "--mean", "0.5", "0.5", "0.5", "--std", "0.5", "0.5", "0.5"),
    ),
    BenchModel(
        "tiny_transformer_fallback",
        f"{ZOO}:tiny_transformer",
        ["1,3,64,64"],
        "executorch",
        False,
        "Small vision transformer: LayerNorm and attention on the CPU fallback",
    ),
]


def compile_model(spec: BenchModel, out_dir: Path, args) -> Path:
    suffix = ".xtp" if spec.mode == "executorch" else ".xtb"
    bundle = out_dir / f"{spec.name}{suffix}"
    if (
        bundle.exists()
        and bundle.with_suffix(".json").exists()
        and not args.rebuild_models
    ):
        return bundle
    command = [
        sys.executable,
        str(EXAMPLES_DIR / "aot_espnn_compiler.py"),
        "--model",
        spec.model,
        "--mode",
        spec.mode,
        "--output",
        str(bundle),
        "--name",
        spec.name,
        "--calib-samples",
        str(args.calib_samples),
        "--eval-samples",
        str(args.eval_samples),
        "--vector-samples",
        "2",
    ]
    for shape in spec.shapes:
        command += ["--input-shape", shape]
    if spec.images and args.images:
        command += [
            "--calib-dir",
            str(Path(args.images) / "train"),
            "--eval-dir",
            str(Path(args.images) / "val"),
        ]
    if not any(flag in spec.extra for flag in ("--mean",)) and spec.images:
        command += IMAGENET[0] + IMAGENET[1]
    command += list(spec.extra)
    print(f"[compile] {spec.name}", flush=True)
    log = out_dir / f"{spec.name}.compile.log"
    result = subprocess.run(command, capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"compiling {spec.name} failed; see {log}")
    return bundle


def group_bundles(entries, budget: int):
    groups, current, size = [], [], 0
    for entry in sorted(entries, key=lambda e: len(e[1]) + len(e[2])):
        entry_size = len(entry[1]) + len(entry[2]) + 256
        if entry_size > budget:
            yield None, entry
            continue
        if current and size + entry_size > budget:
            groups.append(current)
            current, size = [], 0
        current.append(entry)
        size += entry_size
    if current:
        groups.append(current)
    for group in groups:
        yield group, None


def markdown(target: str, chip: dict, rows: list[dict], calib_samples: int) -> str:
    lines = [
        f"# ESP-NN backend benchmarks: {target}",
        "",
        f"Board: {chip.get('target', target)}, {chip.get('cores', '?')} cores at "
        f"{chip.get('cpu_mhz', '?')} MHz, {chip.get('flash_mb', '?')} MB flash, "
        f"{chip.get('psram_kb', '?')} KB PSRAM, ESP-IDF {chip.get('idf', '?')}.",
        "",
        "| Model | Input | Path | Operators | Model size | Arena | Latency mean (min) | FPS | On-board check | Host accuracy |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row.get(k, ""))
                for k in (
                    "description",
                    "input",
                    "path",
                    "operators",
                    "size",
                    "arena",
                    "latency",
                    "fps",
                    "check",
                    "accuracy",
                )
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Latency is the model invocation only (no input preprocessing), timed over "
        "repeated runs after a bit-exact (or SQNR) check against the host reference."
    )
    lines.append("")
    lines.append(
        f"Host accuracy is the int8 model on the host (the board's output is bit-identical) "
        f"after post-training quantization with {calib_samples} calibration images and no "
        "QAT, on 32 Imagenette validation images mapped to ImageNet classes. Models with "
        "random weights report top-1 agreement or SQNR against the float model instead."
    )
    return "\n".join(lines) + "\n"


def summarize(
    spec: BenchModel,
    report: dict,
    case: dict | None,
    failure: str | None,
    board_chip: dict,
) -> dict:
    row = {
        "name": spec.name,
        "description": spec.description,
        "input": " / ".join(spec.shapes),
    }
    if spec.mode == "standalone":
        row["path"] = "standalone XTS3"
        row["operators"] = f"{sum(report.get('operations', {}).values())} int8"
        row["size"] = f"{report.get('blob_bytes', 0) / 1e6:.2f} MB"
    else:
        row["path"] = "ExecuTorch .pte"
        row["operators"] = (
            f"{report.get('delegated_ops', 0)} delegated, "
            f"{len(report.get('cpu_ops', []))} CPU"
        )
        row["size"] = f"{report.get('pte_bytes', 0) / 1e6:.2f} MB"
    evaluation = report.get("evaluation", {})
    if evaluation.get("labelled") and not spec.model.startswith(ZOO):
        row["accuracy"] = (
            f"top-1 {evaluation['device_correct']}/{evaluation['labelled']} "
            f"(float {evaluation['float_correct']}), agrees with float "
            f"{evaluation['top1_agree_with_float']}/{evaluation['samples']}"
        )
    elif evaluation.get("samples"):
        row["accuracy"] = (
            f"top-1 agrees with float {evaluation['top1_agree_with_float']}/{evaluation['samples']}"
        )
    else:
        row["accuracy"] = f"{report.get('sqnr_vs_float_db', 0):.1f} dB vs float"
    if failure or case is None:
        row.update(latency="-", fps="-", arena="-", check=failure or "not run")
        return row
    mean_us = float(case.get("us", 0))
    row["latency"] = (
        f"{mean_us / 1000:.1f} ms ({float(case.get('min_us', mean_us)) / 1000:.1f})"
    )
    row["fps"] = f"{1e6 / mean_us:.2f}" if mean_us else "-"
    row["arena"] = f"{int(case.get('arena', 0)) / 1024:.0f} KB"
    arena = re.search(r"allocate (\d+)-byte activation arena", case.get("error", ""))
    if case.get("status") != "PASS" and arena:
        free = board_chip.get("internal_free_kb", "?") if board_chip else "?"
        row.update(
            latency="-",
            fps="-",
            arena=f"{int(arena.group(1)) / 1024:.0f} KB",
            check=(
                f"does not fit RAM ({free} KB free, no PSRAM)"
                if board_chip.get("psram_kb") == "0"
                else "does not fit RAM"
            ),
        )
        return row
    if case.get("status") != "PASS":
        row["check"] = f"FAIL ({case.get('reason', 'mismatch')})"
    elif case["kind"] == "standalone":
        row["check"] = f"bit-exact {case.get('exact')}/{case.get('samples')}" + (
            ", ANSI too" if case.get("ansi") == "1" else ""
        )
    elif case.get("exact_required") == "1":
        row["check"] = "bit-exact"
    else:
        row["check"] = f"{float(case.get('sqnr_db', 0)):.1f} dB vs host"
    if case["kind"] == "executorch" and case.get("delegate_us"):
        row[
            "latency"
        ] += f"; delegate {int(case['delegate_us']) / 1000:.1f}, CPU {max(int(case['cpu_us']), 0) / 1000:.1f}"
    return row


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument(
        "--target", help="override the detected chip (esp32s3, esp32, ...)"
    )
    parser.add_argument("--models", help="comma separated subset of model names")
    parser.add_argument(
        "--images", help="Imagenette-style folder with train/ and val/ for calibration"
    )
    parser.add_argument("--calib-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=EXAMPLES_DIR / "benchmarks")
    parser.add_argument(
        "--work-dir", type=Path, default=EXAMPLES_DIR / "benchmarks" / "work"
    )
    parser.add_argument("--rebuild-models", action="store_true")
    parser.add_argument("--skip-executorch", action="store_true")
    parser.add_argument("--executorch-build-dir", type=Path)
    parser.add_argument("--executorch-source-dir", type=Path)
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="live results at http://127.0.0.1:8770/",
    )
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    chip = board.detect_chip(args.port)
    target = args.target or chip.target
    print(
        f"[board] {chip.description} -> target {target}, {chip.flash_mb} MB flash",
        flush=True,
    )
    selected = [
        m for m in MODELS if not args.models or m.name in args.models.split(",")
    ]
    if args.skip_executorch:
        selected = [m for m in selected if m.mode != "executorch"]

    dashboard = None
    if args.dashboard:
        import board_dashboard

        dashboard = board_dashboard.start(open_browser=True)

    model_dir = args.work_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    bundles = {"standalone": [], "executorch": []}
    for spec in selected:
        try:
            bundle = compile_model(spec, model_dir, args)
        except RuntimeError as error:
            print(f"[compile] {error}", flush=True)
            reports[spec.name] = {"error": str(error)}
            continue
        reports[spec.name] = json.loads(bundle.with_suffix(".json").read_text())[0]
        _, entries = read_bundle(bundle)
        bundles[spec.mode].extend(entries)

    cases: dict[str, dict] = {}
    failures: dict[str, str] = {}
    board_chip: dict = {}
    for mode, entries in bundles.items():
        if not entries:
            continue
        budget = chip.app_partition_bytes - FIRMWARE_ESTIMATE[mode]
        build = board.RunnerBuild(
            target=target,
            executorch=mode == "executorch",
            build_dir=args.work_dir / f"build-{target}-{mode}",
            executorch_build_dir=args.executorch_build_dir,
            executorch_source_dir=args.executorch_source_dir,
        )
        index = -1
        for group, too_big in group_bundles(entries, budget):
            if too_big is not None:
                failures[too_big[0]] = f"does not fit {chip.flash_mb} MB flash"
                continue
            index += 1
            magic = b"XTP1" if mode == "executorch" else b"XTB1"
            path = (
                args.work_dir
                / f"{target}-{mode}-{index}{'.xtp' if mode == 'executorch' else '.xtb'}"
            )
            write_bundle(path, magic, group)
            names = [name for name, _, _ in group]
            print(f"[board] {mode} group {index}: {', '.join(names)}", flush=True)
            job = f"{mode}-{index}"
            if dashboard:
                dashboard.begin_job(
                    job,
                    f"{target} {mode} group {index}",
                    {name: {"host": reports.get(name, {})} for name in names},
                )
            try:
                result = board.run_bundle(
                    build,
                    path,
                    args.port,
                    args.iterations,
                    args.timeout,
                    args.work_dir / "logs" / job,
                    on_line=(lambda line: dashboard.log(line)) if dashboard else None,
                    on_case=(
                        (lambda case, job=job: dashboard.case(job, case))
                        if dashboard
                        else None
                    ),
                    on_status=(
                        (lambda status, job=job: dashboard.status(job, status))
                        if dashboard
                        else None
                    ),
                )
            except RuntimeError as error:
                for name in names:
                    failures[name] = str(error).splitlines()[0]
                if dashboard:
                    dashboard.status(job, "build failed")
                continue
            board_chip = result.chip or board_chip
            for case in result.cases:
                cases[case.get("name", "?")] = case
            for name in names:
                if name not in cases:
                    failures[name] = (
                        "did not run (out of memory or crash; see board.log)"
                    )
            if dashboard:
                dashboard.status(job, "PASS" if result.passed else "FAIL")

    rows = []
    for spec in selected:
        report = reports.get(spec.name, {})
        if "error" in report:
            rows.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input": " / ".join(spec.shapes),
                    "check": "compile failed",
                }
            )
            continue
        rows.append(
            summarize(
                spec, report, cases.get(spec.name), failures.get(spec.name), board_chip
            )
        )
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": target,
        "chip": board_chip or {"target": target},
        "esptool": chip.description,
        "date": time.strftime("%Y-%m-%d"),
        "iterations": args.iterations,
        "rows": rows,
        "raw_cases": cases,
        "failures": failures,
    }
    (args.output / f"{target}.json").write_text(json.dumps(payload, indent=2))
    (args.output / f"{target}.md").write_text(
        markdown(target, payload["chip"], rows, args.calib_samples)
    )
    print((args.output / f"{target}.md").read_text())
    if dashboard:
        dashboard.finish()
        print("dashboard still serving at http://127.0.0.1:8770/ (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0 if all(not failures.get(spec.name) for spec in selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
