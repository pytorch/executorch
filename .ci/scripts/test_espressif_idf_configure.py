#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configure the ESP-IDF component's two-model consumer on this host.

For the nightly macOS and Windows jobs. Installs ESP-IDF and its ESP32-S3 tools
under --work-dir/<--idf-version> with ESP-IDF's own Python environment, then runs
`idf.py reconfigure` on idf_component_smoke with the smoke and mul models and
the component from test_espressif_idf_nightly.sh prepare: once with the kernel
bundle prepare made, and once in automatic mode, where this Python (which must
import executorch) makes the bundle. Both bundles must select the same kernels.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ET_ROOT = Path(__file__).resolve().parents[2]


def run(cmd, env):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], env=env, check=True)


def idf_environment(work, idf_version):
    idf_path = work / "esp-idf"
    env = dict(
        os.environ,
        IDF_PATH=str(idf_path),
        IDF_TOOLS_PATH=str(work / "idf-tools"),
        IDF_PYTHON_ENV_PATH=str(work / "idf_python_env"),
    )
    # windows_job runs under Git Bash, whose MSYSTEM makes idf_tools.py refuse to run.
    env.pop("MSYSTEM", None)
    if not idf_path.exists():
        run(
            ["git", "-c", "core.longpaths=true", "clone", "--depth", "1"]
            + ["--branch", idf_version, "--recurse-submodules", "--shallow-submodules"]
            + ["https://github.com/espressif/esp-idf.git", idf_path],
            env,
        )
    idf_tools = [sys.executable, idf_path / "tools/idf_tools.py"]
    run(
        idf_tools + ["install", "--targets", "esp32s3", "required", "cmake", "ninja"],
        env,
    )
    # install-python-env fails if IDF_PYTHON_ENV_PATH doesn't exist yet.
    Path(env["IDF_PYTHON_ENV_PATH"]).mkdir(exist_ok=True)
    run(idf_tools + ["install-python-env"], env)
    exported = subprocess.run(
        [str(c) for c in idf_tools + ["export", "--format", "key-value"]],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in exported.splitlines():
        key, _, value = line.partition("=")
        if key == "PATH":
            value = value.replace("%PATH%", env["PATH"]).replace("$PATH", env["PATH"])
        if key:
            env[key] = value
    scripts = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return env, Path(env["IDF_PYTHON_ENV_PATH"]) / scripts


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--inputs", type=Path, required=True, help="prepare's directory"
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--idf-version", default="v6.1")
    args = parser.parse_args(argv)
    work = args.work_dir.resolve() / args.idf_version
    work.mkdir(parents=True, exist_ok=True)
    inputs = args.inputs.resolve()
    env, idf_python = idf_environment(work, args.idf_version)
    env["IDF_TARGET"] = "esp32s3"

    models = ";".join(
        (inputs / "models" / f"{m}.pte").as_posix() for m in ("smoke", "mul")
    )
    common = [
        idf_python,
        Path(env["IDF_PATH"]) / "tools/idf.py",
        "-C",
        ET_ROOT / "examples/espressif/idf_component_smoke",
        f"-DET_COMPONENT_DIR={(inputs / 'component/executorch').as_posix()}",
        f"-DET_SMOKE_MODELS={models}",
    ]
    bundled = work / "bundled"
    run(
        common
        + ["-B", bundled, f"-DSDKCONFIG={(bundled / 'sdkconfig').as_posix()}"]
        + [f"-DET_SMOKE_KERNELS={(inputs / 'models/smoke_mul_kernels').as_posix()}"]
        + ["reconfigure"],
        env,
    )
    automatic = work / "automatic"
    run(
        common
        + ["-B", automatic, f"-DSDKCONFIG={(automatic / 'sdkconfig').as_posix()}"]
        + [f"-DEXECUTORCH_PYTHON={Path(sys.executable).as_posix()}", "reconfigure"],
        env,
    )

    def selection(bundle):
        kernels = json.loads((bundle / "kernels.json").read_text())
        return {k: kernels[k] for k in ("models", "libraries", "prim_ops")}

    made = selection(automatic / "executorch_kernels")
    if made != selection(inputs / "models/smoke_mul_kernels"):
        sys.exit(f"FAIL: automatic mode selected {made}, unlike prepare's bundle")
    print(
        "PASS: the two-model consumer configured with a bundle and in automatic mode."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
