# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the shared pytest suites from a Buck-packaged source tree."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest
import torch
from executorch.backends.test.suite.flow import all_flows
from executorch.backends.test.suite.reporting import begin_test_session
from executorch.extension.pybindings.portable_lib import _get_registered_backend_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("operators", "models"))
    parser.add_argument("--flow", default="cpu_fp32")
    parser.add_argument("--report", type=Path, default=Path("backend_test_report.json"))
    parser.add_argument("--seed", type=int, default=0)
    args, pytest_args = parser.parse_known_args()
    flows = all_flows()
    if args.flow not in flows:
        parser.error(f"Unknown flow {args.flow!r}; available: {', '.join(flows)}")
    backend_name = {"cpu": "CpuBackend", "xnnpack": "XnnpackBackend"}.get(
        flows[args.flow].backend
    )
    if backend_name and backend_name not in _get_registered_backend_names():
        parser.error(f"Runtime backend {backend_name} is not linked")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    begin_test_session(None, seed=args.seed)
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    return pytest.main(
        [
            "-c",
            "/dev/null",
            "--rootdir",
            str(Path(__file__).parent),
            "-p",
            "no:cacheprovider",
            "-p",
            "pytest_jsonreport.plugin",
            "-p",
            "pytest_timeout",
            str(Path(__file__).parent / args.suite),
            "-m",
            f"flow_{args.flow}",
            "--json-report",
            f"--json-report-file={args.report.resolve()}",
            *pytest_args,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
