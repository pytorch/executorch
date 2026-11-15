#!/usr/bin/env bash
# shellcheck disable=SC2034
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

TORCHCI_REPORT_ARGS=()
if [[ -n "${RUNNER_TEST_RUN_REPORTS_DIR:-}" ]]; then
  # Reporting is best-effort, as in PyTorch: a failed preflight warns and
  # leaves the tests running without the plugin.
  # The interpreter may include a launcher such as conda run.
  # shellcheck disable=SC2086
  if ${TORCHCI_PYTHON:-${PYTHON_EXECUTABLE:-python}} -c '
import os
import pytest
from torch.testing._internal.torchci import environment, plugin, report

if report.SCHEMA_VERSION != "1.0":
    raise RuntimeError("torchci requires report schema 1.0")
if tuple(map(int, pytest.__version__.split(".")[:2])) < (7, 3):
    raise RuntimeError("torchci requires pytest >= 7.3")
job_id = os.environ.get("JOB_ID", "")
if not job_id.isdigit() or int(job_id) <= 0:
    raise RuntimeError("torchci requires a positive JOB_ID")
if not os.environ.get("GITHUB_REPOSITORY"):
    raise RuntimeError("torchci requires GITHUB_REPOSITORY")
environment.capture()
'; then
    TORCHCI_REPORT_ARGS=(
      -p torch.testing._internal.torchci.plugin
      "--torchci-report-prefix=${RUNNER_TEST_RUN_REPORTS_DIR}/${1:-pytest}"
    )
  else
    echo "::warning::torchci: test-run reports disabled because the preflight failed; tests run without them"
  fi
fi
