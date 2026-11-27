#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -euo pipefail

evals_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo_dir=$(cd -- "${evals_dir}/../../.." && pwd)
export EXECUTORCH_EVAL_CACHE="${EXECUTORCH_EVAL_CACHE:-${HOME}/.cache/executorch-evals}"

eval_python="${EXECUTORCH_EVAL_CACHE}/terminal-bench/venv/bin/python"
if [[ ! -x "${eval_python}" ]]; then
  echo "Run bash ${evals_dir}/terminal_bench/setup.sh first." >&2
  exit 2
fi
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${eval_python}" -m "executorch.examples.llm_server.evals.terminal_bench.runner" "$@"
