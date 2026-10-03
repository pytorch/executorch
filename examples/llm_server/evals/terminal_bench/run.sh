#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -euo pipefail

evals_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo_dir=$(cd -- "${evals_dir}/../../.." && pwd)
harness_name=terminal-bench
harness=terminal_bench
export EXECUTORCH_EVAL_CACHE="${EXECUTORCH_EVAL_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/executorch-evals}"


if [[ "$(uname -s)" == Linux ]] && ! docker info >/dev/null 2>&1 &&
   [[ " $(id -nG) " != *" docker "* ]] &&
   [[ " $(id -nG "$(id -un)") " == *" docker "* ]]; then
  # setup.sh may have just added Docker access while the caller's shell still
  # has its original supplementary groups.
  printf -v retry '%q ' bash "${evals_dir}/run.sh" "${harness_name}" "$@"
  exec sg docker -c "${retry}"
fi

eval_python="${EXECUTORCH_EVAL_CACHE}/terminal-bench/venv/bin/python"
if [[ ! -x "${eval_python}" ]]; then
  echo "Run bash ${evals_dir}/setup.sh terminal-bench first." >&2
  exit 2
fi
if [[ -z "${EXECUTORCH_EVAL_SERVER_PYTHON:-}" ]]; then
  EXECUTORCH_EVAL_SERVER_PYTHON=$(command -v python || command -v python3)
  export EXECUTORCH_EVAL_SERVER_PYTHON
fi
if [[ -z "${EXECUTORCH_EVAL_AGENT_HOST:-}" ]] && [[ "$(uname -s)" == Darwin ]]; then
  if [[ "$(docker context show 2>/dev/null || true)" == colima* ]]; then
    export EXECUTORCH_EVAL_AGENT_HOST=host.lima.internal
  fi
fi
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${eval_python}" -m "executorch.examples.llm_server.evals.${harness}" "$@"
