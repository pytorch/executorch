#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
  echo "Terminal-Bench evaluations currently support macOS only." >&2
  exit 2
fi
if [[ $# != 0 ]]; then echo "Unknown setup argument: $1" >&2; exit 2; fi

evals_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export EXECUTORCH_EVAL_CACHE="${EXECUTORCH_EVAL_CACHE:-${HOME}/.cache/executorch-evals}"

harness_dir="${evals_dir}/terminal_bench"
if ! docker info >/dev/null 2>&1; then
  if ! command -v brew >/dev/null; then
    echo "Install Homebrew (https://brew.sh), or start an existing Docker Desktop installation, then rerun setup." >&2
    exit 2
  fi
  brew install colima docker docker-compose
  if [[ "$(uname -m)" == arm64 ]]; then
    colima start --cpu 4 --memory 8 --vm-type vz --vz-rosetta
  else
    colima start --cpu 4 --memory 8
  fi
fi
docker info >/dev/null
if ! docker compose version >/dev/null 2>&1; then
  brew install docker-compose
  plugin_dir="${DOCKER_CONFIG:-${HOME}/.docker}/cli-plugins"
  mkdir -p "${plugin_dir}"
  ln -s "$(brew --prefix)/bin/docker-compose" "${plugin_dir}/docker-compose"
fi
docker compose version
command -v git >/dev/null

export PATH="${EXECUTORCH_EVAL_CACHE}/bin:${PATH}"
if ! command -v uv >/dev/null; then
  # Install uv into the evaluation cache without changing shell startup files.
  installer=$(mktemp)
  trap 'rm -f "${installer}"' EXIT
  curl -LsSf https://astral.sh/uv/0.8.22/install.sh -o "${installer}"
  UV_UNMANAGED_INSTALL="${EXECUTORCH_EVAL_CACHE}/bin" sh "${installer}"
  export PATH="${EXECUTORCH_EVAL_CACHE}/bin:${PATH}"
fi
eval_env="${EXECUTORCH_EVAL_CACHE}/terminal-bench/venv"
if [[ ! -x "${eval_env}/bin/python" ]]; then
  uv venv --python 3.12 "${eval_env}"
fi
uv pip install --python "${eval_env}/bin/python" -r "${harness_dir}/requirements.txt"
server_env="${EXECUTORCH_EVAL_CACHE}/server-venv"
if [[ ! -x "${server_env}/bin/python" ]]; then
  uv venv --python 3.12 "${server_env}"
fi
uv pip install --python "${server_env}/bin/python" -r "${evals_dir}/../python/requirements.txt"
echo "Server Python: ${server_env}/bin/python"
echo "Dependencies ready. Configure your model, worker, and tokenizer paths in a local TOML."
echo "Run bash ${harness_dir}/run.sh --config /path/to/model.toml"
