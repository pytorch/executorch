#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -euo pipefail

evals_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export EXECUTORCH_EVAL_CACHE="${EXECUTORCH_EVAL_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/executorch-evals}"

harness_dir="${evals_dir}/terminal_bench"
ci=false
if [[ "${1:-}" == --ci ]]; then ci=true; shift; fi
if [[ $# != 0 ]]; then echo "Unknown setup argument: $1" >&2; exit 2; fi

if ! docker info >/dev/null 2>&1; then
  if "${ci}"; then
    echo "CI requires a running Docker daemon accessible by this user." >&2
    exit 2
  fi
  case "$(uname -s)" in
    Darwin)
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
      ;;
    Linux)
      # shellcheck disable=SC1091
      source /etc/os-release
      if [[ "${ID}" != ubuntu ]]; then
        echo "Automatic Docker installation supports Ubuntu. Install and start Docker for ${ID}, then rerun setup." >&2
        exit 2
      fi
      privilege=()
      if [[ ${EUID} != 0 ]]; then privilege=(sudo); fi
      "${privilege[@]}" apt-get update
      "${privilege[@]}" apt-get install -y ca-certificates curl
      "${privilege[@]}" install -m 0755 -d /etc/apt/keyrings
      "${privilege[@]}" curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
      "${privilege[@]}" chmod a+r /etc/apt/keyrings/docker.asc
      docker_repo="deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable"
      echo "${docker_repo}" | "${privilege[@]}" tee /etc/apt/sources.list.d/docker.list >/dev/null
      "${privilege[@]}" apt-get update
      "${privilege[@]}" apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
      "${privilege[@]}" systemctl enable --now docker
      if [[ ${EUID} != 0 ]]; then
        "${privilege[@]}" usermod -aG docker "$(id -un)"
        # Continue in the new group without requiring a logout/login cycle.
        printf -v retry 'bash %q' "${harness_dir}/setup.sh"
        exec sg docker -c "${retry}"
      fi
      ;;
    *) echo "Install Docker on this platform, then rerun setup." >&2; exit 2 ;;
  esac
fi
docker info >/dev/null
if ! docker compose version >/dev/null 2>&1 && ! "${ci}" && [[ "$(uname -s)" == Darwin ]]; then
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
echo "Ready. Run bash ${harness_dir}/run.sh --config /path/to/model.toml"
