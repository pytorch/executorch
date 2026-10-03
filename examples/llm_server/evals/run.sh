#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -euo pipefail

evals_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "${1:---help}" in
  terminal-bench) harness=terminal_bench ;;
  --help|-h)
    echo "Usage: bash examples/llm_server/evals/run.sh terminal-bench [driver options]"
    exit 0 ;;
  *) echo "Unsupported evaluation: $1" >&2; exit 2 ;;
esac
shift
exec bash "${evals_dir}/${harness}/run.sh" "$@"
