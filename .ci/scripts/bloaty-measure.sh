#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: bash .ci/scripts/bloaty-measure.sh <job_name> <head_elf> <strip_tool>
#
# Runs bloaty against the head ELF, writes metadata.json + full.txt +
# head_only.txt to artifacts-to-be-uploaded/, and appends a markdown table
# to $GITHUB_STEP_SUMMARY.
#
# Best-effort: never exits non-zero — the size jobs that source this should
# not fail because of a bloaty hiccup.

set -uo pipefail

job_name=$1
head_elf=$2
strip_tool=$3
head_sha=${GITHUB_HEAD_SHA:-${GITHUB_SHA:-unknown}}

(
  # conda-forge bloaty depends on a newer libstdc++ than the ubuntu-22.04
  # docker images ship, so pull libstdcxx-ng into the same env and invoke
  # via `conda run` so library paths are set correctly.
  bloaty_env=/tmp/bloaty-conda-env
  if [[ ! -x "${bloaty_env}/bin/bloaty" ]]; then
    conda create -y -p "${bloaty_env}" -c conda-forge bloaty libstdcxx-ng || exit 1
  fi
  bloaty_cmd=("conda" "run" "--no-capture-output" "-p" "${bloaty_env}" "bloaty")
  "${bloaty_cmd[@]}" --version || exit 1

  tmp_out=/tmp/bloaty-out
  rm -rf "${tmp_out}" && mkdir -p "${tmp_out}"
  BLOATY="${bloaty_cmd[*]}" python3 .github/scripts/bloaty_diff.py measure \
    --head "${head_elf}" \
    --job "${job_name}" \
    --binary-name size_test \
    --head-sha "${head_sha}" \
    --strip-tool "${strip_tool}" \
    --out "${tmp_out}" || exit 1
  mkdir -p artifacts-to-be-uploaded
  mv "${tmp_out}"/* artifacts-to-be-uploaded/
) || echo "bloaty report failed; continuing"
