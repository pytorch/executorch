#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Runs an exported CUDA program twice with nothing but the prebuilt
# executor_runner and its runtime libraries: empty HOME, TMPDIR, working
# directory and Triton / Inductor / CUDA cache directories, no Python or
# compiler on PATH and no other environment variables. Checks that
# - the runner links no PyTorch, Python or Triton library,
# - neither run starts another program or opens a file of a Python, PyTorch or
#   Triton installation, a compiler, or a cache,
# - neither run leaves a file behind (the runtime removes the shared library
#   it extracts from the .pte), and
# - both runs write the same output, which matches the eager reference.
#
# Usage: test_cuda_clean_runtime.sh RUNNER MODEL_DIR MODEL_NAME [DTYPE]
#   MODEL_DIR holds MODEL_NAME.pte, aoti_cuda_blob.ptd and the
#   MODEL_NAME_input_<i>.bin / MODEL_NAME_expected_0.bin files written by
#   `python -m examples.cuda.scripts.export --save_example_io`. DTYPE is the
#   numpy dtype of the output (default float32).

set -euo pipefail

runner="$(realpath "$1")"
model_dir="$(realpath "$2")"
model="$3"
dtype="${4:-float32}"

fail() {
  echo "::error::$*"
  exit 1
}

strace_bin="$(command -v strace || true)"
if [ -z "${strace_bin}" ]; then
  conda install -y -c conda-forge strace
  strace_bin="$(command -v strace)"
fi

work="$(mktemp -d)"
mkdir -p "${work}"/{home,tmp,cwd,out,bin} "${work}"/cache/{xdg,triton,inductor,cuda}
inputs="$(find "${model_dir}" -maxdepth 1 -name "${model}_input_*.bin" | sort -V | paste -sd, -)"

ldd "${runner}" | tee "${work}/ldd.txt"
if grep -E "libtorch|libc10|libpython|libtriton|not found" "${work}/ldd.txt"; then
  fail "executor_runner must resolve every library and link no PyTorch, Python or Triton library"
fi
# The directories the runner's libraries resolve to, plus the inherited ones
# (e.g. where the container mounts the CUDA driver) that are not part of a
# Python installation.
library_path="$(
  {
    awk '$3 ~ /^\// {print $3}' "${work}/ldd.txt" | xargs -n1 dirname
    tr ':' '\n' <<<"${LD_LIBRARY_PATH:-}" | grep -vE 'site-packages|/python[0-9.]*/' || true
  } | grep -v '^$' | awk '!seen[$0]++' | paste -sd: -
)"
# strace may itself come from a conda environment.
strace_library_path="$(dirname "$(dirname "${strace_bin}")")/lib"

forbidden='site-packages|/python[0-9.]*/|libpython|libtorch|libc10|triton|inductor|nvcc|ptxas|nvrtc|/bin/(cc|c\+\+|gcc|g\+\+|clang|ld)$'

for run in 1 2; do
  trace="${work}/strace.${run}"
  (
    cd "${work}/cwd"
    env -i \
      HOME="${work}/home" \
      TMPDIR="${work}/tmp" \
      XDG_CACHE_HOME="${work}/cache/xdg" \
      TRITON_CACHE_DIR="${work}/cache/triton" \
      TORCHINDUCTOR_CACHE_DIR="${work}/cache/inductor" \
      CUDA_CACHE_PATH="${work}/cache/cuda" \
      CUDA_CACHE_DISABLE=1 \
      PATH="${work}/bin" \
      LD_LIBRARY_PATH="${library_path}:${strace_library_path}" \
      "${strace_bin}" -f -qq -o "${trace}" \
      -e trace=execve,execveat,open,openat,creat,mkdir,mkdirat,rename,renameat,renameat2,link,linkat,symlink,symlinkat \
      "${runner}" \
      --model_path "${model_dir}/${model}.pte" \
      --data_path "${model_dir}/aoti_cuda_blob.ptd" \
      --inputs "${inputs}" \
      --output_file "${work}/out/run${run}" \
      --print_output none
  )

  execs="$(grep -E 'execve(at)?\(' "${trace}" | grep -v "\"${runner}\"" || true)"
  [ -z "${execs}" ] || fail "run ${run} started another program: ${execs}"
  opened="$(grep -E '(open|openat|creat)\(' "${trace}" | grep -vE '= -1 ' \
    | grep -oE '"[^"]*"' | tr -d '"' | sort -u)"
  echo "run ${run} opened:"
  echo "${opened}"
  bad="$(grep -E "${forbidden}" <<<"${opened}" || true)"
  [ -z "${bad}" ] || fail "run ${run} opened: ${bad}"
  written="$(grep -E '(open|openat)\(.*O_(WRONLY|RDWR|CREAT)|creat\(|mkdir|rename|link\(' "${trace}" \
    | grep -vE '= -1 ' | grep -oE '"[^"]*"' | tr -d '"' \
    | grep -vE "^(${work}/(tmp|out)/|/dev/|/proc/self/)" || true)"
  [ -z "${written}" ] || fail "run ${run} wrote outside its output and temporary directories: ${written}"
  left="$(find "${work}/home" "${work}/tmp" "${work}/cwd" "${work}"/cache/* -mindepth 1)"
  [ -z "${left}" ] || fail "run ${run} left files behind: ${left}"
done

cmp "${work}/out/run1-0.bin" "${work}/out/run2-0.bin"
python - "${work}/out/run1-0.bin" "${model_dir}/${model}_expected_0.bin" "${dtype}" <<'PY'
import sys

import numpy as np

output, expected = (np.fromfile(path, dtype=sys.argv[3]) for path in sys.argv[1:3])
assert output.shape == expected.shape, (output.shape, expected.shape)
np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
print(f"output matches the eager reference ({output.size} values)")
PY
echo "Clean runtime check passed for ${model}"
