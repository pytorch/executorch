#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pin pytest-xdist's worker count, but only when the container is allowed less
# CPU than the machine it landed on. `auto` asks psutil for the machine's
# physical cores, which inside an OSDC pod is the whole node, so the workers
# exhaust the pod's memory. nproc honours the pod's cpuset, which is what
# pytorch relies on for OMP_NUM_THREADS on the same fleet.
#
# Left alone when unconstrained. `auto` already discounts hyperthreads there,
# while nproc counts them, so overriding it would double the workers and halve
# the memory each one gets.

if [[ -z "${PYTEST_XDIST_AUTO_NUM_WORKERS:-}" ]] && command -v nproc >/dev/null 2>&1; then
  cpus_allowed="$(nproc)"
  cpus_installed="$(nproc --all)"
  echo "pytest-xdist: ${cpus_allowed} of ${cpus_installed} CPUs available"
  if [[ "${cpus_allowed}" -lt "${cpus_installed}" ]]; then
    export PYTEST_XDIST_AUTO_NUM_WORKERS="${cpus_allowed}"
    echo "PYTEST_XDIST_AUTO_NUM_WORKERS=${PYTEST_XDIST_AUTO_NUM_WORKERS}"
  fi
fi
