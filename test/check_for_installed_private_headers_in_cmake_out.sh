#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script verifies that all headers that are installed under
# cmake-out/include/executorch are exported_headers of some Buck
# target. (It does *not* verify the reverse, namely that all
# exported_headers of every Buck target that should have been built
# when that directory was installed are actually installed.)
#
# Ideally, "some Buck target" would include any target in the whole
# repo, but we cannot yet buck query the whole repo. (See
# .ci/scripts/unittest-buck2.sh.) Instead, we query a manually-curated
# list of targets.

set -euxo pipefail

BUCK_HEADERS_TEMPFILE=$(mktemp /tmp/check_private_headers_buck.txt.XXXXXX)
ACTUAL_HEADERS_TEMPFILE=$(mktemp /tmp/check_private_headers_installed.txt.XXXXXX)
SOURCE_ROOT_DIR=$(git rev-parse --show-toplevel)
BUCK2=$(python3 "${SOURCE_ROOT_DIR}/tools/cmake/resolve_buck.py" --cache_dir="${SOURCE_ROOT_DIR}/buck2-bin")
if [[ "$BUCK2" == "buck2" ]]; then
  BUCK2=$(command -v buck2)
fi

"${SOURCE_ROOT_DIR}/scripts/print_exported_headers.py" \
    --buck2=$(realpath "$BUCK2") --targets \
    //extension/data_loader: //extension/evalue_util: \
    //extension/flat_tensor: //extension/llm/runner: //extension/kernel_util: //extension/module: \
    //extension/runner_util: //extension/tensor: //extension/threadpool: \
    | sort > "${BUCK_HEADERS_TEMPFILE}"
find "${SOURCE_ROOT_DIR}/cmake-out/include/executorch" -name '*.h' | \
    sed -e "s|${SOURCE_ROOT_DIR}/cmake-out/include/executorch/||" | \
    # Don't complain about generated Functions.h \
    grep -E -v 'Functions.h$' |  sort > "${ACTUAL_HEADERS_TEMPFILE}"
ACTUAL_HEADERS_NOT_EXPORTED_IN_BUCK=$(comm -13 "${BUCK_HEADERS_TEMPFILE}" "${ACTUAL_HEADERS_TEMPFILE}")
if [[ -n "${ACTUAL_HEADERS_NOT_EXPORTED_IN_BUCK}" ]]; then
    >&2 echo "The following non-exported headers were installed:
${ACTUAL_HEADERS_NOT_EXPORTED_IN_BUCK}"
    exit 1
fi
