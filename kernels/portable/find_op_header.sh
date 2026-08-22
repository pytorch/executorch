#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -o errexit
set -o nounset
set -o pipefail

# The directory containing this shell script, regardless of the CWD when it
# was invoked.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
readonly SCRIPT_DIR

# The fbsource root that contains this script, even if it was run from xplat.
FBSOURCE_DIR="$(cd -- "${SCRIPT_DIR}" && hg root)"
readonly FBSOURCE_DIR

# The buck target that generates the header file.
readonly GENRULE_TARGET='fbcode//executorch/kernels/portable:generated_lib_combined'

# Prints the path to the generated NativeFunctions.h to stdout.
print_header_path() {
  # buck2 will produce a line like
  # fbcode//executorch/kernels/portable:generated_lib_combined[NativeFunctions.h] buck-out/v2/gen/fbcode/d839c731f5505c62/executorch/codegen/__generated_lib_combined__/out/NativeFunctions.h
  # The sed command chops off everything before the space character.
  # The relative path is relative to fbsource, so we print that first.
  echo -n "${FBSOURCE_DIR}/"
  (
    cd "${FBSOURCE_DIR}/fbcode"
    buck2 build --show-output \
        "${GENRULE_TARGET}[NativeFunctions.h]" 2>&1 \
        | grep '/NativeFunctions.h' \
        | head -1 \
        | sed -e 's/.* //'
  )
}

main() {
  echo "===== Generating header files ====="
  (
    cd "${FBSOURCE_DIR}/fbcode"
    buck2 build "${GENRULE_TARGET}"
  )
  echo ""
  echo "Header file: $(print_header_path)"
}

main "$@"
