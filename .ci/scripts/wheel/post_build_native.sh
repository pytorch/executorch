#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

cd "${NATIVE_WHEEL_ROOT}/backends/native/python"
wheels=(dist/executorch_native-*.whl)
[[ ${#wheels[@]} == 1 && -f "${wheels[0]}" ]]
python -m auditwheel repair "${wheels[0]}" --wheel-dir repaired
rm "${wheels[0]}"
mv repaired/*.whl dist/
wheels=(dist/executorch_native-*.whl)
[[ ${#wheels[@]} == 1 && -f "${wheels[0]}" ]]

# Only the export environment needs PyTorch and the source checkout.
torchao_version=$(cd "${NATIVE_WHEEL_ROOT}" && python -c \
    'from install_requirements import TORCHAO_NIGHTLY_VERSION; print(TORCHAO_NIGHTLY_VERSION)')
python -m pip install torch "torchao==${torchao_version}" numpy flatbuffers packaging pyyaml ruamel.yaml tabulate \
    --index-url https://download.pytorch.org/whl/test/cpu \
    --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
    --extra-index-url https://pypi.org/simple
mkdir -p "${NATIVE_WHEEL_DEPS}/export-source"
ln -s "${NATIVE_WHEEL_ROOT}" "${NATIVE_WHEEL_DEPS}/export-source/executorch"
PYTHONPATH="${NATIVE_WHEEL_DEPS}/export-source" python \
    "${NATIVE_WHEEL_ROOT}/.ci/scripts/wheel/export_native.py" \
    --output "${NATIVE_WHEEL_DEPS}/fixtures"
python "${NATIVE_WHEEL_ROOT}/.ci/scripts/wheel/test_native.py" \
    --wheel "${wheels[0]}" --source "${NATIVE_WHEEL_ROOT}" \
    --fixtures "${NATIVE_WHEEL_DEPS}/fixtures"

# Nova's artifact steps also consume dist at the repository root.
mkdir -p "${NATIVE_WHEEL_ROOT}/dist"
cp "${wheels[0]}" "${NATIVE_WHEEL_ROOT}/dist/"
