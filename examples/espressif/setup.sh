#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

install_idf=true
install_emulator=true
minimal=false
for arg in "$@"; do
    case "${arg}" in
        --minimal) minimal=true ;;
        --skip-idf) install_idf=false ;;
        --skip-emulator) install_emulator=false ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--minimal] [--skip-idf] [--skip-emulator]

Install ESP-IDF and esp-emulator for ESP32-S3 development.
By default, install ESP-IDF's standard ESP32-S3 development tools.
--minimal installs only the Xtensa compiler, SDK Python packages, and emulator.
ESP-IDF Python dependencies are installed in the active venv or conda environment.
ESPRESSIF_TOOLS_DIR selects the install directory (default: ~/.cache/executorch-espressif).
IDF_PATH and IDF_TOOLS_PATH can select an existing checkout of the pinned SDK and tool directory.
EOF
            exit 0
            ;;
        *) echo "Unknown option: ${arg}" >&2; exit 1 ;;
    esac
done

tools_dir="${ESPRESSIF_TOOLS_DIR:-${HOME}/.cache/executorch-espressif}"
idf_version=v6.1
emulator_version=0.48.0
mkdir -p "${tools_dir}"
tools_dir=$(cd "${tools_dir}" && pwd)

if "${install_idf}"; then
    export IDF_PATH="${IDF_PATH:-${tools_dir}/esp-idf}"
    export IDF_TOOLS_PATH="${IDF_TOOLS_PATH:-${tools_dir}/idf-tools}"
    IDF_PYTHON_ENV_PATH=$(python - <<'PY'
import sys
from pathlib import Path

if sys.prefix == sys.base_prefix and not (Path(sys.prefix) / "conda-meta").is_dir():
    sys.exit("Activate your ExecuTorch venv or conda environment before running setup.sh.")
print(sys.prefix)
PY
    )
    export IDF_PYTHON_ENV_PATH
    # ESP-IDF may delete and recreate an environment whose pip check fails.
    "${IDF_PYTHON_ENV_PATH}/bin/python" -m pip --version
    if [[ ! -d "${IDF_PATH}" ]]; then
        git clone --depth 1 --branch "${idf_version}" --recurse-submodules \
            --shallow-submodules https://github.com/espressif/esp-idf.git "${IDF_PATH}"
    fi
    IDF_PATH=$(cd "${IDF_PATH}" && pwd)
    if [[ "$(git -C "${IDF_PATH}" describe --tags --exact-match 2>/dev/null)" != "${idf_version}" ]]; then
        echo "setup.sh requires ESP-IDF ${idf_version}; select a matching IDF_PATH or a fresh ESPRESSIF_TOOLS_DIR." >&2
        exit 1
    fi
    mkdir -p "${IDF_TOOLS_PATH}"
    IDF_TOOLS_PATH=$(cd "${IDF_TOOLS_PATH}" && pwd)
    python "${IDF_PATH}/tools/python_version_checker.py"
    if "${minimal}"; then
        python "${IDF_PATH}/tools/idf_tools.py" install --targets esp32s3 xtensa-esp-elf
    else
        python "${IDF_PATH}/tools/idf_tools.py" install --targets esp32s3
    fi
    python "${IDF_PATH}/tools/idf_tools.py" install-python-env
fi

if "${install_emulator}"; then
    curl --fail --silent --show-error --location --retry 3 \
        "https://raw.githubusercontent.com/espressif/esp-emulator/v${emulator_version}/install.sh" \
        -o "${tools_dir}/install-esp-emu.sh"
    sh "${tools_dir}/install-esp-emu.sh" --version "${emulator_version}" \
        --bin-dir "${tools_dir}/bin" --quiet
    "${tools_dir}/bin/esp-emu" --version
fi

if "${install_idf}" || [[ ! -f "${tools_dir}/setup_path.sh" ]]; then
    {
        printf 'export ESPRESSIF_TOOLS_DIR=%q\n' "${tools_dir}"
        if "${install_idf}"; then
            printf 'export IDF_PATH=%q\nexport IDF_TOOLS_PATH=%q\nexport IDF_PYTHON_ENV_PATH=%q\n' \
                "${IDF_PATH}" "${IDF_TOOLS_PATH}" "${IDF_PYTHON_ENV_PATH}"
            if "${minimal}"; then
                printf 'IDF_SKIP_TOOLS_CHECK=1 '
            fi
            cat <<'EOF'
source "${IDF_PATH}/export.sh" || return
export IDF_TARGET=esp32s3
EOF
        fi
        cat <<'EOF'
export PATH="${ESPRESSIF_TOOLS_DIR}/bin:${PATH}"
EOF
    } > "${tools_dir}/setup_path.sh"
fi
printf '\nActivate the installed tools with:\nsource %q\n' "${tools_dir}/setup_path.sh"
