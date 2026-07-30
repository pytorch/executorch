#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -u

echo "Validating Arm public API manifests"

manifest_failures=0
for manifest_path in backends/arm/public_api_manifests/api_manifest_*.toml; do
    if [[ ! -f "${manifest_path}" ]]; then
        continue
    fi
    manifest_name="${manifest_path##*/}"
    echo
    echo "=== ${manifest_name} ==="
    validator_output=$(
        python backends/arm/scripts/public_api_manifest/validate_public_api_manifest.py \
            --manifest "${manifest_path}" 2>&1
    )
    validator_status=$?
    printf '%s\n' "${validator_output}"
    if [[ ${validator_status} -ne 0 ]]; then
        manifest_failures=$((manifest_failures + 1))
    fi
done

echo
if [[ ${manifest_failures} -eq 0 ]]; then
    echo "Arm public API manifests OK"
else
    echo "${manifest_failures} manifest(s) failed validation"
    exit 1
fi
