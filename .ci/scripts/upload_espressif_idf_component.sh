#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Uploads the component archive that test_espressif_idf_nightly.sh prepare made
# to an ESP Component Registry as pytorch/executorch, and succeeds if the
# registry already has that version.
#
#   upload_espressif_idf_component.sh <dir> <registry URL> [--dry-run]
#
# Without --dry-run the job needs `id-token: write`: like
# espressif/upload-components-ci-action, this authenticates with a GitHub OIDC
# token for the registry, which accepts it when the component's trusted
# uploader names this repository and the job's environment. --dry-run
# validates the archive and needs no token.

set -euo pipefail

usage="usage: $0 <dir> <registry URL> [--dry-run]"
if (( $# < 2 || $# > 3 )) || [[ $# == 3 && "$3" != --dry-run ]]; then
    echo "${usage}" >&2
    exit 1
fi
dir=$1 registry=$2 dry_run=("${@:3}")
version=$(cat "${dir}/version")

python -m pip install --progress-bar off --require-hashes \
    -r "$(dirname "${BASH_SOURCE[0]}")/espressif_idf_upload_requirements.txt"
export IDF_COMPONENT_REGISTRY_URL="${registry}"
if (( ${#dry_run[@]} == 0 )); then
    token=$(curl -sSfG --data-urlencode "audience=${registry}" \
        -H "Authorization: Bearer ${ACTIONS_ID_TOKEN_REQUEST_TOKEN:?needs id-token: write}" \
        "${ACTIONS_ID_TOKEN_REQUEST_URL}" | jq -er .value)
    echo "::add-mask::${token}"
    export IDF_COMPONENT_API_TOKEN="${token}"
fi
compote component upload --namespace pytorch --name executorch \
    --archive "${dir}/dist/executorch_${version}.tgz" --allow-existing "${dry_run[@]}"
