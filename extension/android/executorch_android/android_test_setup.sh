#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi
which "${PYTHON_EXECUTABLE}"

BASEDIR=$(dirname "$(realpath $0)")
DEFAULT_GOLDEN_ARTIFACTS_BASE_URL="https://gha-artifacts.s3.amazonaws.com/pytorch/executorch" # @lint-ignore
DEFAULT_GOLDEN_ARTIFACTS_BASE_URL+="/test-backend-artifacts/golden-artifacts-xnnpack"
GOLDEN_ARTIFACTS_BASE_URL="${GOLDEN_ARTIFACTS_BASE_URL:-${DEFAULT_GOLDEN_ARTIFACTS_BASE_URL}}"
export GOLDEN_ARTIFACTS_BASE_URL

prepare_xor() {
  pushd "${BASEDIR}/../../training/"
  python3 -m examples.XOR.export_model  --outdir "${BASEDIR}/src/androidTest/resources/"
  mv "${BASEDIR}/src/androidTest/resources/xor.pte" "${BASEDIR}/src/androidTest/resources/xor_full.pte"
  python3 -m examples.XOR.export_model  --outdir "${BASEDIR}/src/androidTest/resources/" --external
  popd
}

prepare_tinyllama() {
  local S3_BASE="https://ossci-android.s3.amazonaws.com/executorch/stories/snapshot-20260114"
  curl -fL --retry 3 --retry-all-errors -C - \
    "${S3_BASE}/stories110M.pte" \
    --output "${BASEDIR}/src/androidTest/resources/stories.pte"
  curl -fL --retry 3 --retry-all-errors -C - \
    "${S3_BASE}/tokenizer.model" \
    --output "${BASEDIR}/src/androidTest/resources/tokenizer.bin"
}

find_latest_golden_artifacts_url() {
  "${PYTHON_EXECUTABLE}" <<'PY'
from datetime import datetime, timedelta, timezone
import os
import sys
import urllib.error
import urllib.request

base_url = os.environ["GOLDEN_ARTIFACTS_BASE_URL"]
lookback_days = int(os.environ.get("GOLDEN_ARTIFACT_LOOKBACK_DAYS", "14"))

for hours_ago in range(lookback_days * 24):
    timestamp = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime(
        "%y%m%d%H"
    )
    url = f"{base_url}/golden_artifacts_{timestamp}.zip"
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if 200 <= response.status < 300:
                print(url)
                sys.exit(0)
    except urllib.error.HTTPError as error:
        if error.code not in (403, 404):
            print(f"Skipping {url}: HTTP {error.code}", file=sys.stderr)
    except Exception as error:
        print(f"Skipping {url}: {error}", file=sys.stderr)

print(
    f"No XNNPACK golden artifact found under {base_url} from the last {lookback_days} days",
    file=sys.stderr,
)
sys.exit(1)
PY
}

prepare_golden() {
  local url="${GOLDEN_ARTIFACTS_URL:-}"
  if [[ -z "${url}" ]]; then
    url=$(find_latest_golden_artifacts_url)
  fi
  echo "Downloading XNNPACK golden artifacts from ${url}"
  rm -rf /tmp/golden /tmp/golden.zip
  curl -fL --retry 3 --retry-all-errors -o /tmp/golden.zip "$url"
  unzip -o /tmp/golden.zip -d /tmp/golden/
  for model in mobilenet_v2 vit_b_16; do
    cp "/tmp/golden/xnnpack/${model}.pte" "${BASEDIR}/src/androidTest/resources/"
    cp /tmp/golden/xnnpack/${model}_input*.bin "${BASEDIR}/src/androidTest/resources/"
    cp /tmp/golden/xnnpack/${model}_expected_output*.bin "${BASEDIR}/src/androidTest/resources/" 2>/dev/null || echo "Warning: no expected_output files for ${model}"
  done
}

prepare_xor
prepare_tinyllama
prepare_golden
