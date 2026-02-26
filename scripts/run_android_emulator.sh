#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# This script is originally adopted from https://github.com/pytorch/pytorch/blob/main/android/run_tests.sh
ADB_PATH=$ANDROID_HOME/platform-tools/adb

echo "Waiting for emulator boot to complete"
# shellcheck disable=SC2016
$ADB_PATH wait-for-device shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 5; done;'

# The device will be created by ReactiveCircus/android-emulator-runner GHA
echo "List all running emulators"
$ADB_PATH devices

adb uninstall org.pytorch.executorch.test || true
adb install -t android-test-debug-androidTest.apk

# Download and push test artifacts via adb instead of bundling in APK
adb shell mkdir -p /data/local/tmp/executorch

S3_BASE="https://ossci-android.s3.amazonaws.com/executorch/stories/snapshot-20260114"
curl -sL -o /tmp/stories.pte "${S3_BASE}/stories110M.pte"
curl -sL -o /tmp/tokenizer.bin "${S3_BASE}/tokenizer.model"
adb push /tmp/stories.pte /data/local/tmp/executorch/
adb push /tmp/tokenizer.bin /data/local/tmp/executorch/

GOLDEN_URL="https://gha-artifacts.s3.amazonaws.com/pytorch/executorch/test-backend-artifacts/golden-artifacts-xnnpack/golden_artifacts_26022500.zip"
GOLDEN_FILES=(
  mobilenet_v2.pte
  mobilenet_v2_input.bin
  mobilenet_v2_expected_output.bin
  vit_b_16.pte
  vit_b_16_input.bin
  vit_b_16_expected_output.bin
)
curl -sL -o /tmp/golden.zip "$GOLDEN_URL"
unzip -o /tmp/golden.zip "${GOLDEN_FILES[@]/#/xnnpack/}" -d /tmp/golden/
for f in "${GOLDEN_FILES[@]}"; do
  adb push "/tmp/golden/xnnpack/$f" /data/local/tmp/executorch/
done

adb logcat -c
adb shell am instrument -w -r \
  org.pytorch.executorch.test/androidx.test.runner.AndroidJUnitRunner >result.txt 2>&1
adb logcat -d > logcat.txt
cat logcat.txt
grep -q FAILURES result.txt && cat result.txt
grep -q FAILURES result.txt && exit -1
exit 0
