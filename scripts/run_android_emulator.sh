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

# Push large test assets that are not bundled in the APK
if [ -d push-artifacts ] && [ "$(ls -A push-artifacts 2>/dev/null)" ]; then
  adb shell mkdir -p /data/local/tmp/executorch
  adb push push-artifacts/. /data/local/tmp/executorch/
fi

# Download and push golden test artifacts (models + input/output bins)
GOLDEN_URL="https://gha-artifacts.s3.amazonaws.com/pytorch/executorch/test-backend-artifacts/golden-artifacts-xnnpack/golden_artifacts_26022500.zip"
curl -sL -o /tmp/golden.zip "$GOLDEN_URL"
unzip -o /tmp/golden.zip -d /tmp/golden/
adb push /tmp/golden/xnnpack/. /data/local/tmp/executorch/

adb logcat -c
adb shell am instrument -w -r \
  org.pytorch.executorch.test/androidx.test.runner.AndroidJUnitRunner >result.txt 2>&1
adb logcat -d > logcat.txt
cat logcat.txt
grep -q FAILURES result.txt && cat result.txt
grep -q FAILURES result.txt && exit -1
exit 0
