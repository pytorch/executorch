#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# This script is originally adopted from https://github.com/pytorch/pytorch/blob/main/android/run_tests.sh
ADB_PATH=$ANDROID_HOME/platform-tools/adb

adb_shell_with_retries() {
  local attempts="$1"
  shift

  for ((i = 1; i <= attempts; i++)); do
    if "$ADB_PATH" shell "$@"; then
      return 0
    fi
    sleep 5
    "$ADB_PATH" wait-for-device
  done

  return 1
}

echo "Waiting for emulator boot to complete"
# shellcheck disable=SC2016
$ADB_PATH wait-for-device shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 5; done;'
$ADB_PATH wait-for-device

# sys.boot_completed flips before system_server finishes registering the
# settings provider and PackageManager. Installing before then fails with
# "Cannot access system provider: 'settings' before system providers are
# installed!". Wait until PackageManager actually answers.
echo "Waiting for package manager to become available"
adb_shell_with_retries 60 cmd package list packages >/dev/null

echo "Unlock emulator and disable animations"
adb_shell_with_retries 5 input keyevent 82 || true
adb_shell_with_retries 5 settings put global window_animation_scale 0.0 || true
adb_shell_with_retries 5 settings put global transition_animation_scale 0.0 || true
adb_shell_with_retries 5 settings put global animator_duration_scale 0.0 || true

# The device will be created by ReactiveCircus/android-emulator-runner GHA
echo "List all running emulators"
$ADB_PATH devices

"$ADB_PATH" uninstall org.pytorch.executorch.test || true

# Retry the install: even after PackageManager answers, a cold software
# boot can transiently drop adb or the framework can still be settling.
install_ok=false
for ((i = 1; i <= 5; i++)); do
  if "$ADB_PATH" install -r -t android-test-debug-androidTest.apk; then
    install_ok=true
    break
  fi
  sleep 10
  "$ADB_PATH" wait-for-device
done
if [ "$install_ok" != "true" ]; then
  echo "Failed to install test APK after retries" >&2
  exit 1
fi

"$ADB_PATH" logcat -c
"$ADB_PATH" shell am instrument -w -r \
  org.pytorch.executorch.test/androidx.test.runner.AndroidJUnitRunner >result.txt 2>&1
"$ADB_PATH" logcat -d > logcat.txt
cat logcat.txt
grep -q FAILURES result.txt && cat result.txt
grep -q FAILURES result.txt && exit -1
exit 0
