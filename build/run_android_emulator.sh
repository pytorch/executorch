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

adb install -t app-debug.apk
adb install -t app-debug-androidTest.apk

adb shell mkdir -p /data/local/tmp/llama
adb push model.pte /data/local/tmp/llama
adb push tokenizer.bin /data/local/tmp/llama
adb shell am instrument -w -r com.example.executorchllamademo.test/androidx.test.runner.AndroidJUnitRunner

adb install -t android-test-debug.apk
adb install -t android-test-debug-androidTest.apk

adb shell am instrument -w -r org.pytorch.executorch.test/androidx.test.runner.AndroidJUnitRunner
