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
$ADB_PATH wait-for-device shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 30; done;'

# The device will be created by ReactiveCircus/android-emulator-runner GHA
echo "List all running emulators"
$ADB_PATH devices

# TODO: Run tests on emulator here
