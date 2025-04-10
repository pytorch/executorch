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

mkdir -p extension/android/executorch_android/src/androidTest/resources
cp extension/module/test/resources/add.pte extension/android/executorch_android/src/androidTest/resources

pushd extension/android
ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew :executorch_android:testDebugUnitTest
ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew :executorch_android:assembleAndroidTest
popd
