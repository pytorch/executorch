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

mkdir -p "${BUILD_AAR_DIR}"/executorch_android/src/androidTest/resources
cp extension/module/test/resources/add.pte "${BUILD_AAR_DIR}"/executorch_android/src/androidTest/resources

pushd "${BUILD_AAR_DIR}"
ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew :executorch_android:testDebugUnitTest
ANDROID_HOME="${ANDROID_SDK:-/opt/android/sdk}" ./gradlew :executorch_android:assembleAndroidTest
popd
