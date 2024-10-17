#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

BUILD_AAR_DIR="$(mktemp -d)"
mkdir -p "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}" "${BUILD_AAR_DIR}/libs"
cp "${CMAKE_OUT}"/extension/android/libexecutorch_jni.so "${BUILD_AAR_DIR}/jni/${ANDROID_ABI}/libexecutorch.so"
cp extension/android/build/libs/executorch.jar "${BUILD_AAR_DIR}/libs"
echo \<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\" \
  package=\"org.pytorch.executorch\"\> \
  \<uses-sdk android:minSdkVersion=\"19\" /\> \
  \</manifest\> > "${BUILD_AAR_DIR}/AndroidManifest.xml"
pushd "${BUILD_AAR_DIR}"
zip -r executorch-llama.aar libs jni/${ANDROID_ABI} AndroidManifest.xml
popd
mkdir -p examples/demo-apps/android/LlamaDemo/app/libs
mv "${BUILD_AAR_DIR}/executorch-llama.aar" examples/demo-apps/android/LlamaDemo/app/libs
