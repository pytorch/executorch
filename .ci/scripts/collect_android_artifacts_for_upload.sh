#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


collect_artifacts_to_be_uploaded() {
  ARTIFACTS_DIR_NAME="$1"
  DEMO_APP_DIR="${ARTIFACTS_DIR_NAME}/llm_demo"
  # The app directory is named using its build flavor as a suffix.
  mkdir -p "${DEMO_APP_DIR}"
  # Collect the app and its test suite
  cp examples/demo-apps/android/LlamaDemo/app/build/outputs/apk/debug/*.apk "${DEMO_APP_DIR}" || true
  cp examples/demo-apps/android/LlamaDemo/app/build/outputs/apk/androidTest/debug/*.apk "${DEMO_APP_DIR}" || true
  # Collect JAR and AAR
  cp extension/android/build/libs/executorch.jar "${DEMO_APP_DIR}"
  find "${BUILD_AAR_DIR}/" -name 'executorch*.aar' -exec cp {} "${DEMO_APP_DIR}" \;
  # Collect MiniBench APK
  MINIBENCH_APP_DIR="${ARTIFACTS_DIR_NAME}/minibench"
  mkdir -p "${MINIBENCH_APP_DIR}"
  cp extension/benchmark/android/benchmark/app/build/outputs/apk/debug/*.apk "${MINIBENCH_APP_DIR}" || true
  cp extension/benchmark/android/benchmark/app/build/outputs/apk/androidTest/debug/*.apk "${MINIBENCH_APP_DIR}" || true
  # Collect Java library test
  JAVA_LIBRARY_TEST_DIR="${ARTIFACTS_DIR_NAME}/library_test_dir"
  mkdir -p "${JAVA_LIBRARY_TEST_DIR}"
  cp extension/android_test/build/outputs/apk/debug/*.apk "${JAVA_LIBRARY_TEST_DIR}" || true
  cp extension/android_test/build/outputs/apk/androidTest/debug/*.apk "${JAVA_LIBRARY_TEST_DIR}" || true
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  collect_artifacts_to_be_uploaded "$@"
fi
