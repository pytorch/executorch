/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fbjni/fbjni.h>
#include <jni.h>

#ifdef EXECUTORCH_ANDROID_PROFILING

#include "jni_etdump.h"

namespace executorch {
namespace extension {
namespace jni {

// ============================================================================
// Global ETDump Manager
// ============================================================================

// Global ETDump manager (one per process)
static std::unique_ptr<ETDumpManager> g_etdump_manager = nullptr;

extern "C" ETDumpManager* getGlobalETDumpManager() {
  return g_etdump_manager.get();
}

// ============================================================================
// JNI Methods for ETDump Java class
// ============================================================================

// Initialize ETDump manager
extern "C" JNIEXPORT void JNICALL
Java_org_pytorch_executorch_ETDump_nativeInit(JNIEnv* env, jclass clazz) {
  if (!g_etdump_manager) {
    g_etdump_manager = std::make_unique<ETDumpManager>();
  }
}

// Enable profiling
extern "C" JNIEXPORT jboolean JNICALL
Java_org_pytorch_executorch_ETDump_nativeEnableProfiling(
    JNIEnv* env,
    jclass clazz) {
  if (!g_etdump_manager) {
    g_etdump_manager = std::make_unique<ETDumpManager>();
  }
  g_etdump_manager->enableProfiling();
  return JNI_TRUE;
}

// Disable profiling
extern "C" JNIEXPORT void JNICALL
Java_org_pytorch_executorch_ETDump_nativeDisableProfiling(
    JNIEnv* env,
    jclass clazz) {
  if (g_etdump_manager) {
    g_etdump_manager->disableProfiling();
  }
}

// Check if profiling is enabled
extern "C" JNIEXPORT jboolean JNICALL
Java_org_pytorch_executorch_ETDump_nativeIsProfilingEnabled(
    JNIEnv* env,
    jclass clazz) {
  if (!g_etdump_manager) {
    return JNI_FALSE;
  }
  return g_etdump_manager->isProfilingEnabled() ? JNI_TRUE : JNI_FALSE;
}

} // namespace jni
} // namespace extension
} // namespace executorch

#endif // EXECUTORCH_ANDROID_PROFILING
