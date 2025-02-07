/*
 * Copyright (C) 2023 MediaTek Inc., this file is modified on 02/26/2021
 * by MediaTek Inc. based on MIT License .
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the ""Software""), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ""AS IS"", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <android/log.h>
#include <dlfcn.h>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <utility>
using namespace std;

typedef enum {
  LOW_POWER_MODE = 0, // For model execution preference
  FAST_SINGLE_ANSWER_MODE, // For model execution preference
  SUSTAINED_SPEED_MODE, // For model execution preference
  FAST_COMPILE_MODE, // For model compile preference
  PERFORMANCE_MODE_MAX,
} PERFORMANCE_MODE_E;

//------------------------------------- -------------------------------------
#define APUWARE_LOG_D(format, ...) \
  __android_log_print(             \
      ANDROID_LOG_DEBUG, "APUWARELIB", format "\n", ##__VA_ARGS__);

#define APUWARE_LOG_E(format, ...) \
  __android_log_print(             \
      ANDROID_LOG_ERROR, "APUWARELIB", format "\n", ##__VA_ARGS__);

inline void* voidFunction() {
  return nullptr;
}

// ApuWareUtils library construct
struct ApuWareUtilsLib {
  static struct ApuWareUtilsLib& GetInstance() {
    static struct ApuWareUtilsLib instance;
    return instance;
  }

  ApuWareUtilsLib() {
    load();
  }

  using AcquirePerformanceLockPtr =
      std::add_pointer<int32_t(int32_t, PERFORMANCE_MODE_E, uint32_t)>::type;
  using AcquirePerfParamsLockPtr =
      std::add_pointer<int32_t(int32_t, uint32_t, int32_t[], uint32_t)>::type;
  using ReleasePerformanceLockPtr = std::add_pointer<bool(int32_t)>::type;

  // Open a given library and load symbols
  bool load() {
    void* handle = nullptr;
    const std::string libraries[] = {
        "libapuwareutils_v2.mtk.so", "libapuwareutils.mtk.so"};
    for (const auto& lib : libraries) {
      handle = dlopen(lib.c_str(), RTLD_LAZY | RTLD_LOCAL);
      if (handle) {
        APUWARE_LOG_D("dlopen %s", lib.c_str());
        acquirePerformanceLock =
            reinterpret_cast<decltype(acquirePerformanceLock)>(
                dlsym(handle, "acquirePerformanceLockInternal"));
        acquirePerfParamsLock =
            reinterpret_cast<decltype(acquirePerfParamsLock)>(
                dlsym(handle, "acquirePerfParamsLockInternal"));
        releasePerformanceLock =
            reinterpret_cast<decltype(releasePerformanceLock)>(
                dlsym(handle, "releasePerformanceLockInternal"));
        return mEnable = acquirePerformanceLock && releasePerformanceLock &&
            acquirePerfParamsLock;
      } else {
        APUWARE_LOG_E("unable to open library %s", lib.c_str());
      }
    }
    return false;
  }

  bool mEnable = false;

  AcquirePerformanceLockPtr acquirePerformanceLock =
      reinterpret_cast<decltype(acquirePerformanceLock)>(voidFunction);
  AcquirePerfParamsLockPtr acquirePerfParamsLock =
      reinterpret_cast<decltype(acquirePerfParamsLock)>(voidFunction);
  ReleasePerformanceLockPtr releasePerformanceLock =
      reinterpret_cast<decltype(releasePerformanceLock)>(voidFunction);
};

class ScopePerformancer {
 public:
  ScopePerformancer(uint32_t ms = 2000)
      : mLib(ApuWareUtilsLib::GetInstance()), mMs(ms) {
    mLock = mLib.mEnable;
    if (mLock) {
      APUWARE_LOG_D("Powerhal Up");
      mRunning.store(true);
      mThread = std::thread(&ScopePerformancer::acquireLockRepeatedly, this);
    }
  };

  void Stop() {
    if (mRunning.load()) {
      mRunning.store(false);
      mCond.notify_one();
    }
  }

  ~ScopePerformancer() {
    Stop();
    if (mThread.joinable()) {
      mThread.join();
    }
    if (mHalHandle != 0 && mLock) {
      APUWARE_LOG_D("Powerhal Free");
      mLib.releasePerformanceLock(mHalHandle);
      mHalHandle = 0;
    }
  }

 private:
  void acquireLockRepeatedly() {
    std::unique_lock<std::mutex> lock(mMutex);
    while (mRunning.load()) {
      mHalHandle =
          mLib.acquirePerformanceLock(mHalHandle, FAST_SINGLE_ANSWER_MODE, mMs);
      mCond.wait_for(lock, std::chrono::milliseconds(1000), [this] {
        return !mRunning.load();
      });
    }
  }

  struct ApuWareUtilsLib mLib;

  bool mLock = false;

  int mHalHandle = 0;

  uint32_t mMs;

  std::atomic<bool> mRunning{false};

  std::thread mThread;

  std::mutex mMutex;

  std::condition_variable mCond;
};