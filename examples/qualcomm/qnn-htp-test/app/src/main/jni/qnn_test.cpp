// clang-format off
// Minimal QNN HTP cDSP test — no SDK headers needed, types inlined.
// clang-format on

#include <android/log.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <jni.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sstream>
#include <string>

#define TAG "QnnHtpTest"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

typedef uint32_t Qnn_ErrorHandle_t;
typedef uint32_t QnnLog_Level_t;
#define QNN_SUCCESS 0

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn)(
    const void*** providerList,
    uint32_t* numProviders);

// Offsets determined by brute-force probe on S23 (QNN 2.36 + 2.44):
//   Offset 48: backendCreate -> returns SUCCESS
//   Offset 240: logCreate -> returns SUCCESS with handle
// The versioned sub-struct has one extra fn ptr at [40] before backendCreate.
#define OFF_BACKEND_CREATE 48
#define OFF_LOG_CREATE 240

typedef void (*QnnLogCallbackFn)(
    const char* fmt,
    QnnLog_Level_t level,
    uint64_t timestamp,
    va_list argp);

typedef Qnn_ErrorHandle_t (*QnnLogCreateFn)(
    QnnLogCallbackFn callback,
    QnnLog_Level_t maxLevel,
    void** logHandle);

typedef Qnn_ErrorHandle_t (*QnnBackendCreateFn)(
    void* logHandle,
    const void** config,
    void** backendHandle);

static void log_and_append(std::ostringstream& ss, const char* fmt, ...) {
  char buf[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  LOGI("%s", buf);
  ss << buf << "\n";
}

static void qnnLogCallback(
    const char* fmt,
    QnnLog_Level_t level,
    uint64_t timestamp,
    va_list argp) {
  if (!fmt)
    return;
  char buf[512];
  vsnprintf(buf, sizeof(buf), fmt, argp);
  LOGI("[QNN-L%u] %s", level, buf);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_qnntest_MainActivity_runQnnHtpTest(
    JNIEnv* env,
    jobject thiz,
    jstring nativeLibDirJStr) {
  std::ostringstream log;
  const char* nativeLibDir = env->GetStringUTFChars(nativeLibDirJStr, nullptr);
  bool backendCreated = false;

  log_and_append(log, "=== QNN HTP cDSP Test ===");

  log_and_append(log, "[1] ADSP_LIBRARY_PATH = %s", nativeLibDir);
  setenv("ADSP_LIBRARY_PATH", nativeLibDir, 1);

  void* rpcLib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
  log_and_append(
      log, "[2] libcdsprpc.so: %s", rpcLib ? "LOADED OK" : dlerror());
  if (rpcLib)
    dlclose(rpcLib);

  void* htpLib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!htpLib) {
    log_and_append(log, "FAIL: %s", dlerror());
    goto done;
  }
  log_and_append(log, "[3] libQnnHtp.so: LOADED OK");

  {
    auto getProviders =
        (QnnInterfaceGetProvidersFn)dlsym(htpLib, "QnnInterface_getProviders");
    const void** providers = nullptr;
    uint32_t numProviders = 0;
    Qnn_ErrorHandle_t err = getProviders(&providers, &numProviders);
    if (err != QNN_SUCCESS || numProviders == 0 || !providers ||
        !providers[0]) {
      log_and_append(log, "FAIL: getProviders err=%u n=%u", err, numProviders);
      goto cleanup;
    }

    const uint8_t* iface = (const uint8_t*)providers[0];

    uint32_t backendId;
    memcpy(&backendId, iface + 0, sizeof(backendId));

    const char* providerName;
    memcpy(&providerName, iface + 8, sizeof(providerName));

    uint32_t coreVer[3], beVer[3];
    memcpy(coreVer, iface + 16, sizeof(coreVer));
    memcpy(beVer, iface + 28, sizeof(beVer));

    log_and_append(log, "[4] Interface:");
    log_and_append(log, "    backendId: %u", backendId);
    log_and_append(
        log, "    providerName: %s", providerName ? providerName : "(null)");
    log_and_append(
        log,
        "    coreApiVersion: %u.%u.%u",
        coreVer[0],
        coreVer[1],
        coreVer[2]);
    log_and_append(
        log, "    backendApiVersion: %u.%u.%u", beVer[0], beVer[1], beVer[2]);

    log_and_append(log, "[5] logCreate (QNN_LOG_LEVEL_ERROR=1) ...");
    QnnLogCreateFn logCreate;
    memcpy(&logCreate, iface + OFF_LOG_CREATE, sizeof(logCreate));
    void* logHandle = nullptr;
    err = logCreate(qnnLogCallback, 1, &logHandle);
    log_and_append(log, "    result: err=%u handle=%p", err, logHandle);

    if (err != QNN_SUCCESS) {
      log_and_append(log, "    Retrying with nullptr callback ...");
      err = logCreate(nullptr, 1, &logHandle);
      log_and_append(log, "    result: err=%u handle=%p", err, logHandle);
    }

    log_and_append(
        log, "[6] backendCreate(logHandle=%p, config=NULL) ...", logHandle);
    log_and_append(log, "    (This is the call that opens FastRPC to cDSP)");
    QnnBackendCreateFn backendCreate;
    memcpy(&backendCreate, iface + OFF_BACKEND_CREATE, sizeof(backendCreate));
    void* backendHandle = nullptr;
    err = backendCreate(logHandle, nullptr, &backendHandle);

    log_and_append(log, "");
    if (err == QNN_SUCCESS) {
      backendCreated = true;
      log_and_append(log, "========================================");
      log_and_append(log, "SUCCESS! backendCreate returned 0");
      log_and_append(log, "QNN HTP backend created successfully!");
      log_and_append(log, "cDSP FastRPC WORKS from untrusted_app.");
      log_and_append(log, "========================================");
    } else {
      log_and_append(log, "========================================");
      log_and_append(log, "backendCreate error: %u", err);
      log_and_append(log, "========================================");

      if (err >= 1000 && err < 2000) {
        log_and_append(log, "Range: QNN_BACKEND_ERROR (1000-1999)");
      } else if (err == 2000) {
        log_and_append(log, "QNN_COMMON_ERROR_GENERAL (2000)");
      } else if (err >= 4000 && err < 5000) {
        log_and_append(log, "TRANSPORT ERROR -- SELinux blocking!");
      } else if (err >= 6000 && err < 7000) {
        log_and_append(log, "TRANSPORT COMMUNICATION ERROR");
      }
    }

    log_and_append(log, "");
    log_and_append(log, "[7] Process: pid=%d uid=%d", getpid(), getuid());

    std::string skelPath = std::string(nativeLibDir) + "/libQnnHtpV73Skel.so";
    log_and_append(
        log,
        "[8] Skel check: %s %s",
        skelPath.c_str(),
        (access(skelPath.c_str(), R_OK) == 0) ? "EXISTS" : "MISSING!");
  }

cleanup:
  // Don't dlclose if backend is alive — handles are sentinels (e.g. 0x1)
  // that crash on free, and unloading the lib under an active backend is UB.
  if (!backendCreated) {
    dlclose(htpLib);
  }
done:
  env->ReleaseStringUTFChars(nativeLibDirJStr, nativeLibDir);
  return env->NewStringUTF(log.str().c_str());
}
