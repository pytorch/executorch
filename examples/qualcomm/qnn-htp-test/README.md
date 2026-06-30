# QNN HTP cDSP Test APK

Minimal Android app that tests whether a regular installed app (`untrusted_app` SELinux
domain) can initialize the QNN HTP backend and open a FastRPC channel to the Hexagon
cDSP on Qualcomm Snapdragon devices.

## What This Tests

Tap a button, call `backendCreate()` via the HTP backend. This opens FastRPC to the
cDSP. If the device blocks it, you get an error. If it succeeds, unsigned PD access
works from a normal installed app.

**Tested and confirmed working on Galaxy S23 (SM8550, SELinux enforcing, untrusted_app).**

## Setup

### 1. Get the QNN Libraries

Download the QNN runtime AAR from Maven Central and extract the native libs:

```bash
curl -L -o /tmp/qnn-runtime.aar \
  "https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/2.44.0/qnn-runtime-2.44.0.aar"

mkdir -p app/src/main/jniLibs/arm64-v8a
cd app/src/main/jniLibs/arm64-v8a
unzip /tmp/qnn-runtime.aar 'jni/arm64-v8a/*.so'
mv jni/arm64-v8a/*.so .
rm -rf jni
```

This includes skels for V68–V81, covering S23 through S25.

### 2. Build

```bash
./gradlew assembleDebug
```

### 3. Install & Run

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Launch the app, tap **"Run QNN HTP Test"**. Results appear on screen.

## Interpreting Results

- **`SUCCESS! backendCreate returned 0`** → cDSP works from `untrusted_app`
- **Error 4000+** → Transport/SELinux block
- **Error 2000** → Check skel version matches device Hexagon arch

## Supported Devices

| Chipset | Hexagon | Devices |
|---------|---------|---------|
| SM8550 | V73 | Galaxy S23 family |
| SM8650 | V75 | Galaxy S24 family |
| SM8750 | V79/V81 | Galaxy S25 family |

## How It Works

QNN libraries are bundled inside the APK via `jniLibs`. On install, Android extracts
them (`extractNativeLibs="true"`). The JNI code:

1. Sets `ADSP_LIBRARY_PATH` to the app's `nativeLibraryDir`
2. `dlopen("libQnnHtp.so")` → `dlsym("QnnInterface_getProviders")`
3. Calls `logCreate()` and `backendCreate()` via the interface function pointer table
4. Reports success or failure

No QNN SDK headers needed — types are inlined. The manifest entry
`<uses-native-library android:name="libcdsprpc.so"/>` makes the vendor FastRPC client
accessible on Android 12+.
