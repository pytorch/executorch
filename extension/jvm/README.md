# ExecuTorch JVM (Desktop)

ExecuTorch Java/Kotlin bindings for standard desktop JVMs (Linux, macOS, Windows).

This module is a thin sibling of the Android artifact: all platform-neutral API classes
(`Module`, `Tensor`, `EValue`, `DType`, …) come from the shared
[`extension/java`](../java) module (`org.pytorch:executorch-java`). This module only adds
desktop-specific behavior: classpath-based native library delivery and console logging.

## Usage

```kotlin
import org.pytorch.executorch.ExecuTorchJvm
import org.pytorch.executorch.Module

fun main() {
    // Install the desktop native loader before touching any other ExecuTorch API.
    ExecuTorchJvm.init()

    val module = Module.load("/path/to/model.pte")
    // ...
}
```

`ExecuTorchJvm.init()` is explicit and compile-checked — no reflection, no ServiceLoader.

## Native library delivery

The API jar contains no native binaries. Each desktop OS/arch ships as a classified
Maven artifact, following the standard API-jar + per-platform-native-jars pattern:

```
org.pytorch:executorch-jvm:<version>                  (API jar)
org.pytorch:executorch-jvm:<version>:linux-x86_64     (native jar, Linux x86_64)
org.pytorch:executorch-jvm:<version>:linux-aarch64
org.pytorch:executorch-jvm:<version>:macos-x86_64
org.pytorch:executorch-jvm:<version>:macos-aarch64
org.pytorch:executorch-jvm:<version>:windows-x86_64
org.pytorch:executorch-jvm:<version>:windows-aarch64
```

At runtime, `NativeLibraryLoader` extracts `native/<os>/<arch>/libexecutorch_jni.{so,dylib,dll}`
from the classpath to a temp directory and loads it. At publish time, CI stages binaries under
`extension/cmake-out-jvm/<classifier>/` (built with `extension/android/CMakeLists.txt`,
non-Android branch) and the `jarNative*` Gradle tasks in this module package them into the
classified jars.

Gradle dependency example (Linux x86_64 host):

```groovy
implementation "org.pytorch:executorch-jvm:<version>"
implementation "org.pytorch:executorch-jvm:<version>:linux-x86_64"
```
