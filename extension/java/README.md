# ExecuTorch shared Java sources

This directory holds the **platform-neutral** Java/Kotlin API sources
(`Module`, `Tensor`, `EValue`, `DType`, `LlmModule`, `AsrModule`, training,
…) that are shared by every Java-platform artifact.

## How the sources are consumed

These sources are **not** published as their own artifact. Each platform
build compiles them directly into its own, self-contained artifact by
referencing this directory as an additional source root:

- **Android** (`extension/android/executorch_android`) adds
  `../../java/src/main/java` via `java.srcDirs`, so the shipping AAR embeds
  the same classes it always has — no new transitive dependency, unchanged
  bytecode target (Java 11), unchanged manifest.
- **Desktop JVM** (`extension/jvm`, added separately) compiles the same
  sources into its own jar.

## Contract for sources in this directory

1. **No platform APIs.** Sources here must not import `android.*`,
   `java.awt.*`, or any other platform-specific API. The only permitted
   dependencies are the JDK, fbjni, and soloader.
2. **Logging goes through the `org.pytorch.executorch.Log` facade.** The
   facade is referenced by simple name (same package, no import) and is
   *not* defined in this directory. Each platform artifact compiles exactly
   one implementation (Android: backed by `android.util.Log`; desktop:
   console-backed), so there is never a duplicate-class collision and each
   platform keeps its native logging behavior.
3. **Native library loading stays per-platform.** Shared sources load the
   native library through soloader's `NativeLoader`, whose delegate is
   configured by the platform runtime, not by a cross-platform mutable
   global.
