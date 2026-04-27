# ExecuTorch Android AAR — Consumer ProGuard/R8 Rules
#
# These rules are automatically applied to any app that depends on the
# ExecuTorch AAR. They prevent R8/ProGuard from stripping classes and
# methods that are called from native (JNI) code.

# Keep all classes and members annotated with @DoNotStrip.
# These are explicitly marked as JNI entry points.
-keep @com.facebook.jni.annotations.DoNotStrip class * { *; }
-keepclassmembers class * {
    @com.facebook.jni.annotations.DoNotStrip *;
}

# Keep all native methods across ExecuTorch packages.
-keepclasseswithmembernames class org.pytorch.executorch.** {
    native <methods>;
}

# Keep HybridData fields (accessed by fbjni via reflection).
-keepclassmembers class org.pytorch.executorch.** {
    com.facebook.jni.HybridData *;
}

# Keep ExecutorchRuntimeException and its factory/subclasses.
# These are instantiated from JNI via jni_helper.cpp.
-keep class org.pytorch.executorch.ExecutorchRuntimeException { *; }
-keep class org.pytorch.executorch.ExecutorchRuntimeException$* { *; }

# Keep EValue (fields and type codes accessed from native code).
-keep class org.pytorch.executorch.EValue { *; }

# Keep Tensor and its inner classes. The Tensor class has methods and fields
# accessed from JNI (dtypeJniCode, shape, getRawDataBuffer, nativeNewTensor).
-keep class org.pytorch.executorch.Tensor { *; }
-keep class org.pytorch.executorch.Tensor$* { *; }

# Keep LlmCallback interface methods (invoked from native code).
-keep interface org.pytorch.executorch.extension.llm.LlmCallback { *; }

# Keep AsrCallback interface methods (invoked from native code).
-keep interface org.pytorch.executorch.extension.asr.AsrCallback { *; }
