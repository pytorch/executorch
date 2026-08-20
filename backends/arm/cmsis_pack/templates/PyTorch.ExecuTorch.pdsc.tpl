<?xml version="1.0" encoding="UTF-8"?>
<!--
  Copyright 2026 Arm Limited and/or its affiliates.

  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree.
-->
<package schemaVersion="1.7"
  xmlns:xs="http://www.w3.org/2001/XMLSchema-instance" 
  xs:noNamespaceSchemaLocation="PACK.xsd">
  
  <vendor>PyTorch</vendor>
  <name>ExecuTorch</name>
  <description overview="Documentation/README.md">ExecuTorch: PyTorch Edge Runtime for on-device AI inference on Arm Cortex-M processors.</description>
  
  <url>https://github.com/pytorch/executorch/releases/download/%{RELEASE_VERSION}%/</url>
  <license>LICENSE</license>
  <repository type="git">https://github.com/pytorch/executorch.git</repository>
  
  <releases>
    <release version="%{RELEASE_VERSION}%" date="%{RELEASE_DATE}%">
      ExecuTorch %{RELEASE_VERSION}% - PyTorch Edge Runtime
      - Portable operators for Cortex-M
      - Ethos-U NPU backend support
      - Cortex-M optimized kernels
    </release>
    %{HISTORY}%
  </releases>

  <requirements>
    <packages>
      <package name="CMSIS" vendor="ARM" version="6.0.0:6.99.99"/>
    </packages>
  </requirements>

  <taxonomy>
    <description Cclass="Machine Learning">Software Components for Machine Learning</description>
    <description Cclass="Machine Learning" Cgroup="ExecuTorch">ExecuTorch PyTorch Edge Runtime</description>
    <description Cclass="Machine Learning" Cgroup="ExecuTorch Operators">ExecuTorch Operator Components</description>
  </taxonomy>

  <conditions>
    <!-- Toolchain conditions -->
    <condition id="Cortex-M">
      <description>Cortex-M processor</description>
      <accept Dcore="Cortex-M0"/>
      <accept Dcore="Cortex-M0+"/>
      <accept Dcore="Cortex-M3"/>
      <accept Dcore="Cortex-M4"/>
      <accept Dcore="Cortex-M7"/>
      <accept Dcore="Cortex-M23"/>
      <accept Dcore="Cortex-M33"/>
      <accept Dcore="Cortex-M55"/>
      <accept Dcore="Cortex-M85"/>
    </condition>

    <condition id="GCC">
      <description>GNU Compiler</description>
      <require Tcompiler="GCC"/>
    </condition>
    
    <condition id="AC6">
      <description>Arm Compiler 6</description>
      <require Tcompiler="ARMCC" Toptions="AC6"/>
    </condition>
    
    <condition id="CLANG">
      <description>LLVM/Clang Compiler</description>
      <require Tcompiler="CLANG"/>
    </condition>

    <!-- Component dependency conditions -->
    <condition id="Runtime">
      <description>ExecuTorch Runtime required</description>
      <require condition="Cortex-M"/>
      <require Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Runtime"/>
    </condition>
    
    <condition id="Kernel Utils">
      <description>Kernel registration utilities required</description>
      <require Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Kernel Utils"/>
    </condition>
    
    <condition id="CMSIS-NN">
      <description>CMSIS-NN optimized backend</description>
      <require condition="Runtime"/>
      <require Cclass="CMSIS" Cgroup="NN Lib"/>
    </condition>
    
    <condition id="Ethos-U">
      <description>Ethos-U NPU backend (Cortex-M host, bare-metal)</description>
      <require condition="Cortex-M"/>
      <require Cclass="Machine Learning" Cgroup="NPU Support" Csub="Ethos-U Driver"/>
    </condition>

    <!-- Operator conditions - each operator requires Kernel Utils -->
%{OPERATOR_CONDITIONS}%
  </conditions>

  <components>
    <!-- ==================== Core Runtime ==================== -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Runtime" Cversion="%{RELEASE_VERSION}%" condition="Cortex-M">
      <description>ExecuTorch Core Runtime - Required for all ExecuTorch applications</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_RUNTIME     /* ExecuTorch Runtime */
      </RTE_Components_h>
      <Pre_Include_Global_h>
        /* ExecuTorch global configuration */
        #define C10_USING_CUSTOM_GENERATED_MACROS
        #define FLATBUFFERS_MAX_ALIGNMENT 1024
        #define FLATBUFFERS_LOCALE_INDEPENDENT 0
      </Pre_Include_Global_h>
      <files>
%{RUNTIME_FILES}%
        <file category="include" name="include/"/>
        <!-- AC6 sysroot has no <sys/types.h>; runtime/platform/compiler.h
             includes it unconditionally on non-MSVC. The shim under
             armclang_shims/sys/types.h forwards to #include_next on
             GCC/CLANG (which both ship the real header) and supplies
             ssize_t directly under AC6. -->
        <file category="include" name="armclang_shims/" condition="AC6"/>
      </files>
    </component>

    <!-- ==================== Kernel Utils ==================== -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Kernel Utils" Cversion="%{RELEASE_VERSION}%" condition="Runtime">
      <description>ExecuTorch Kernel Registration Utilities</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_KERNEL_UTILS     /* ExecuTorch Kernel Utils */
      </RTE_Components_h>
      <files>
%{KERNEL_UTILS_FILES}%
      </files>
    </component>

    <!-- ==================== Kernel Registration ==================== -->
    <!-- 
      This component provides unified kernel registration for all operators.
      Each operator registration is guarded by #ifdef RTE_ML_EXECUTORCH_OP_<NAME>,
      which is defined by the corresponding operator component.
      This allows selective registration based on which operators are selected.
    -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Kernel Registration" Cversion="%{RELEASE_VERSION}%" condition="Kernel Utils">
      <description>ExecuTorch Kernel Registration - Registers all selected operators</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_KERNEL_REGISTRATION     /* ExecuTorch Kernel Registration */
      </RTE_Components_h>
      <files>
        <file category="sourceCpp" name="src/registration/RegisterAllKernels.cpp"/>
      </files>
    </component>

    <!-- ==================== Portable Operators ==================== -->
%{PORTABLE_OPERATOR_COMPONENTS}%

    <!-- ==================== Quantized Operators ==================== -->
%{QUANTIZED_OPERATOR_COMPONENTS}%

    <!-- ==================== Cortex-M Operators ==================== -->
    <!-- Each per-op component ships its own op_*.cpp from the Cortex-M
         backend ops/ tree. The ones that route through CMSIS-NN add a
         <require condition="CMSIS-NN"/> on top of "Kernel Utils" via
         their generated condition; quantize/dequantize/softmax are
         pure-CPU and require only "Kernel Utils". -->
%{CORTEX_M_OPERATOR_COMPONENTS}%

    <!-- ==================== Backends ==================== -->
    
    <!-- Ethos-U Backend (Cortex-M host, bare-metal) -->
    <!-- Ships only EthosUBackend_Cortex_M.cpp. The Cortex-A/Linux host
         variant is intentionally not exposed in this pack and may be
         added in a follow-up release. -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Backend EthosU" Cversion="%{RELEASE_VERSION}%" condition="Ethos-U">
      <description>ExecuTorch Ethos-U NPU Backend - Cortex-M host (bare-metal). Hardware acceleration for Ethos-U55/U65</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_BACKEND_ETHOS_U     /* ExecuTorch Ethos-U Backend (Cortex-M host) */
      </RTE_Components_h>
      <Pre_Include_Global_h>
        #define EXECUTORCH_BUILD_ARM_BAREMETAL 1
        #define ET_USE_ETHOS_U_BACKEND 1
      </Pre_Include_Global_h>
      <files>
%{ETHOS_U_BACKEND_CORTEX_M_FILES}%
      </files>
    </component>

  </components>

  <examples>
    <example name="ExecuTorch Inference" doc="README.md" folder="examples/inference">
      <description>Basic ExecuTorch inference example</description>
      <project>
        <environment name="csolution" load="inference.csolution.yml"/>
      </project>
    </example>
  </examples>

</package>
