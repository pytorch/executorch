<?xml version="1.0" encoding="UTF-8"?>
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

    <condition id="Cortex-A">
      <description>Cortex-A processor (Linux userspace host)</description>
      <accept Dcore="Cortex-A5"/>
      <accept Dcore="Cortex-A7"/>
      <accept Dcore="Cortex-A9"/>
      <accept Dcore="Cortex-A15"/>
      <accept Dcore="Cortex-A17"/>
      <accept Dcore="Cortex-A32"/>
      <accept Dcore="Cortex-A35"/>
      <accept Dcore="Cortex-A53"/>
      <accept Dcore="Cortex-A55"/>
      <accept Dcore="Cortex-A57"/>
      <accept Dcore="Cortex-A65"/>
      <accept Dcore="Cortex-A72"/>
      <accept Dcore="Cortex-A73"/>
      <accept Dcore="Cortex-A75"/>
      <accept Dcore="Cortex-A76"/>
      <accept Dcore="Cortex-A77"/>
      <accept Dcore="Cortex-A78"/>
      <accept Dcore="Cortex-A510"/>
      <accept Dcore="Cortex-A710"/>
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

    <condition id="Ethos-U Linux">
      <description>Ethos-U NPU backend (Cortex-A host, Linux userspace driver)</description>
      <require condition="Cortex-A"/>
      <!-- Linux userspace driver headers (ethosu.hpp, uapi/ethosu.h) are
           provided by the consumer project, not by this pack. -->
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

    <!-- ==================== Backends ==================== -->
    
    <!-- Ethos-U Backend (Cortex-M host, bare-metal) -->
    <!-- Ships only EthosUBackend_Cortex_M.cpp; mutually exclusive with the
         "Backend EthosU Linux" component below via the Cortex-M/Cortex-A
         Dcore conditions. The two host TUs define the same platform_*
         symbols, so they cannot be linked together. -->
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

    <!-- Ethos-U Backend (Cortex-A host, Linux userspace driver) -->
    <!-- Ships only EthosUBackend_Cortex_A.cpp. This TU #includes
         <ethosu.hpp> and <uapi/ethosu.h> from the Linux userspace driver
         stack, which must be supplied by the consumer project. Selectable
         only on Cortex-A targets. -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Backend EthosU Linux" Cversion="%{RELEASE_VERSION}%" condition="Ethos-U Linux">
      <description>ExecuTorch Ethos-U NPU Backend - Cortex-A host via Linux userspace driver. Requires ethosu.hpp / uapi/ethosu.h from the consumer project.</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_BACKEND_ETHOS_U_LINUX     /* ExecuTorch Ethos-U Backend (Cortex-A/Linux host) */
      </RTE_Components_h>
      <Pre_Include_Global_h>
        #define ET_USE_ETHOS_U_BACKEND 1
      </Pre_Include_Global_h>
      <files>
%{ETHOS_U_BACKEND_CORTEX_A_FILES}%
      </files>
    </component>
    
    <!-- Cortex-M Backend -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Backend CortexM" Cversion="%{RELEASE_VERSION}%" condition="CMSIS-NN">
      <description>ExecuTorch Cortex-M Backend - CMSIS-NN optimized operators</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_BACKEND_CORTEX_M     /* ExecuTorch Cortex-M Backend */
      </RTE_Components_h>
      <Pre_Include_Global_h>
        #define EXECUTORCH_BUILD_CORTEX_M 1
      </Pre_Include_Global_h>
      <files>
%{CORTEX_M_BACKEND_FILES}%
      </files>
    </component>

    <!-- ==================== Platform Stubs for Bare-Metal ==================== -->
    
    <!-- Bare-Metal Platform Abstraction Layer -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Platform Bare-Metal" Cversion="%{RELEASE_VERSION}%" condition="Runtime">
      <description>ExecuTorch Bare-Metal PAL - Platform stubs for ARM Cortex-M bare-metal targets</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_PLATFORM_BARE_METAL     /* ExecuTorch Bare-Metal PAL */
      </RTE_Components_h>
      <files>
        <file category="sourceCpp" name="src/stubs/posix_stub.cpp"/>
      </files>
    </component>

    <!-- Random Operation Stubs -->
    <component Cclass="Machine Learning" Cgroup="ExecuTorch" Csub="Stubs RandomOps" Cversion="%{RELEASE_VERSION}%" condition="Runtime">
      <description>ExecuTorch Random Op Stubs - Stubs for rand/randn on platforms without std::random_device</description>
      <RTE_Components_h>
        #define RTE_ML_EXECUTORCH_STUBS_RANDOM_OPS     /* ExecuTorch Random Op Stubs */
      </RTE_Components_h>
      <files>
        <file category="sourceCpp" name="src/stubs/random_ops_stubs.cpp"/>
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
