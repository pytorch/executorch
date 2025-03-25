.. _home:

Welcome to the ExecuTorch Documentation
=======================================

.. raw:: html
   <div>
    <img src="_static/img/et-logo.png" alt="Logo" width="200">
    <p><strong>ExecuTorch</strong> is PyTorch's solution to training and inference on the Edge. </p>
    <h3>Key Value Propositions</h3>
      <ul>
        <li><strong>Portability:</strong> Compatibility with a wide variety of computing platforms, from high-end mobile phones to highly constrained embedded systems and microcontrollers.</li>
        <li><strong>Productivity:</strong> Enabling developers to use the same toolchains and Developer Tools from PyTorch model authoring and conversion, to debugging and deployment to a wide variety of platforms.</li>
        <li><strong>Performance:</strong> Providing end users with a seamless and high-performance experience due to a lightweight runtime and utilizing full hardware capabilities such as CPUs, NPUs, and DSPs.</li>
      </ul>
   </div>
   <div>
   <h3>Support for:</h3>
   <div><strong>Strong Model Support</strong> LLMs (Large Language Models), CV (Comupter Vision), ASR (Automiatic Speaech Recognition), TTS (Text To Speech)</div>
   <div><strong>All Major Platforms</strong> Android, Mac, Linux, Windows</div>
   <div><strong>Rich Acceration Support</strong> Apple, Arm, Cadence, MediaTek, Qualcomm, Vulkan, XNNPACK</div>
   </div>
   <hr style="margin: 5px;"/>
   <div class="documentation-navigation">
      <div>Introduction</div>
      <ul>
         <li><a href="intro-overview">Overview</a></li>
         <li><a href="intro-how-it-works">How it Works</a></li>
         <li><a href="getting-started-architecture">Getting Started with Architecture</a></li>
         <li><a href="concepts">Concepts</a></li>
      </ul>
         <div>Usage</div>
      <ul>
         <li><a href="getting-started">Getting Started</a></li>
         <li><a href="using-executorch-export">Using Executorch Export</a></li>
         <li><a href="using-executorch-android">Using Executorch on Android</a></li>
         <li><a href="using-executorch-ios">Using Executorch on iOS</a></li>
         <li><a href="using-executorch-cpp">Using Executorch with C++</a></li>
         <li><a href="using-executorch-runtime-integration">Runtime Integration</a></li>
         <li><a href="using-executorch-troubleshooting">Troubleshooting</a></li>
         <li><a href="using-executorch-building-from-source">Building from Source</a></li>
         <li><a href="using-executorch-faqs">FAQs</a></li>
      </ul>
      <div>Examples</div>
      <ul>
         <li><a href="demo-apps-android.md">Android Demo Apps</a></li>
         <li><a href="demo-apps-ios.md">iOS Demo Apps</a></li>
      </ul>
      <div>Backends</div>
      <ul>
         <li><a href="backends-overview">Overview</a></li>
         <li><a href="backends-xnnpack">XNNPACK</a></li>
         <li><a href="backends-coreml">Core ML</a></li>
         <li><a href="backends-mps">MPS</a></li>
         <li><a href="backends-vulkan">Vulkan</a></li>
         <li><a href="backends-arm-ethos-u">ARM Ethos-U</a></li>
         <li><a href="backends-qualcomm">Qualcomm</a></li>
         <li><a href="backends-mediatek">MediaTek</a></li>
         <li><a href="backends-cadence">Cadence</a></li>
      </ul>
      <div>Tutorials</div>
      <ul>
         <!-- No items listed -->
      </ul>
      <div>Developer Tools</div>
      <ul>
         <li><a href="devtools-overview">Overview</a></li>
         <li><a href="bundled-io">Bundled IO</a></li>
         <li><a href="etrecord">ETRecord</a></li>
         <li><a href="etdump">ETDump</a></li>
         <li><a href="runtime-profiling">Runtime Profiling</a></li>
         <li><a href="model-debugging">Model Debugging</a></li>
         <li><a href="model-inspector">Model Inspector</a></li>
         <li><a href="memory-planning-inspection">Memory Planning Inspection</a></li>
         <li><a href="delegate-debugging">Delegate Debugging</a></li>
         <li><a href="devtools-tutorial">Tutorial</a></li>
      </ul>
      <div>Runtime</div>
      <ul>
         <li><a href="runtime-overview">Overview</a></li>
         <li><a href="extension-module">Extension Module</a></li>
         <li><a href="extension-tensor">Extension Tensor</a></li>
         <li><a href="running-a-model-cpp-tutorial">Running a Model (C++ Tutorial)</a></li>
         <li><a href="runtime-backend-delegate-implementation-and-linking">Backend Delegate Implementation and Linking</a></li>
         <li><a href="runtime-platform-abstraction-layer">Platform Abstraction Layer</a></li>
      </ul>
      <div>Portable C++ Programming</div>
      <ul>
         <li><a href="pte-file-format">PTE File Format</a></li>
      </ul>
      <div>API Reference</div>
      <ul>
         <li><a href="export-to-executorch-api-reference">Export to Executorch API Reference</a></li>
         <li><a href="executorch-runtime-api-reference">Executorch Runtime API Reference</a></li>
         <li><a href="runtime-python-api-reference">Runtime Python API Reference</a></li>
         <li><a href="api-life-cycle">API Life Cycle</a></li>
         <li><a href="https://pytorch.org/executorch/main/javadoc/">Javadoc</a></li>
      </ul>
      <div>Quantization</div>
      <ul>
         <li><a href="quantization-overview">Overview</a></li>
      </ul>
      <div>Kernel Library</div>
      <ul>
         <li><a href="kernel-library-overview">Overview</a></li>
         <li><a href="kernel-library-custom-aten-kernel">Custom ATen Kernel</a></li>
         <li><a href="kernel-library-selective-build">Selective Build</a></li>
      </ul>
      <div>Working with LLMs</div>
      <ul>
         <li><a href="llm/llama">Llama</a></li>
         <li><a href="llm/llama-demo-android">Llama on Android</a></li>
         <li><a href="llm/llama-demo-ios">Llama on iOS</a></li>
         <li><a href="llm/build-run-llama3-qualcomm-ai-engine-direct-backend">Llama on Android via Qualcomm backend</a></li>
         <li><a href="llm/getting-started">Intro to LLMs in Executorch</a></li>
      </ul>

      <div>Backend Development</div>
      <ul>
         <li><a href="backend-delegates-integration">Delegates Integration</a></li>
         <li><a href="backend-delegates-xnnpack-reference">XNNPACK Reference</a></li>
         <li><a href="backend-delegates-dependencies">Dependencies</a></li>
         <li><a href="compiler-delegate-and-partitioner">Compiler Delegate and Partitioner</a></li>
         <li><a href="debug-backend-delegate">Debug Backend Delegate</a></li>
      </ul>
      <div>IR Specification</div>
      <ul>
         <li><a href="ir-exir">EXIR</a></li>
         <li><a href="ir-ops-set-definition">Ops Set Definition</a></li>
      </ul>
      <div>Compiler Entry Points</div>
      <ul>
         <li><a href="compiler-backend-dialect">Backend Dialect</a></li>
         <li><a href="compiler-custom-compiler-passes">Custom Compiler Passes</a></li>
         <li><a href="compiler-memory-planning">Memory Planning</a></li>
      </ul>

      <div>Contributing</div>
      <ul>
         <li><a href="contributing">Contributing</a></li>
      </ul>
   </div>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Introduction
   :hidden:

   intro-overview
   intro-how-it-works
   getting-started-architecture
   concepts

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Usage
   :hidden:

   getting-started
   using-executorch-export
   using-executorch-android
   using-executorch-ios
   using-executorch-cpp
   using-executorch-runtime-integration
   using-executorch-troubleshooting
   using-executorch-building-from-source
   using-executorch-faqs

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Examples
   :hidden:

   demo-apps-android.md
   demo-apps-ios.md

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Backends
   :hidden:

   backends-overview
   backends-xnnpack
   backends-coreml
   backends-mps
   backends-vulkan
   backends-arm-ethos-u
   backends-qualcomm
   backends-mediatek
   backends-cadence

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Developer Tools
   :hidden:

   devtools-overview
   bundled-io
   etrecord
   etdump
   runtime-profiling
   model-debugging
   model-inspector
   memory-planning-inspection
   delegate-debugging
   devtools-tutorial

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Runtime
   :hidden:

   runtime-overview
   extension-module
   extension-tensor
   running-a-model-cpp-tutorial
   runtime-backend-delegate-implementation-and-linking
   runtime-platform-abstraction-layer
   portable-cpp-programming
   pte-file-format

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   export-to-executorch-api-reference
   executorch-runtime-api-reference
   runtime-python-api-reference
   api-life-cycle
   Javadoc <https://pytorch.org/executorch/main/javadoc/>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quantization
   :hidden:

   quantization-overview

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Kernel Library
   :hidden:

   kernel-library-overview
   kernel-library-custom-aten-kernel
   kernel-library-selective-build

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Working with LLMs
   :hidden:

   Llama <llm/llama>
   Llama on Android <llm/llama-demo-android>
   Llama on iOS <llm/llama-demo-ios>
   Llama on Android via Qualcomm backend <llm/build-run-llama3-qualcomm-ai-engine-direct-backend>
   Intro to LLMs in Executorch <llm/getting-started>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Backend Development
   :hidden:

   backend-delegates-integration
   backend-delegates-xnnpack-reference
   backend-delegates-dependencies
   compiler-delegate-and-partitioner
   debug-backend-delegate

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: IR Specification
   :hidden:

   ir-exir
   ir-ops-set-definition

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Compiler Entry Points
   :hidden:

   compiler-backend-dialect
   compiler-custom-compiler-passes
   compiler-memory-planning

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Contributing
   :hidden:

   contributing
