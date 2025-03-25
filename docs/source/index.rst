.. _home:

ExecuTorch: Powerful On-Device AI Framework
==========================================

.. image:: /_static/img/et-logo.png
   :width: 200px
   :align: center
   :alt: ExecuTorch Logo

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 10px; margin-bottom: 20px;">
     <a href="https://github.com/pytorch/executorch/graphs/contributors">
       <img src="https://img.shields.io/github/contributors/pytorch/executorch?style=for-the-badge&color=blue" alt="Contributors">
     </a>
     <a href="https://github.com/pytorch/executorch/stargazers">
       <img src="https://img.shields.io/github/stars/pytorch/executorch?style=for-the-badge&color=blue" alt="Stargazers">
     </a>
     <a href="https://discord.gg/Dh43CKSAdc">
       <img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community">
     </a>
   </div>

**ExecuTorch** is an end-to-end solution for on-device inference and training. It powers much of Meta's on-device AI experiences across Facebook, Instagram, Meta Quest, Ray-Ban Meta Smart Glasses, WhatsApp, and more.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Quickstart
      :link: getting-started-setup
      :link-type: doc
      :img-top: /_static/img/icon-quickstart.svg
      :class-card: sd-text-center sd-shadow-sm sd-rounded-3

      Get up and running quickly with ExecuTorch.
      
   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc
      :img-top: /_static/img/icon-tutorial.svg
      :class-card: sd-text-center sd-shadow-sm sd-rounded-3

      Step-by-step guides for common tasks.

   .. grid-item-card:: Concepts
      :link: concepts
      :link-type: doc
      :img-top: /_static/img/icon-concepts.svg
      :class-card: sd-text-center sd-shadow-sm sd-rounded-3

      Key concepts and terminology.
      
   .. grid-item-card:: LLM Support
      :link: llm/getting-started
      :link-type: doc
      :img-top: /_static/img/icon-llm.svg
      :class-card: sd-text-center sd-shadow-sm sd-rounded-3

      Run large language models on-device.
      
   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc
      :img-top: /_static/img/icon-api.svg
      :class-card: sd-text-center sd-shadow-sm sd-rounded-3

      Detailed API documentation.
      
   .. grid-item-card:: Architecture
      :link: getting-started-architecture
      :link-type: doc
      :img-top: /_static/img/icon-architecture.svg
      :class-card: sd-text-center sd-shadow-sm sd-rounded-3

      Understand how ExecuTorch works.

Key Features
-----------

.. tab-set::

   .. tab-item:: Portability
      :sync: key1
      
      **Cross-platform compatibility**
      
      ExecuTorch works across a wide variety of computing platforms, from high-end mobile phones to highly constrained embedded systems and microcontrollers.

   .. tab-item:: Productivity
      :sync: key2
      
      **Seamless development workflow**
      
      Use the same toolchains and developer tools from PyTorch model authoring and conversion, to debugging and deployment across diverse platforms.

   .. tab-item:: Performance
      :sync: key3
      
      **Optimized for devices**
      
      Provide end users with a seamless, high-performance experience through a lightweight runtime that fully utilizes hardware capabilities like CPUs, NPUs, and DSPs.

Platforms & Hardware Support
---------------------------

.. grid:: 2

   .. grid-item::
      :columns: 6

      **Operating Systems**

      - iOS
      - Mac
      - Android
      - Linux
      - Microcontrollers

   .. grid-item::
      :columns: 6

      **Hardware Acceleration**

      - Apple
      - Arm
      - Cadence
      - MediaTek
      - Qualcomm
      - Vulkan
      - XNNPACK

Supported Model Types
--------------------

ExecuTorch supports a wide range of AI models, including:

- **LLMs** (Large Language Models)
- **CV** (Computer Vision)
- **ASR** (Automatic Speech Recognition)
- **TTS** (Text to Speech)

Quick Links
----------

.. button-ref:: getting-started-setup
   :ref-type: doc
   :color: primary
   :expand:

   Get Started

.. button-ref:: https://github.com/pytorch/executorch
   :color: secondary
   :expand:

   GitHub Repository

.. button-ref:: https://discord.gg/Dh43CKSAdc
   :color: secondary
   :expand:

   Join Discord

How ExecuTorch Works
-------------------

.. image:: /_static/img/executorch-workflow.png
   :width: 100%
   :alt: ExecuTorch Workflow

ExecuTorch provides an end-to-end solution for deploying PyTorch models to devices:

1. **Export** - Convert your PyTorch model to the ExecuTorch format
2. **Optimize** - Apply quantization, pruning, and other optimizations
3. **Deploy** - Run your model on-device with the lightweight runtime

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

Tutorials and Examples
~~~~~~~~~~~~~~~~~~~~~~

Ready to experiment? Check out some of the
ExecuTorch tutorials.

.. customcardstart::

.. customcarditem::
   :header: Exporting to ExecuTorch Tutorial
   :card_description: A tutorial for exporting a model and lowering a it to be runnable on the ExecuTorch runtime.
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/export-to-executorch-tutorial.html
   :tags: Export,Delegation,Quantization

.. customcarditem::
   :header: Running an ExecuTorch Model C++ Tutorial
   :card_description: A tutorial for setting up memory pools, loading a model, setting inputs, executing the model, and retrieving outputs on device.
   :image: _static/img/generic-pytorch-logo.png
   :link: running-a-model-cpp-tutorial.html
   :tags:

.. customcarditem::
   :header: Simplified Runtime APIs Tutorial
   :card_description: A simplified tutorial for executing the model on device.
   :image: _static/img/generic-pytorch-logo.png
   :link: extension-module.html
   :tags:

.. customcarditem::
   :header: Managing Tensor Memory in C++ Tutorial
   :card_description: A tutorial for managing the dynamic memory when working with tensors.
   :image: _static/img/generic-pytorch-logo.png
   :link: extension-tensor.html
   :tags:

.. customcarditem::
   :header: Using the ExecuTorch Developer Tools to Profile a Model
   :card_description: A tutorial for using the ExecuTorch Developer Tools to profile and analyze a model with linkage back to source code.
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/devtools-integration-tutorial.html
   :tags: devtools

.. customcarditem::
   :header: Integrating and Running ExecuTorch on Apple Platforms
   :card_description: A tutorial on integrating, using, and troubleshooting the ExecuTorch runtime on iOS.
   :image: _static/img/generic-pytorch-logo.png
   :link: apple-runtime.html
   :tags: iOS, macOS

.. customcarditem::
   :header: Building an ExecuTorch iOS Demo App
   :card_description: A demo tutorial that explains how to build ExecuTorch into iOS frameworks and run an iOS app.
   :image: _static/img/demo_ios_app.jpg
   :link: demo-apps-ios.html
   :tags: Delegation,iOS

.. customcarditem::
   :header: Building an ExecuTorch Android Demo App
   :card_description: A demo tutorial that explains how to build ExecuTorch into a JNI library and build an Android app.
   :image: _static/img/android_app.png
   :link: demo-apps-android.html
   :tags: Delegation,Android

.. customcarditem::
   :header: Lowering a Model as a Delegate
   :card_description: An end-to-end example showing how to lower a model as a delegate
   :image: _static/img/generic-pytorch-logo.png
   :link: examples-end-to-end-to-lower-model-to-delegate.html
   :tags: Export,Delegation

..
   First-party backends that are good intros for readers.

.. customcarditem::
   :header: Building and Running ExecuTorch with XNNPACK Backend
   :card_description: A demo tutorial for lowering and exporting models with the XNNPACK Backend
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorial-xnnpack-delegate-lowering.html
   :tags: Export,Backend,Delegation,Quantization,XNNPACK

.. customcarditem::
   :header: Building and Running ExecuTorch with Vulkan Backend
   :card_description: A tutorial that walks you through the process of building ExecuTorch with Vulkan Backend
   :image: _static/img/generic-pytorch-logo.png
   :link: backends-vulkan.html
   :tags: Export,Backend,Delegation,Vulkan

..
   Alphabetical by backend name. Be sure to keep the same order in the Tutorials
   toctree entry above.

.. customcarditem::
   :header: Building and Running ExecuTorch with ARM Ethos-U Backend
   :card_description: A tutorial that walks you through the process of building ExecuTorch with ARM Ethos-U Backend
   :image: _static/img/generic-pytorch-logo.png
   :link: executorch-arm-delegate-tutorial.html
   :tags: Export,Backend,Delegation,ARM,Ethos-U

.. customcarditem::
   :header: Building and Running ExecuTorch with CoreML Backend
   :card_description: A tutorial that walks you through the process of building ExecuTorch with CoreML Backend
   :image: _static/img/generic-pytorch-logo.png
   :link: backends-coreml.html
   :tags: Export,Backend,Delegation,CoreML

.. customcarditem::
   :header: Building and Running ExecuTorch with MediaTek Backend
   :card_description: A tutorial that walks you through the process of building ExecuTorch with MediaTek Backend
   :image: _static/img/generic-pytorch-logo.png
   :link: backends-mediatek-backend.html
   :tags: Export,Backend,Delegation,MediaTek

.. customcarditem::
   :header: Building and Running ExecuTorch with MPS Backend
   :card_description: A tutorial that walks you through the process of building ExecuTorch with MPSGraph Backend
   :image: _static/img/generic-pytorch-logo.png
   :link: backends-mps.html
   :tags: Export,Backend,Delegation,MPS,MPSGraph

.. customcarditem::
   :header: Building and Running ExecuTorch with Qualcomm AI Engine Direct Backend
   :card_description: A tutorial that walks you through the process of building ExecuTorch with Qualcomm AI Engine Direct Backend
   :image: _static/img/generic-pytorch-logo.png
   :link: backends-qualcomm.html
   :tags: Export,Backend,Delegation,QNN

.. customcarditem::
   :header: Building and Running ExecuTorch on Xtensa HiFi4 DSP
   :card_description: A tutorial that walks you through the process of building ExecuTorch for an Xtensa Hifi4 DSP using custom operators
   :image: _static/img/generic-pytorch-logo.png
   :link: backends-cadence.html
   :tags: Export,Custom-Operators,DSP,Xtensa

.. customcardend::
