.. _home:

Welcome to the ExecuTorch Documentation
=======================================

.. important::
   v0.4.0 is a beta release of ExecuTorch. As of this release, the API will
   follow the `API Lifecycle and Deprecation Policy <api-life-cycle.html>`__,
   and the ``.pte`` binary format will comply with the `Runtime Compatibility
   Policy
   <https://github.com/pytorch/executorch/tree/main/runtime/COMPATIBILITY.md>`__.
   This helps ensure that application developers can update to the latest
   version of ExecuTorch without breaking existing integration code, in
   accordance with these policies. If any issues arise or compatibility breaks
   occur, please `report them in GitHub
   <https://github.com/pytorch/executorch/issues/new/choose>`__.

   We welcome any feedback, suggestions, and bug reports from the community
   to help us improve the technology. Please use the `PyTorch Forums
   <https://discuss.pytorch.org/c/executorch>`__ for discussion and feedback
   about ExecuTorch using the **ExecuTorch** category, and our `GitHub
   repository <https://github.com/pytorch/executorch/issues>`__ for bug
   reporting.

.. raw:: html

   <div class="et-page-column-row">
     <div class="et-page-column1"><p><strong>ExecuTorch</strong> is a PyTorch platform that provides infrastructure to run PyTorch programs everywhere from AR/VR wearables to standard on-device iOS and Android mobile deployments. One of the main goals for ExecuTorch is to enable wider customization and deployment capabilities of the PyTorch programs.</p>
     <p>ExecuTorch heavily relies on such PyTorch technologies as <a href="https://pytorch.org/docs/stable/torch.compiler.html">torch.compile</a> and <a href="https://pytorch.org/docs/main/export.html">torch.export</a>. If you are not familiar with these APIs, you might want to read about them in the PyTorch documentation before diving into the ExecuTorch documentation.</p></div>
     <div class="et-page-column2"><img src="_static/img/ExecuTorch-Logo-cropped.svg" alt="ExecuTorch logo" title="ExecuTorch logo"></div>
   </div>

The ExecuTorch source is hosted on GitHub at
https://github.com/pytorch/executorch.

Getting Started
~~~~~~~~~~~~~~~

Topics in this section will help you get started with ExecuTorch.

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        What is ExecuTorch?
        :img-top: _static/img/card-background.svg
        :link: intro-overview.html
        :link-type: url

        A gentle introduction to ExecuTorch. In this section,
        you will learn about main features of ExecuTorch
        and how you can use them in your projects.

     .. grid-item-card:: :octicon:`file-code;1em`
        Getting started with ExecuTorch
        :img-top: _static/img/card-background.svg
        :link: getting-started-setup.html
        :link-type: url

        A step-by-step tutorial on how to get started with
        ExecuTorch.

     .. grid-item-card:: :octicon:`file-code;1em`
        ExecuTorch Llama
        :img-top: _static/img/card-background.svg
        :link: llm/llama.html
        :link-type: url

        Learn about running Llama models via ExecuTorch

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
   :caption: Getting Started
   :hidden:

   getting-started-setup
   export-overview
   runtime-build-and-cross-compilation


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/export-to-executorch-tutorial
   running-a-model-cpp-tutorial
   extension-module
   extension-tensor
   tutorials/devtools-integration-tutorial
   apple-runtime
   demo-apps-ios
   demo-apps-android
   examples-end-to-end-to-lower-model-to-delegate
   tutorial-xnnpack-delegate-lowering
   build-run-vulkan
   ..
      Alphabetical by backend name. Be sure to keep the same order in the
      customcarditem entries below.
   executorch-arm-delegate-tutorial
   build-run-coreml
   build-run-mps
   build-run-qualcomm-ai-engine-direct-backend
   build-run-xtensa

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Working with LLMs
   :hidden:

   Llama <llm/llama>
   Llama on Android <llm/llama-demo-android>
   Llama on iOS <llm/llama-demo-ios>
   Llama on Android via Qualcomm backend <llm/build-run-llama3-qualcomm-ai-engine-direct-backend>
   Intro to LLMs in ExecuTorch <llm/getting-started>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   export-to-executorch-api-reference
   executorch-runtime-api-reference
   runtime-python-api-reference
   api-life-cycle

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

   compiler-delegate-and-partitioner
   compiler-backend-dialect
   compiler-custom-compiler-passes
   compiler-memory-planning

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Runtime
   :hidden:

   runtime-overview
   runtime-backend-delegate-implementation-and-linking
   runtime-platform-abstraction-layer
   portable-cpp-programming
   pte-file-format

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
   :maxdepth: 1
   :caption: Backend Delegates
   :hidden:

   native-delegates-executorch-xnnpack-delegate
   native-delegates-executorch-vulkan-delegate
   backend-delegates-integration
   backend-delegates-dependencies
   debug-backend-delegate

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
   :link: build-run-vulkan.html
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
   :link: build-run-coreml.html
   :tags: Export,Backend,Delegation,CoreML

.. customcarditem::
   :header: Building and Running ExecuTorch with MPS Backend
   :card_description: A tutorial that walks you through the process of building ExecuTorch with MPSGraph Backend
   :image: _static/img/generic-pytorch-logo.png
   :link: build-run-mps.html
   :tags: Export,Backend,Delegation,MPS,MPSGraph

.. customcarditem::
   :header: Building and Running ExecuTorch with Qualcomm AI Engine Direct Backend
   :card_description: A tutorial that walks you through the process of building ExecuTorch with Qualcomm AI Engine Direct Backend
   :image: _static/img/generic-pytorch-logo.png
   :link: build-run-qualcomm-ai-engine-direct-backend.html
   :tags: Export,Backend,Delegation,QNN

.. customcarditem::
   :header: Building and Running ExecuTorch on Xtensa HiFi4 DSP
   :card_description: A tutorial that walks you through the process of building ExecuTorch for an Xtensa Hifi4 DSP using custom operators
   :image: _static/img/generic-pytorch-logo.png
   :link: build-run-xtensa.html
   :tags: Export,Custom-Operators,DSP,Xtensa

.. customcardend::
