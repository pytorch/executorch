.. _home:

Welcome to the ExecuTorch Documentation
=======================================

.. important::
   This is a preview version of ExecuTorch and should be used for testing and
   evaluation purposes only. It is not recommended for use in production
   settings. We welcome any feedback, suggestions, and bug reports from the
   community to help us improve the technology. Please use the
   `PyTorch Forums <https://discuss.pytorch.org/>`__ for discussion and feedback
   about ExecuTorch using the tag **#executorch** and our
   `GitHub repository <https://github.com/pytorch/executorch/issues>`__ for bug
   reporting.

**ExecuTorch** is a PyTorch platform that provides infrastructure to run
PyTorch programs everywhere from AR/VR wearables to standard on-device
iOS and Android mobile deployments. One of the main
goals for ExecuTorch is to enable wider customization and deployment
capabilities of the PyTorch programs.

ExecuTorch heavily relies on such PyTorch technologies as `torch.compile
<https://pytorch.org/docs/stable/torch.compiler.html>`__ and `torch.export
<https://pytorch.org/docs/main/export.html>`__. If you are not familiar with
these APIs, you might want to read about them in the PyTorch documentation
before diving into the ExecuTorch documentation.

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
        ExecuTorch Intermediate Representation API
        :img-top: _static/img/card-background.svg
        :link: ir-exir.html
        :link-type: url

        Learn about EXIR, a graph-based intermediate
        representation (IR) of PyTorch programs.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Introduction
   :hidden:

   intro-overview
   concepts
   intro-how-it-works

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting-started-architecture
   getting-started-setup
   runtime-build-and-cross-compilation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/export-to-executorch-tutorial
   running-a-model-cpp-tutorial
   examples-end-to-end-to-lower-model-to-delegate
   demo-apps-ios
   demo-apps-android
   build-run-xtensa
   build-run-qualcomm-ai-engine-direct-backend
   build-run-mps
   tutorials/sdk-integration-tutorial
   executorch-arm-delegate-tutorial
   tutorial-xnnpack-delegate-lowering

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Exporting to ExecuTorch
   :hidden:

   export-overview

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   export-to-executorch-api-reference
   executorch-runtime-api-reference

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
   runtime-custom-memory-allocator
   runtime-error-handling
   runtime-platform-abstraction-layer

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
   kernel-library-selective_build

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Native Delegates
   :hidden:

   native-delegates-executorch-xnnpack-delegate

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: SDK
   :hidden:

   sdk-overview
   sdk-bundled-io
   sdk-etrecord
   sdk-etdump
   sdk-profiling
   sdk-inspector
   sdk-delegate-integration
   sdk-tutorial

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
   :header: Using the ExecuTorch SDK to profile a model
   :card_description: A tutorial for using the ExecuTorch SDK to profile and analyze a model with linkage back to source code.
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/sdk-integration-tutorial.html
   :tags: SDK

.. customcarditem::
   :header: Running an ExecuTorch Model C++ Tutorial
   :card_description: A tutorial for setting up memory pools, loading a model, setting inputs, executing the model, and retrieving outputs on device.
   :image: _static/img/generic-pytorch-logo.png
   :link: running-a-model-cpp-tutorial.html
   :tags:

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
   :header: XNNPACK Backend Delegate Lowering Tutorial
   :card_description: A demo tutorial for lowering and export models with the XNNPACK Backend
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorial-xnnpack-delegate-lowering.html
   :tags: Export,Delegation,Quantization,XNNPACK

.. customcarditem::
   :header: Building and Running ExecuTorch on Xtensa HiFi4 DSP
   :card_description: A tutorial that walks you through the process of building ExecuTorch for an Xtensa Hifi4 DSP.
   :image: _static/img/generic-pytorch-logo.png
   :link: build-run-xtensa.html
   :tags: DSP

.. customcardend::
