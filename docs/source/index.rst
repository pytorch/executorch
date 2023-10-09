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

ExecuTorch heavily relies on such PyTorch technologies as TorchDynamo
and torch.export. If you are not familiar with these APIs, you might want
to read about them in the PyTorch documentation before diving into
the ExecuTorch documentation.

Features described in this documentation are classified by release status:

  *Stable:*  These features will be maintained long-term and there should
  generally be no major performance limitations or gaps in documentation.
  We also expect to maintain backwards compatibility (although
  breaking changes can happen and notice will be given one release ahead
  of time).

  *Beta:*  These features are tagged as Beta because the API may change based on
  user feedback, because the performance needs to improve, or because
  coverage across operators is not yet complete. For Beta features, we are
  committing to seeing the feature through to the Stable classification.
  We are not, however, committing to backwards compatibility.

  *Prototype:*  These features are typically not available as part of
  binary distributions like PyPI or Conda, except sometimes behind run-time
  flags, and are at an early stage for feedback and testing.

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
   ir-high-order-operators

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Compiler Entry Points
   :hidden:

   examples-end-to-end-to-lower-model-to-delegate
   compiler-delegate-and-partitioner
   compiler-kernel-fusion-pass
   compiler-custom-compiler-passes
   compiler-memory-planning

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Runtime
   :hidden:

   runtime-overview
   runtime-build-and-cross-compilation
   runtime-backend-delegate-implementation-and-linking
   runtime-custom-memory-allocator
   runtime-error-handling
   runtime-platform-abstraction-layer

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quantization
   :hidden:

   quantization-custom-quantization

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Kernel Library
   :hidden:

   kernel-library-overview
   kernel-library-custom-aten-kernel

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: SDK
   :hidden:

   sdk-overview
   sdk-etrecord
   sdk-etdump
   sdk-profiling
   sdk-inspector
   sdk-bundled-io
   sdk-delegate-integration

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/export-to-executorch-tutorial
   running-a-model-cpp-tutorial
   build-run-xtensa
   tutorials/sdk-integration-tutorial

Tutorials and Examples
~~~~~~~~~~~~~~~~~~~~~~

Ready to experiment? Check out some of the
ExecuTorch tutorials.

.. customcardstart::

.. customcarditem::
   :header: Exporting to ExecuTorch Tutorial
   :card_description: A tutorial for exporting a model and lowering a it to be runnable on the ExecuTorch runtime.
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/export-to-executorch.html
   :tags: Export,Delegation,Quantization

.. customcarditem::
   :header: Integrating with ExecuTorch SDK Tutorial
   :card_description: A tutorial for integrating a model with runtime results and utilizing it on the ExecuTorch SDK.
   :image: _static/img/generic-pytorch-logo.png
   :link: tutorials/sdk-integration.html
   :tags: Export

.. customcarditem::
   :header: Running an ExecuTorch Model C++ Tutorial
   :card_description: A tutorial for setting up memory pools, loading a model, setting inputs, executing the model, and retrieving outputs on device.
   :image: _static/img/generic-pytorch-logo.png
   :link: running-a-model-cpp.html
   :tags:

.. customcardend::
