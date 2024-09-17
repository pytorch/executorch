# ExecuTorch Overview

**ExecuTorch** is an end-to-end solution for enabling on-device inference
capabilities across mobile and edge devices including wearables, embedded
devices and microcontrollers. It is part of the PyTorch Edge ecosystem and
enables efficient deployment of PyTorch models to edge devices.

Key value propositions of ExecuTorch are:

- **Portability:** Compatibility with a wide variety of computing platforms,
  from high-end mobile phones to highly constrained embedded systems and
  microcontrollers.
- **Productivity:** Enabling developers to use the same toolchains and Developer
  Tools from PyTorch model authoring and conversion, to debugging and deployment
  to a wide variety of platforms.
- **Performance:** Providing end users with a seamless and high-performance
  experience due to a lightweight runtime and utilizing full hardware
  capabilities such as CPUs, NPUs, and DSPs.

## Why ExecuTorch?

Supporting on-device AI presents unique challenges with diverse hardware,
critical power requirements, low/no internet connectivity, and realtime
processing needs. These constraints have historically prevented or slowed down
the creation of scalable and performant on-device AI solutions. We designed
ExecuTorch, backed by our industry partners like Meta, Arm, Apple, and Qualcomm,
to be highly portable and provide superior developer productivity without losing
on performance.

## How is ExecuTorch Different from PyTorch Mobile (Lite Interpreter)?

PyTorch Mobile uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to
allow PyTorch models to run on devices with limited resources. ExecuTorch has a
significantly smaller memory size and a dynamic memory footprint resulting in
superior performance and portability compared to PyTorch Mobile. Also, ExecuTorch
does not rely on TorchScript, and instead leverages PyTorch 2 compiler and export
functionality for on-device execution of PyTorch models.

Read more in-depth technical overview topics about ExecuTorch:

- [How ExecuTorch Works](intro-how-it-works.md)
- [High-level Architecture and Components of ExecuTorch](getting-started-architecture.md)
- [ExecuTorch Runtime Overview](runtime-overview.md)
