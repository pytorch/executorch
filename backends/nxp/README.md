# NXP eIQ Neutron Backend

This subtree contains the ExecuTorch Backend implementation for the
[eIQ® Neutron Neural Processing Unit (NPU)](https://www.nxp.com/applications/technologies/ai-and-machine-learning/eiq-neutron-npu:EIQ-NEUTRON-NPU).

The eIQ® Neutron NPU is a highly scalable accelerator core architecture providing machine learning (ML) acceleration,
able to support common and critical tasks for edge AI such as anomaly detection, speech recognition,
image classification, object detection, facial recognition, image segmentation, and generative AI use cases like 
large and small language models (LLMs & SLMs) and text-to-speech (TTS).
The architecture provides power and performance optimized NPUs integrated with NXP's broad portfolio of 
microcontrollers and applications processors.

The eIQ Neutron NPUs offer support for a wide variety of neural network types such as CNN, RNN, TCN and Transformer 
networks, as well as the ability to adapt and scale to new model architectures, topologies and layer types introduced
to AI workloads. ML application development with the eIQ Neutron NPU is fully supported by the 
[eIQ machine learning software development environment](https://www.nxp.com/design/design-center/software/eiq-ml-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT).
The eIQ AI SW Stack provides a streamlined development experience for developers and end-users of NXP products.
eIQ extensions connect broader AI ecosystems to the edge, such as the NVIDIA TAO extension, which enables developers to bring AI models trained and fine-tuned with TAO to NXP-powered edge devices.


## Supported NXP platforms
At this moment following eIQ® Neutron NPU variants and NXP platforms are supported by the NXP eIQ Neutron Backend:

* **eIQ Neutron N3-64**, available on [i.MX RT700](https://www.nxp.com/products/i.MX-RT700)

In the future the NXP eIQ Neutron Backend will be extended to support [i.MX 9 Application Processors](https://www.nxp.com/products/processors-and-microcontrollers/arm-processors/i-mx-applications-processors/i-mx-9-processors:IMX9-PROCESSORS) 
with eIQ Neutron NPU, like the [i.MX 95](https://www.nxp.com/products/iMX95).


## Layout
TBD

## Backend Status and Maturity
**Current Status:** Prototype Quality

The eIQ Neutron NPU Backend should be considered as prototype quality at this moment. Subject to significant changes and 
improvements. NXP and the ExecuTorch community is actively developing this codebase. 

## Help & Improvements
If you have problems or questions or have suggestions for ways to make
implementation and testing better, please reach out to the NXP representative for the SoC you are interested in using,
or your distribution partner contact.

Or raise the issue here on ExecuTorch GitHub, label it with `module: nxp` and our ML team will address it on a priority-basis.
