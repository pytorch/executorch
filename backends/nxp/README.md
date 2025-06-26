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


## Backend Status and Maturity
**Current Status:** Prototype Quality

The eIQ Neutron NPU Backend should be considered as prototype quality at this moment. Subject to significant changes and
improvements. NXP and the ExecuTorch community is actively developing this codebase.

## Neutron Backend implementation and SW architecture
Neutron Backend uses the eIQ Neutron Converter as ML compiler to compile the delegated subgraph to Neutron microcode. 
The Neutron Converter accepts the ML model in LiteRT format, for the **eIQ Neutron N3** class  therefore the Neutron Backend uses the LiteRT flatbuffers format as IR between the ExecuTorch and Neutron Converter ML compiler. 

The Neutron Backend in its early prototype phase, is based on existing NXP products, such as 
onnx2tflite, known from the NXP's eIQ Toolkit. 
The **onnx2tflite** is a converter from the ONNX format to LiteRT (formerly known as TFLite).
It consists of 3 stages: 
* ONNX Model Parsing
* Tensor Format Inference, to identify tensors using channel-first layer
* ONNX to LiteRT Conversion 
* Optimization Passes, which operate on top of the LiteRT format
* LiteRT Serialization 

Due to the similarities between ONNX to LiteRT and Edge to LiteRT conversion, the Neutron Backend's 
currently leverages the Tensor format Inference and LiteRT Optimizer. 
This shall be considered as temporary solution, intended to be replaced with: 
* Dim Order (https://github.com/pytorch/executorch/issues/4873)
* Corresponding ExecuTorch/ATen passes

before reaching higher maturity status by the end of 2025. 

## Layout
The current code base is as follows:
* `backend/ir/` - TFLite/LiteRT based IR to represent the Edge Subgraph, taken from onnx2tflite code base and extended to
  support Edge Dialect to LiteRT conversion.
    * `backend/ir/converter` - Neutron Backends conversion from Edge (ATen) Dialect to LiteRT, TFLite. The subfolder
      `node_conveters` is structured as single module for each Edge operator.
    * `backend/ir/lib` - automatically generated handlers from LiteRT flatbuffers schema
    * `backend/ir/tflite_generator` and `backend/ir/tflite_optimizer` handle the serialization
       of the in-memory built subgraph for delegation into LiteRT/TFLite flatbuffers 
       representation. Code taken from the onnx2tflite tool.
*  `quantizer` - Neutron Backends quantizer implementation. 

## Help & Improvements
If you have problems or questions or have suggestions for ways to make
implementation and testing better, please reach out to the NXP representative for the SoC you are interested in using,
or your distribution partner contact.

Or raise the issue here on ExecuTorch GitHub, label it with `module: nxp` and our ML team will address it on a priority-basis.
