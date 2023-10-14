# Quantization Overview
Quantization is a process that reduces the precision of computations and lowers memory footprint in the model. To learn more, please visit the [ExecuTorch concepts page](./concepts.md#quantization). This is particularly useful for edge devices, which typically have limited resources such as processing power, memory, and battery life. By using quantization, we can make our models more efficient and enable them to run effectively on these devices.

In terms of flow, quantization happens early in the ExecuTorch stack:

![ExecuTorch Entry Points](/_static/img/executorch-entry-points.png).

A more detailed workflow can be found in the [ExecuTorch tutorial](./tutorials/export-to-executorch-tutorial).

Quantization is usually tied to execution backends that have quantized operators implemented. Thus each backend is opinionated about how the model should be quantized, expressed in a backend specific ``Quantizer`` class. ``Quantizer`` provides API for modeling users in terms of how they want their model to be quantized and also passes on the user intention to quantization workflow.

Backend developers will need to implement their own ``Quantizer`` to express how different operators or operator patterns are quantized in their backend. This is accomplished via [Annotation API](https://pytorch.org/tutorials/prototype/pt2e_quantizer.html) provided by quantization workflow. Since Quantizer is also user facing, it will expose specific APIs for modeling users to configure how they want the model to be quantized. Each backend should provide their own API documentation for their ``Quantizer``.

Modeling user will use the ``Quantizer`` specific to their target backend to quantize their model, e.g. ``XNNPACKQuantizer``.

For an example quantization flow with ``XNPACKQuantizer``, more docuemntations and tutorials, please see ``Performing Quantization`` section in [ExecuTorch tutorial](./tutorials/export-to-executorch-tutorial).
