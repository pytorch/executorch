# {BACKEND_NAME} Quantization

Document quantization schemes and flows for the backend. This should include a description of each scheme and a code example to perform quantization. Example sections for PT2E and quantize_ are included below, to be replaced with details for the target backend.

For each supported quantization scheme, include the following:
 * What is the quantization scheme?
   * How are weights quantized?
   * How are activations quantized? Static or dynamic?
   * How many bits?
   * What is the granularity? Per-tensor, per-channel, group/block-wise?
 * What are the steps to quantize a model with this scheme?
 * Include a code sample.
 * If the quantization flow only supports a small set of operators - for example, linear only - note this.

### Supported Quantization Schemes
The {BACKEND_NAME} delegate supports the following quantization schemes:

- {QUANTIZATION_SCHEME_1}
- {QUANTIZATION_SCHEME_2}

### {QUANTIZATION_METHOD_1} using the PT2E Flow

[Description]

[Code Sample]

### LLM Quantization with quantize_

[Description]

[Code Sample]
