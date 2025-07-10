# Deploying LLMs to Executorch

ExecuTorch is designed to support all types of machine learning models, and LLMs are no exception.
In this section we demonstrate how to leverage ExecuTorch to performantly run state of the art
LLMs on-device out of the box with our provided export LLM APIs, acceleration backends, quantization
libraries, tokenizers, and more.

We encourage users to use this project as a starting point and adapt it to their specific needs,
which includes creating your own versions of the tokenizer, sampler, acceleration backends, and
other components. We hope this project serves as a useful guide in your journey with LLMs and ExecuTorch.


## Prerequisites

To follow this guide, you'll need to install ExecuTorch. Please see [Setting Up ExecuTorch](../getting-started.md#installation).

## Next steps

- [Exporting popular LLMs out of the box](export-llm.md)
- [Exporting custom LLMs](export-custom-llm.md)
- [Running with C++](run-with-c-plus-plus.md)
- [Running on Android (XNNPack)](llama-demo-android.md)
- [Running on Android (Qualcomm)](build-run-llama3-qualcomm-ai-engine-direct-backend.md)
- [Running on iOS](llama-demo-ios.md)
