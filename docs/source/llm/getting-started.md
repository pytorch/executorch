# Getting Started with LLMs via ExecuTorch

This section provides guidance on enabling Large Language Models (LLMs), starting with a simple example and gradually introducing new concepts to improve performance and productivity.

## Prerequisites

- To run this tutorial, you’ll first need to first [Set up your ExecuTorch environment](../getting-started-setup.md).

- We highly suggest you to check out [LLama2 README](../../../examples/models/llama2/README.md) in our examples for end-to-end Llama2 mobile demo.


## Simple “Hello World” LLM example

Let's create a simple LLM app from scratch. TODO

## Quantization

Most LLMs are too large to fit into a mobile phone, making quantization necessary. In this example, we will demonstrate how to use the XNNPACKQuantizer to quantize the model and run it on a CPU. TODO

## Use Mobile Acceleration

One of the benefits of ExecuTorch is the ability to delegate to mobile accelerators. Now, we will show a few examples of how to easily take advantage of mobile accelerators. TODO

## Debugging and Profiling

It is sometimes necessary to profile and inspect the execution process. In this example, we will demonstrate how the ExecuTorch SDK can be used to identify which operations are being executed on which hardware.  TODO

## How to use custom kernels

In some cases, it is necessary to write custom kernels or import them from another source in order to achieve the desired performance. In this example, we will demonstrate how to use the `kvcache_with_sdpa` kernel.

## How to build Mobile Apps

Here's how to finally build a mobile app on Android and iOS. TODO
