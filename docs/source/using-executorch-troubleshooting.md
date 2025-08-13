# Profiling and Debugging

To faciliate model and runtime integration, ExecuTorch provides tools to profile model resource utilization, numerics, and more. This section describes the available troubleshooting tools and steps to resolve issues when integrating ExecuTorch.

## General Troubleshooting Steps

- To troubleshoot failure of runtime API calls, such as loading or running a model, ensure that ExecuTorch framework logging is enabled. See [Logging](using-executorch-runtime-integration.md#logging) for more information.
- As a prelimatinary step to troubleshoot slow run times, ensure that performance testing is being done in a release build, and that the model is delegated. See [Inference is Slow](using-executorch-faqs.md#inference-is-slow--performance-troubleshooting) for more information.
- Check [Frequently Asked Questions](using-executorch-faqs.md) for common issues and questions encountered during install, model export, and runtime integration.

## Developer Tools

The ExecuTorch developer tools, or devtools, are a collection of tooling for troubleshooting model performance, numerics, and resource utilization. See [Introduction to the ExecuTorch Developer Tools](devtools-overview.md) for an overview of the available developer tools and usage.

## Next Steps

- [Frequently Asked Questions](using-executorch-faqs.md) for solutions to commonly encountered questions and issues.
- [Introduction to the ExecuTorch Developer Tools](runtime-profiling.md) for a high-level introduction to available developer tooling.
- [Using the ExecuTorch Developer Tools to Profile a Model](https://pytorch.org/executorch/main/tutorials/devtools-integration-tutorial) for information on runtime performance profiling.
- [Inspector APIs](runtime-profiling.md) for reference material on trace inspector APIs.
