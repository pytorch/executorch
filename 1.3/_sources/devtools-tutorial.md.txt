## Developer Tools Usage Tutorials

The ExecuTorch Developer Tools provide capabilities for profiling and debugging your models. We provide step-by-step tutorials for common workflows:

### Profiling Tutorial

Please refer to the [Profiling Tutorial](tutorials/devtools-integration-tutorial) <!-- @lint-ignore --> for a walkthrough on how to profile a model in ExecuTorch using the Developer Tools. This tutorial covers:

- Generating ETRecord and ETDump artifacts
- Using the Inspector API to analyze performance data
- Identifying slow operators and bottlenecks

### Debugging Tutorial

Please refer to the [Debugging Tutorial](tutorials/devtools-debugging-tutorial) <!-- @lint-ignore --> for a walkthrough on how to debug numerical discrepancies in ExecuTorch models. This tutorial covers:

- Capturing intermediate outputs with debug buffers
- Using ``calculate_numeric_gap`` to identify precision issues
- Debugging delegated models (e.g., XNNPACK)
- Comparing runtime outputs with eager model references
