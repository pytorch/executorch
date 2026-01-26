# Benchmarking Infrastructure (Experimental)

The ExecuTorch project introduces an advanced benchmarking infrastructure designed to measure the performance of models on Android and iOS devices. It supports various backend delegates and devices, enabling reproducible performance measurements and facilitating collaborative efforts in performance tuning and debugging. This infrastructure is built on top of the [Nova reusable mobile workflow](https://github.com/pytorch/test-infra/wiki/Testing-Android-and-iOS-apps-on-OSS-CI-using-Nova-reusable-mobile-workflow) powered by PyTorch test-infra.

### Key Features

- **Multiple Models**: Supports a variety of ExecuTorch-enabled models such as `MobileNetV2` etc. Integration with compatible Hugging Face models is coming soon.

- **Device Support**: Includes popular phones like latest Apple iPhone, Google Pixel, and Samsung Galaxy, etc.

- **Backend Delegates**: Supports XNNPACK, Apple CoreML and MPS, Qualcomm QNN, and more in the near future.

- **Benchmark Apps:** Generic apps that support both GenAI and non-GenAI models, capable of measuring performance offline. [Android App](android/benchmark/) | [iOS App](apple/Benchmark/). Popular Android and iOS profilers with in-depth performance analysis will be integrated with these apps in the future.

- **Performance Monitoring**: Stores results in a database with a dashboard for tracking performance and detecting regressions.

> **Disclaimer:** The infrastructure is new and experimental. We're working on improving its accessibility and stability over time.


## Dashboard

The ExecuTorch Benchmark Dashboard tracks performance metrics for various models across different backend delegates and devices. It enables users to compare metrics, monitor trends, and identify optimizations or regressions in Executorch. The dashboard is accessible at **[ExecuTorch Benchmark Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fexecutorch)**. <!-- @lint-ignore -->

**Comprehensive Comparisons**:
- Analyze performance differences between backend delegates (e.g., XNNPACK, CoreML, QNN, MPS) for the same model.
- Compare performance across different models.
- Track performance changes over time and across different commits.

**Metrics Tracking**:
- Monitor essential metrics such as load time and inference time. For LLMs, additional metrics like tokens/s are available.
- Observe performance trends over time to identify improvements or regressions.

**Visualizations**:
- View detailed performance data through charts and graphs.
- Color-coded highlights for improvements (green) and regressions (red) exceeding 5% compared to the baseline.


## Supported Use Cases

The benchmarking infrastructure currently supports two major use-cases:

- **On-Demand Model Benchmarking:** Users can trigger benchmarking requests via GitHub Actions workflow dispatch UI. This feature will help backend developers collaborate with the ExecuTorch team to debug performance issues and advance state-of-the-art (SOTA) performance.

- **Automated Nightly Batched Benchmarking:** The infrastructure performs automated nightly benchmarking to track and monitor performance over time. This allows for consistent performance monitoring and regression detection.


## High-Level Diagram

![Benchmarking Infrastructure](../../docs/source/_static/img/benchmark-infra.png)


## Scheduling On-Demand Benchmarking

Users can schedule a benchmarking workflow on a pull request through GitHub Actions using the workflow dispatch UI. Follow the steps below to trigger benchmarking:
1. Access `pytorch/executorch` repository on GitHub and navigate to the "Actions" tab.
2. Select `android-perf` or `apple-perf` workflow from the list of workflows.
3. Click "Run workflow" and fill in the required parameters for the model you want to benchmark, e.g. branch name, model name and delegate, and device pool, etc.

> **Note:** Write permission to the repo will be needed in order to run the on-demand workflow.


## Retrieving Benchmark Results

The easiest way to view benchmark results is on the [dashboard](README.md#dashboard), while raw results for individual configurations can be manually accessed by downloading the `Customer_Artifacts.zip` from the CI.


## Feedback and Issue Reporting
We encourage users to share feedback or report any issues while using the infra. Please submit your feedback via GitHub Issues.
