(home)=
# Welcome to the ExecuTorch Documentation

**ExecuTorch** is PyTorch's open source export and runtime stack for running AI
locally on phones, wearables, laptops, browsers, embedded systems, and
microcontrollers.

```{note}
Match the documentation version to your installed ExecuTorch release using the
version selector. Examples on the `main` site may require a nightly package or
a source checkout.
```

## Start here

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Run your first model
:link: getting-started
:link-type: doc

Install ExecuTorch, export MobileNet V2 to XNNPACK, and execute the resulting
`.pte` program on the host.

+++
**Open the tutorial →**
:::

:::{grid-item-card} Choose a path
:link: user-pathways
:link-type: doc

Route to the right guide by model, workload, target platform, experience level,
or role.

+++
**Open the decision guide →**
:::

::::

## Why ExecuTorch

- **PyTorch-native deployment:** Capture models with `torch.export`, lower them
  for a chosen target, and retain PyTorch program metadata for debugging.
- **Target-specific acceleration:** Delegate supported graph regions to CPU,
  GPU, NPU, and DSP backends; unpartitioned regions run with the kernels
  included in the runtime.
- **A portable, right-sized runtime:** Integrate `.pte` programs through C++,
  Python, Java/Kotlin, Objective-C/Swift, or JavaScript, and include only the
  operators and backends the application needs.

---

## Proven on real products

::::{grid} 1 1 3 3
:gutter: 2
:class-container: success-showcase

:::{grid-item-card} **Billions of people**
:class-header: bg-primary text-white
:class-body: text-center

Production features across Instagram, WhatsApp, Messenger, and Facebook run
on-device with ExecuTorch.

[Read the Meta engineering story →](https://engineering.fb.com/2025/07/28/android/executorch-on-device-ml-meta-family-of-apps/)
:::

:::{grid-item-card} **Shipping voice transcription**
:class-header: bg-primary text-white
:class-body: text-center

LM Studio uses ExecuTorch and Parakeet TDT for local transcription on macOS and
Windows.

[Read the production case study →](https://pytorch.org/blog/building-voice-agents-with-executorch-a-cross-platform-foundation-for-on-device-audio/)
:::

:::{grid-item-card} **Up to 2× CPU throughput**
:class-header: bg-primary text-white
:class-body: text-center

Liquid AI reports the gain against selected similarly sized models, plus lower
memory use, on its tested laptop and mobile CPUs.

[Read Liquid AI's case study →](https://www.liquid.ai/blog/how-liquid-ai-uses-executorch-to-power-efficient-flexible-on-device-intelligence)
:::
::::

{doc}`Explore all deployments, integrations, and showcases → <success-stories>`

---

## Browse documentation

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Core concepts
:link: intro-section
:link-type: doc

Architecture, export and runtime concepts, and the `.pte` program format.
:::

:::{grid-item-card} Export a model
:link: using-executorch-export
:link-type: doc

Capture, lower, quantize, and validate a PyTorch model for a chosen target.
:::

:::{grid-item-card} Advanced optimization
:link: advanced-topics-section
:link-type: doc

Quantization, memory planning, custom operators, passes, and backends.
:::

:::{grid-item-card} Deploy by platform
:link: edge-platforms-section
:link-type: doc

Android, iOS, desktop, and embedded integration guides.
:::

:::{grid-item-card} Choose a backend
:link: backends-section
:link-type: doc

Compare CPU, GPU, NPU, and DSP acceleration paths for target hardware.
:::

:::{grid-item-card} Work with LLMs
:link: llm/working-with-llms
:link-type: doc

Export, optimize, and deploy text and multimodal generation models.
:::

:::{grid-item-card} Runtime and LLM APIs
:link: api-section
:link-type: doc

Find C++, Python, Java/Kotlin, Objective-C/Swift, and JavaScript APIs.
:::

:::{grid-item-card} Optimize and debug
:link: tools-section
:link-type: doc

Profile execution and inspect programs with ETDump, ETRecord, and numeric
debugging.
:::

:::{grid-item-card} Get help and contribute
:link: support-section
:link-type: doc

Troubleshooting, FAQs, issue reporting, and contribution guidance.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

intro-section
quick-start-section
user-pathways
success-stories
edge-platforms-section
backends-section
llm/working-with-llms
advanced-topics-section
tools-section
api-section
support-section
```
