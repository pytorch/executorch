## Summary
LLaVA is the first multi-modal LLM ExecuTorch supports. In this directory, we
- Host a model definition for [LLavA](https://github.com/haotian-liu/LLaVA).
- Demonstrate how to export LLavA multimodal model to generate ExecuTorch .PTE file.
- Provide a C++ runner, Android/iOS Apps that loads the .pte file, the tokenizer and an image, then generate responses based on user prompt.
- Discuss optimizations went into enabling LlaVA on a phone, and early performance numbers

Tokenizer, image encoder, and the pretrained text model, which is based on Meta
[Llama2-7b](https://llama.meta.com/llama2/), is loaded from Llava
huggingface page [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) .

## What is LLaVA?

[LLaVA](https://llava-vl.github.io/) is a novel end-to-end trained large
multimodal model that combines a vision encoder and Vicuna (a LLama2 based text
model) for general-purpose visual and language understanding, achieving
impressive chat capabilities mimicking spirits of the cutting edge multimodal
models and setting a high bar for accuracy on Science QA.

## Instructions

First you need to generate a .PTE file for the model, along with input image,
and other artifacts. Then you need either a C++ runner, or Android or iOS
application to test things out on device.

### Generate ExecuTorch .PTE and other artifacts

Run the following command to generate `llava.pte`, `tokenizer.bin` and an image
tensor (serialized in TorchScript) `image.pt`.

Prerequisite: run `install_requirements.sh` to install ExecuTorch and run
`examples/models/llava/install_requirements.sh` to install dependencies.

```bash
python -m executorch.examples.models.llava.export_llava --pte-name llava.pte --with-artifacts
```

Currently the whole export process takes about 6 minutes. We also provide a
small test utility to verify the correctness of the exported .pte file. Just run:

```bash
python -m executorch.examples.models.llava.test.test_pte llava.pte
```

### Build C++ Runner

See or run `.ci/scripts/test_llava.sh` shell script to build a C++ runner. This
script also has preliminary support to build the C++ runner for Android.

This also has an image utility Python script to generate image in PyTorch
loadable format. Alternatively, we are working on generating image format which
doesn't need PyTorch to load an image. Motivation for this is to build the C++
runner on Android.

Then you should be able to find `llava_main` binary:

```bash
cmake-out/examples/models/llava/llava_main
```

### Build Mobile Apps

#### Android

We can run LLAVA using the LLAMA Demo Apps. Please refer to [this
tutorial](https://github.com/pytorch/executorch/tree/main/examples/demo-apps/android/LlamaDemo)
to for full instructions on building the Android LLAMA Demo App.

#### iOS

We can run LLAVA using the LLAMA Demo Apps. Please refer to [this
tutorial](https://github.com/pytorch/executorch/tree/main/examples/demo-apps/apple_ios/LLaMA)
to for full instructions on building the iOS LLAMA Demo App.

### Running LLaVA

Run:
```bash
cmake-out/examples/models/llava/llava_main \
    --model_path=llava.pte                 \
    --tokenizer_path=tokenizer.bin         \
    --image_path=image.pt                  \
    --prompt="ASSISTANT:" \
    --seq_len=768                          \
    --temperature=0
```
(see --help for other options).

For this example input used in this example,

![image](https://upload.wikimedia.org/wikipedia/commons/3/3e/Chicago_Bulls_-_New_Jersey_Nets_match_on_March_28%2C_1991.jpg)

You should get a response like (tested on Arm CPUs with ET XNNPACK delegate):

```
ASSISTANT: image captures a basketball game in progress, with several players on the court. ...
```

## Optimizations and Results

Since LLaVA model needs at least 4-bit quantization to fit even within some of
the high-end phones, results presented here correspond to 4-bit groupwise
post-training quantized model.

In addition to that, work is mainly focused on using Arm CPUs and ET XNNPACK delegate.

### Memory Footprint Reduction Techniques

With Llava, we needed to find a way to reduce the memory footprint in order to
make it feasible to run on edge devices. Out of the box, even with 4-bit
quantized weights, the memory footprint is around ~11 GiB, which is
prohibitively large even for high-end Android or iOS devices.

We did several optimizations, which should be already enabled if you follow this
tutorial, to get the memory footprint down to ~5 GiB, which unblocks us to run
on high-end devices.

#### Sharing intermediate memory across delegates

Sharing working memory across ET XNNPACK delegates helps reduce the peak memory
usage for LLMs with many DQLinears. We reduced it by 36.1% (from 10.44GiB to
6.67GiB) for Llava towards unblocking it to run on Phones.

#### Reducing maximum sequence length

To free up more memory, we examined non-constant memory usage, specifically
focusing on intermediate tensors used throughout the model during inference.
The majority of these were found in the KV-cache allocations. Based on “minimum
can get away with” heuristic, we reduced max sequence length number to 768 from
previous default 2048. This adjustment led to a further memory reduction of
approximately 1.23 GiB (from 6.67 GiB to 5.44 GiB).

#### Quantizing embedding weights to 8b

By quantizing the embedding layer to 8 bit, we were able to achieve an
additional memory footprint reduction of approximately 300 MiB, bringing the
total down to ~5 GiB.

### Performance Optimizations

#### Decode performance

This was already heavily optimized through KV-cache and GEMV kernel
optimization efforts for LLama2/3.

#### Encode performance

With image based large prompts, this was the focus of performance
optimizations for LLaVA. We implemented two main optimizations to bring the decode or
prefill performance for the image down by more than 100% from the baseline.

* **Two XNNPACK Partitioners**

For text-only LLMs, our approach involved lowering only DQLinear ops
to XNNPACK and relying on ExecuTorch-optimized operators or custom ops
(utilizing Neon SIMD) to support multiplication, addition, and other
operations. Lowering these operations to XNNPACK significantly improves Time to
First Token (TTFT).


* **New Arm Neon I8mm GEMM kernels**

We introduced new kernels in XNNPACK for the quantization scheme used
here, which upgrades our existing dot-prod based GEMM kernels to i8mm based
GEMM kernels. The new kernel offers significantly improved performance by
leveraging the more efficient SMMLA instruction from Arm Neon. However, it's
worth noting that this instruction is only available on newer Arm CPUs.


### Results

Note this is an active area of development in the ExecuTorch repository. You
will need this PR [5380](https://github.com/pytorch/executorch/pull/5380) to
supply an image to the C++ runner on Android without Torch dependency. This
should be merged soon.

With those caveats out of the way, here are some preliminary numbers (as average of
three runs) for LLaVA using a C++ runner on Android OnePlus12 device with 12GiB
memory.

| Experiment Setup  | Prefill time in seconds | Decode tokens/second |
| :------------- | -------------: | -------------: |
| Baseline  | 29.95  | 8.75 |
| + Two XNNPACK Partitioners  | 17.82  | 8.93 |
| + New Arm Neon i8mm GEMM Kernels  | 14.60 | 8.92 |

We appreciate your feedback. Please let us know if you run into any issues.
