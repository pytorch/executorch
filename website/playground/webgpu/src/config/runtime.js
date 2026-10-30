// Files hosted in the ExecuTorch WebGPU Runtime repo on Hugging Face, pinned to one
// commit and checked against their SHA-256: the runtime's .wasm
// (runtime/executorch_wasm_lib.js is the loader from the same build; see
// runtime/BUILD_INFO.md) and the tokenizer library.
export const RUNTIME_WASM = Object.freeze({
  url: 'https://huggingface.co/executorch-community/ExecuTorch-WebGPU-Runtime/resolve/48a27434e9f3096f100d79a5d1e8437a124ca654/executorch_wasm_lib.wasm',
  sha256: '35a9087d0b6cd86c279e7111adfd54a58156bd993928b20f63348f1043652eb4',
  bytes: 6562188,
});

// @huggingface/transformers 3.0.2, unmodified from npm (Apache-2.0); used for
// tokenizers and chat templates only.
export const TOKENIZER_LIB = Object.freeze({
  url: 'https://huggingface.co/executorch-community/ExecuTorch-WebGPU-Runtime/resolve/48a27434e9f3096f100d79a5d1e8437a124ca654/vendor/transformers/transformers.min.js',
  sha256: '495b03b846b315baab6ff2e2f048085e45684b910799e36c4d7f2b4be00cff72',
  bytes: 748496,
});
