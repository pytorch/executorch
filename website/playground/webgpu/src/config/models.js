// The ONE place model checkpoints are referenced.
//
// Cleared entries point at their Hugging Face repo pinned to one commit, with a
// sha256 per file; the rest ship with `assets.baseUrl: null` and `license.status: 'todo'`.
// The resolver (src/loader/resolve.js) refuses remote weights unless BOTH the
// license is cleared AND a URL is populated, so filling in a URL alone never
// implies legal review happened. To enable a model after review, set
// `license.status: 'cleared'`, record the decision in `license.reviewNote`,
// and point `assets.baseUrl` at the approved repo.

export const LICENSE_STATUS = Object.freeze({
  TODO: 'todo',
  CLEARED: 'cleared',
  BLOCKED: 'blocked',
});

export const MODELS = [
  {
    id: 'llama-3.2-1b-instruct',
    name: 'Llama 3.2 1B Instruct',
    shortName: 'Llama 3.2 1B',
    task: 'text-generation',
    tagline: 'Larger · Built with Llama',
    approxBytes: 744e6,
    approxBytesIsEstimate: false,
    arch: { layers: 16, dModel: 2048, ffn: 8192, heads: 32, kvHeads: 8, vocab: 128256 },
    assets: {
      "baseUrl": "https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-ExecuTorch-WebGPU/resolve/f2b116fc0b746ec8ba8adb93258b9e8b439cefa7/", // @lint-ignore (base URL, files resolve under it)
      "files": [
        "llama3_2_1b_instruct_q4gsw.pte",
        "tokenizer/config.json",
        "tokenizer/generation_config.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json"
      ],
      "sha256": {
        "llama3_2_1b_instruct_q4gsw.pte": "49de6eec3a54f1f45a785152375edafec07cb0fd879a142100340fd2765c40a7",
        "tokenizer/config.json": "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb",
        "tokenizer/generation_config.json": "88effbb63300dbbc7390143fbbdd9d9fa50587b37e8bfd16c8c90d4970a74a36",
        "tokenizer/special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
        "tokenizer/tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
        "tokenizer/tokenizer_config.json": "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e"
      },
      "spec": {
        "pte": "llama3_2_1b_instruct_q4gsw.pte",
        "pteBytes": 734868736,
        "tokenizer": "tokenizer",
        "chat": "template",
        "prefillMax": 512,
        "maxCtx": 8960,
        "eos": [
          128001,
          128008,
          128009
        ]
      }
    },
    credit: { by: 'Meta', url: 'https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct', paper: 'https://arxiv.org/abs/2407.21783' },
    attribution: 'Built with Llama',
    license: {
      status: LICENSE_STATUS.CLEARED,
      reviewNote: "Approved for public redistribution (legal/OSS approval confirmed by the project owner, 2026-10-14).",
      upstream: 'Llama 3.2 Community License + Acceptable Use Policy',
      provenance:
        'Upstream: meta-llama/Llama-3.2-1B-Instruct (gated on Hugging Face). Derivatives carry attribution and naming requirements. ',
    },
  },
  {
    id: 'qwen3-0.6b',
    name: 'Qwen 3 0.6B',
    shortName: 'Qwen 3 0.6B',
    task: 'text-generation',
    tagline: 'Fastest · 100+ languages',
    approxBytes: 416e6,
    approxBytesIsEstimate: false,
    arch: { layers: 28, dModel: 1024, ffn: 3072, heads: 16, kvHeads: 8, vocab: 151936 },
    assets: {
      "baseUrl": "https://huggingface.co/executorch-community/Qwen3-0.6B-ExecuTorch-WebGPU/resolve/65fa8ec0b8996b4a0157cdb89ea766b000b1134a/", // @lint-ignore (base URL, files resolve under it)
      "files": [
        "qwen3_0_6b_q4gsw.pte",
        "tokenizer/config.json",
        "tokenizer/generation_config.json",
        "tokenizer/merges.txt",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json"
      ],
      "sha256": {
        "qwen3_0_6b_q4gsw.pte": "057af5734a6ce17cbf19598bcc0e972aed2401d1522eed41249675c1ab372ce6",
        "tokenizer/config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        "tokenizer/generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        "tokenizer/merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "tokenizer/tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer/tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "tokenizer/vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
      },
      "spec": {
        "pte": "qwen3_0_6b_q4gsw.pte",
        "pteBytes": 400532224,
        "tokenizer": "tokenizer",
        "chat": "chatml-nothink",
        "prefillMax": 512,
        "maxCtx": 8960,
        "eos": [
          151645,
          151643
        ]
      }
    },
    credit: { by: 'Qwen Team, Alibaba Cloud', url: 'https://huggingface.co/Qwen/Qwen3-0.6B', paper: 'https://arxiv.org/abs/2505.09388' },
    license: {
      status: LICENSE_STATUS.CLEARED,
      reviewNote: "Approved for public redistribution (legal/OSS approval confirmed by the project owner, 2026-10-14).",
      upstream: 'Apache-2.0',
      provenance: 'Upstream: Qwen/Qwen3-0.6B.',
    },
  },
  {
    id: 'sam2-hiera-tiny',
    name: 'SAM 2.1 Hiera Tiny',
    shortName: 'SAM 2.1',
    task: 'segmentation',
    approxBytes: 134e6,
    approxBytesIsEstimate: false,
    assets: {
      "baseUrl": "https://huggingface.co/executorch-community/SAM2-Hiera-Tiny-ExecuTorch-WebGPU/resolve/1c8d8006fa21e77a7c3be6b4cf0a1ed54c1e0469/", // @lint-ignore (base URL, files resolve under it)
      "files": [
        "prompt_box_br.bin",
        "prompt_box_tl.bin",
        "prompt_dense.bin",
        "prompt_gaussian.bin",
        "prompt_img_pos.bin",
        "prompt_not_a_point.bin",
        "prompt_point_embed_bg.bin",
        "prompt_point_embed_fg.bin",
        "sam2_decoder.pte",
        "sam2_encoder.pte"
      ],
      "sha256": {
        "prompt_box_br.bin": "a0df40a29af4a9b65d390ae46913d4170762a306fc8061ea87f1d8f4a4a1b0d5",
        "prompt_box_tl.bin": "96c208c44e95f7044bc480951be257c415cfa9c91db907e4575ae3c92d09e6b7",
        "prompt_dense.bin": "52cece57b549dc81439c1156a07a25ff02db8cb81329e8eefb68c90457a884f3",
        "prompt_gaussian.bin": "c37a6a9361886286fc3e868dd279dfa4703919e3fee65c07a07e5a75e980d84f",
        "prompt_img_pos.bin": "89dc4182ee4ac6a366dbfd0e9b825e4d7b6e88ded91b2ab06604eb5139e1cc2c",
        "prompt_not_a_point.bin": "19771c067b922d9d652e6f3dd1f6e2976c68abb8e6fd9db571255ae37687ea50",
        "prompt_point_embed_bg.bin": "91c432d10fcef8e32ce561eb4820beb6a772a61a3a937d7c2af8597061759cfa",
        "prompt_point_embed_fg.bin": "2e037dab5bee002d94bdb12d3141ab952cfe2315f0c9feb53a7c78a7e656ed33",
        "sam2_decoder.pte": "57b91034acc71be2ce1835ea546a174295daee1046e40db0743e0238ef11e278",
        "sam2_encoder.pte": "30f9fb469b779558b6b5c7c3eb995c50e1213ffbe082e31430c929ba2b3ffe20"
      },
      "spec": {
        "encoder": "sam2_encoder.pte",
        "encoderBytes": 115248512,
        "decoder": "sam2_decoder.pte",
        "decoderBytes": 16949252,
        "prompts": {
          "gaussian": "prompt_gaussian.bin",
          "pointPositive": "prompt_point_embed_fg.bin",
          "pointNegative": "prompt_point_embed_bg.bin",
          "notAPoint": "prompt_not_a_point.bin",
          "boxTopLeft": "prompt_box_tl.bin",
          "boxBottomRight": "prompt_box_br.bin",
          "imagePositional": "prompt_img_pos.bin",
          "dense": "prompt_dense.bin"
        },
        "promptTokens": 3,
        "imageSize": 512,
        "hidden": 256,
        "peDim": 128,
        "mean": [
          0.485,
          0.456,
          0.406
        ],
        "std": [
          0.229,
          0.224,
          0.225
        ]
      }
    },
    credit: { by: 'Meta FAIR', url: 'https://huggingface.co/facebook/sam2.1-hiera-tiny', paper: 'https://arxiv.org/abs/2408.00714' },
    license: {
      status: LICENSE_STATUS.CLEARED,
      reviewNote: "Approved for public redistribution (legal/OSS approval confirmed by the project owner, 2026-10-14).",
      upstream: 'Apache-2.0',
      provenance: 'Upstream: facebook/sam2.1-hiera-tiny.',
    },
  },
  {
    id: 'smolvlm-500m-instruct',
    name: 'SmolVLM 500M Instruct',
    shortName: 'SmolVLM-500M',
    task: 'vision-language',
    approxBytes: 934e6,
    approxBytesIsEstimate: false,
    assets: {
      "baseUrl": "https://huggingface.co/executorch-community/SmolVLM-500M-Instruct-ExecuTorch-WebGPU/resolve/2764d1a89e23ca1d2503e6f6e211341317594a74/", // @lint-ignore (base URL, files resolve under it)
      "files": [
        "embed_tokens.i8.bin",
        "embed_tokens.scale.f32.bin",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "vlm_decoder.pte",
        "vlm_meta.json",
        "vlm_prefill.pte",
        "vlm_vision_q8.pte"
      ],
      "sha256": {
        "embed_tokens.i8.bin": "d3ea47247be0b0703ca1246335cb3bca5f68a4068e177475388e287d9eb5f385",
        "embed_tokens.scale.f32.bin": "adefdadaae81d572c6094b6826e850f263939d716b74da03484dc20c0769562a",
        "tokenizer/special_tokens_map.json": "aa0ff906077086dfa9734a7f97f68c825877a48f9468807be65504495cdeef09",
        "tokenizer/tokenizer.json": "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a",
        "tokenizer/tokenizer_config.json": "36c6fd44d07d10fd8180ee6b46dcccf69fb7c06753968ff0d7e17b8bfe17b777",
        "vlm_decoder.pte": "d92697a8c32eeaa1d73603cf608d658cf297fdf41ed81384ec29b7025e78e996",
        "vlm_meta.json": "b36fcf7ef7da8c2ace5271aa5da5ffaa5f10a779db2095cb69f5eadfc9972ae4",
        "vlm_prefill.pte": "1e8671eb77ae2f60c1eb80073454230bfaeb1eb22cee36dd9b028d8e291a3e3f",
        "vlm_vision_q8.pte": "5f09d6028d9bb945ce135d4be9ad499c1081f13d46655d2d1288c207f536ee4e"
      },
      "spec": {
        "files": {
          "vision": {
            "path": "vlm_vision_q8.pte",
            "bytes": 108885888
          },
          "decoder": {
            "path": "vlm_decoder.pte",
            "bytes": 388664832
          },
          "prefill": {
            "path": "vlm_prefill.pte",
            "bytes": 388664320
          },
          "embed": {
            "path": "embed_tokens.i8.bin",
            "bytes": 47308800
          },
          "embedScale": {
            "path": "embed_tokens.scale.f32.bin",
            "bytes": 197120
          }
        },
        "meta": "vlm_meta.json",
        "tokenizer": "tokenizer",
        "prefillMax": 83,
        "maxCtx": 512
      }
    },
    credit: { by: 'Hugging Face', url: 'https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct', paper: 'https://arxiv.org/abs/2504.05299' },
    license: {
      status: LICENSE_STATUS.CLEARED,
      reviewNote: "Approved for public redistribution (legal/OSS approval confirmed by the project owner, 2026-10-14).",
      upstream: 'Apache-2.0',
      provenance:
        'Upstream: HuggingFaceTB/SmolVLM-500M-Instruct (built on a SigLIP vision encoder and a SmolLM2 decoder; check both components). ',
    },
  },
  {
    id: 'whisper-base',
    name: 'Whisper base',
    shortName: 'Whisper base',
    task: 'speech-recognition',
    approxBytes: 241e6,
    approxBytesIsEstimate: false,
    assets: {
      "baseUrl": "https://huggingface.co/executorch-community/Whisper-Base-ExecuTorch-WebGPU/resolve/58b7dbb27117def05bca7cbd8af082554c58bb5a/", // @lint-ignore (base URL, files resolve under it)
      "files": [
        "vocab.json",
        "whisper_cross_kv.pte",
        "whisper_decode_stream.pte",
        "whisper_embed.i8.bin",
        "whisper_embed.scale.f32.bin",
        "whisper_encoder.pte",
        "whisper_lmhead.pte",
        "whisper_mel.pte",
        "whisper_pos_embed.f32.bin"
      ],
      "sha256": {
        "vocab.json": "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9",
        "whisper_cross_kv.pte": "6e8078924168c21a863ff662e9ccba55ed86bb3826dabd28fea77010e3151012",
        "whisper_decode_stream.pte": "bc2f33ad7aa164059e819247365e2e0a75a3f1a2973024e4fdc67a254d7faa8d",
        "whisper_embed.i8.bin": "bc7f8d8336ad935eb8b6374036478d92d679eae156a1d4188d5cc016b5f2dda2",
        "whisper_embed.scale.f32.bin": "1f58851911611b9686851c586365df2fcaaa0727957c05bd163d7c3ccc231f4d",
        "whisper_encoder.pte": "841e7702ba8fa73594f0b4e1400230d6b686af958628ead6b22919e9a3e27c69",
        "whisper_lmhead.pte": "68577fa240d29923058fd72b94db583f3ff171302ad12834c2b610e04c559f8e",
        "whisper_mel.pte": "7cae31829487958454c5a14aecfed0845f6320dd480c24883861fa39b07de497",
        "whisper_pos_embed.f32.bin": "88fa1cdbf2b06f86b0ecb7be0fccfc39e906502986572b8cf5319c250e857169"
      },
      "spec": {
        "files": {
          "mel": "whisper_mel.pte",
          "encoder": "whisper_encoder.pte",
          "crossKv": "whisper_cross_kv.pte",
          "lmHead": "whisper_lmhead.pte",
          "decodeStream": "whisper_decode_stream.pte",
          "embed": "whisper_embed.i8.bin",
          "embedScale": "whisper_embed.scale.f32.bin",
          "posEmbed": "whisper_pos_embed.f32.bin",
          "vocab": "vocab.json"
        },
        "bytes": {
          "mel": 710976,
          "encoder": 82420996,
          "crossKv": 12610560,
          "lmHead": 27184000,
          "decodeStream": 89270912,
          "embed": 26554880,
          "embedScale": 207460,
          "posEmbed": 917504,
          "vocab": 835550
        }
      }
    },
    credit: { by: 'OpenAI', url: 'https://huggingface.co/openai/whisper-base', paper: 'https://arxiv.org/abs/2212.04356' },
    license: {
      status: LICENSE_STATUS.CLEARED,
      reviewNote: "Approved for public redistribution (legal/OSS approval confirmed by the project owner, 2026-10-14).",
      upstream: 'Apache-2.0 (weights, per the upstream model card; the code is MIT)',
      provenance: 'Upstream: openai/whisper-base.',
    },
  },
  {
    id: 'depth-anything-v2-small',
    name: 'Depth Anything V2 Small',
    shortName: 'Depth Anything V2-S',
    task: 'depth-estimation',
    approxBytes: 97e6,
    approxBytesIsEstimate: false,
    assets: {
      "baseUrl": "https://huggingface.co/executorch-community/Depth-Anything-V2-Small-ExecuTorch-WebGPU/resolve/aeba0bbf167caf9007c88b480e8e78fb6e276ccf/", // @lint-ignore (base URL, files resolve under it)
      "files": [
        "depth_v2s_webgpu.pte"
      ],
      "sha256": {
        "depth_v2s_webgpu.pte": "285602ed3a34a26a4d9d36355ce978587ed555e68b6b5fa8a567c1155a3b5160"
      },
      "spec": {
        "pte": "depth_v2s_webgpu.pte",
        "pteBytes": 97307268,
        "inputSize": [
          224,
          224
        ]
      }
    },
    credit: { by: 'Lihe Yang et al. (HKU and TikTok)', url: 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf', paper: 'https://arxiv.org/abs/2406.09414' },
    license: {
      status: LICENSE_STATUS.CLEARED,
      reviewNote: "Approved for public redistribution (legal/OSS approval confirmed by the project owner, 2026-10-14).",
      upstream: 'Apache-2.0 (Small only; Base and Large are CC BY-NC 4.0)',
      provenance: 'Upstream: depth-anything/Depth-Anything-V2-Small.',
    },
  },
  {
    id: 'modnet',
    name: 'MODNet portrait matting',
    shortName: 'MODNet',
    task: 'matting',
    approxBytes: 26e6,
    approxBytesIsEstimate: false,
    assets: {
      "baseUrl": "https://huggingface.co/executorch-community/MODNet-ExecuTorch-WebGPU/resolve/fd4bfe709c151e686bf1de95dcd782238d4f65dd/", // @lint-ignore (base URL, files resolve under it)
      "files": [
        "modnet_webgpu.pte"
      ],
      "sha256": {
        "modnet_webgpu.pte": "0c2ff92e26087e98b4b2b61497d48d278be2474cc3cfabbc16f58da7eb5ad6c4"
      },
      "spec": {
        "pte": "modnet_webgpu.pte",
        "pteBytes": 26109828,
        "inputSize": [
          256,
          256
        ]
      }
    },
    credit: { by: 'Zhanghan Ke et al.', url: 'https://github.com/ZHKKKe/MODNet', paper: 'https://arxiv.org/abs/2011.11961' },
    license: {
      status: LICENSE_STATUS.CLEARED,
      reviewNote: "Approved for public redistribution (legal/OSS approval confirmed by the project owner, 2026-10-14).",
      upstream: 'Apache-2.0',
      provenance: 'Upstream: ZHKKKe/MODNet.',
    },
  },
  {
    id: 'real-esrgan',
    name: 'Real-ESRGAN ×4 (general v3)',
    shortName: 'Real-ESRGAN',
    task: 'super-resolution',
    approxBytes: 4.9e6,
    approxBytesIsEstimate: false,
    assets: {
      "baseUrl": "https://huggingface.co/executorch-community/Real-ESRGAN-General-x4v3-ExecuTorch-WebGPU/resolve/9cd66d389085ebf6cf65532551d0cac128134007/", // @lint-ignore (base URL, files resolve under it)
      "files": [
        "realesr_x4v3_webgpu.pte"
      ],
      "sha256": {
        "realesr_x4v3_webgpu.pte": "a4c40616767829cf6d3dab2c272b147764db1e2a01d610b2a4178e9153a66e5d"
      },
      "spec": {
        "name": "Real-ESRGAN ×4 (general v3)",
        "shortName": "Real-ESRGAN ×4 v3",
        "pte": "realesr_x4v3_webgpu.pte",
        "pteBytes": 4913472,
        "inputSize": [
          64,
          64
        ],
        "scale": 4
      }
    },
    credit: { by: 'Xintao Wang et al. (Tencent ARC)', url: 'https://github.com/xinntao/Real-ESRGAN', paper: 'https://arxiv.org/abs/2107.10833' },
    license: {
      status: LICENSE_STATUS.CLEARED,
      reviewNote: "Approved for public redistribution (legal/OSS approval confirmed by the project owner, 2026-10-14).",
      upstream: 'BSD-3-Clause',
      provenance: 'Upstream: xinntao/Real-ESRGAN, realesr-general-x4v3 weights.',
    },
  },
  {
    id: 'dfine-small',
    name: 'D-FINE small object detection',
    shortName: 'D-FINE S',
    task: 'object-detection',
    approxBytes: 42e6,
    approxBytesIsEstimate: false,
    assets: {
      "baseUrl": "https://huggingface.co/executorch-community/D-FINE-Small-COCO-ExecuTorch-WebGPU/resolve/c5a149dfd9886a16649c2f9c2148b78cbbc06396/", // @lint-ignore (base URL, files resolve under it)
      "files": [
        "dfine_s_coco_webgpu.pte"
      ],
      "sha256": {
        "dfine_s_coco_webgpu.pte": "96cc65f615dc2c79d1da80ef075b831fac8e39e8dcb58af9aff5bab154a4c4d6"
      },
      "spec": {
        "pte": "dfine_s_coco_webgpu.pte",
        "pteBytes": 42260612,
        "inputSize": [
          640,
          640
        ]
      }
    },
    credit: { by: 'Yansong Peng et al. (USTC)', url: 'https://github.com/Peterande/D-FINE', paper: 'https://arxiv.org/abs/2410.13842' },
    license: {
      status: LICENSE_STATUS.CLEARED,
      reviewNote: "Approved for public redistribution (approved and published by the project owner, 2026-10-15).",
      upstream: 'Apache-2.0',
      provenance: 'Upstream: Peterande/D-FINE, D-FINE-S COCO weights via the ustc-community/dfine-small-coco Hugging Face port, verified bit-identical to the official dfine_s_coco.pth.',
    },
  },
];

// Published: license cleared and hosted, so visitors can download it.
export function isPublished(model) {
  return model.license.status === LICENSE_STATUS.CLEARED && Boolean(model.assets?.baseUrl);
}

const BY_ID = new Map(MODELS.map((m) => [m.id, m]));

export function getModel(id) {
  const model = BY_ID.get(id);
  if (!model) throw new Error(`Unknown model id: ${id}`);
  return model;
}
