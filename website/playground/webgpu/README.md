# ExecuTorch WebGPU Playground

The playground for executorch.ai, served at `/playground/webgpu/`. Visitors run real
ExecuTorch models in their own browser on WebGPU: nothing to install, and
prompts, photos, and audio never leave the tab.

Zero dependencies and no build step: plain ES modules, one design system, hash
routing. The folder drops into the site as-is; the homepage links to it from the
nav ("Playground") and a call-to-action band under the hero.

## Run it locally

```bash
node scripts/serve.mjs                       # http://localhost:5173/playground/webgpu/ (+ LAN URLs for the no-WebGPU fallback)
node --test "tests/*.test.mjs"               # licensing gate, internal-reference scan, runtime, loader, wiring
node scripts/check-wiring.mjs                # static wiring check on its own
```

### Local-assets dev mode (real models before hosting is cleared)

```bash
node scripts/serve.mjs --dev-assets <dir> --dev-config <file-or-dir>
```

The server binds loopback only and adds two routes the deployed site never has:
`/__dev__/config.json` (model id → an `assets` block shaped like the one in
`models.js`, with `baseUrl` relative to `<dir>`) and `/__dev_assets__/*` (files
from `<dir>`). A dev entry takes precedence over the hosted one, so a new export
can be tried before it is published. Pages only look for the config when
they are themselves on a loopback host. Shipped model entries keep
`baseUrl: null` and `license.status: 'todo'`, so a deployed copy shows
"not available in this build" until a model is cleared.

To try it on a laptop GPU from a remote dev machine, forward the port
(`ssh -L 5173:localhost:5173 <host>`) and open `http://localhost:5173/playground/webgpu/`;
`localhost` is a secure context, so WebGPU is available.

## What runs

Every tile's badge comes from the `compute` field in `src/config/catalog.js`.

| Demo | Badge | What runs |
|---|---|---|
| On-device Chat | ExecuTorch | Qwen 3 0.6B, Qwen 2.5 0.5B, Llama 3.2 1B (4-bit `.pte`) with batched prefill, KV cache reused across turns, and JS sampling. Voice input through Whisper. |
| Segment Anything | ExecuTorch | SAM 2 Hiera Tiny: encode once, decode a mask per click or box. |
| VLM Q&A | ExecuTorch | SmolVLM-500M on a photo or camera snapshot. |
| Depth, Object Detection | ExecuTorch | Depth Anything V2 Small, YOLO26s, on photos or the live camera. |
| Background Removal, Super-Resolution | ExecuTorch | MODNet (photo or live camera), Real-ESRGAN ×4. |
| Speech to Text | ExecuTorch | Whisper base on the microphone or an audio file. |

## Model licensing gate

All checkpoints are referenced from one module, `src/config/models.js`.
`resolveWeights()` in `src/loader/resolve.js` returns remote weights only when
the license is `cleared` **and** an HTTPS URL is set; `tests/licensing.test.mjs`
asserts both halves. To enable a model after review: set
`license.status: 'cleared'`, record the decision, and set `assets.baseUrl` (a
commit-pinned URL). The download path (`src/loader/fetch-progress.js`) streams
with byte progress and caches in Cache Storage.

## Runtime layer

- `runtime/`: the ExecuTorch Wasm runtime with its WebGPU backend, shared by
  every demo. `runtime/BUILD_INFO.md` says how it was built and when to rebuild it.
- `src/runtime/executorch.js`: starts that runtime once (one WebGPU device),
  loads `.pte` programs, and runs one call into it at a time.
- `src/runtime/sources.js`: where a model's files come from (cleared remote URL,
  or the loopback dev config).
- `src/runtime/llm.js`: ExecuTorch LLM driver (batched prefill, decode, sampling).
- `src/runtime/forward.js`, `whisper.js`, `sam.js`: the other model drivers.
- `src/runtime/emscripten.js`: loads an Emscripten runtime script and its `.wasm`.
- `src/runtime/vlm.js`: the unlisted VLM demo, which still targets its own
  runtime build.

## Shipping to executorch.ai

1. Settle model licensing and host the cleared `.pte` files at commit-pinned
   URLs with CORS; fill in `models.js`. The runtime ships with the site
   (`runtime/`); serve `.wasm` as `application/wasm`.
2. Copy `site/` changes upstream: `playground/` (hub + `webgpu/`), `examples/` (redirect), the homepage nav item and CTA,
   and the `style.css` additions. `scripts/` and `tests/` can stay or be excluded.
3. HTTPS is required (WebGPU needs a secure context). Asset names are not
   content-hashed, so keep HTML/JS/CSS on a short cache or revalidation policy.
4. The runtime is single-threaded, so cross-origin isolation (COOP/COEP) is
   not required.
