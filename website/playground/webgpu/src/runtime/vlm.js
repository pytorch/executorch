// SmolVLM on WebGPU: an Emscripten runtime holding three ExecuTorch programs
// (vision tower, single-token decoder, batched prefill) driven from JS.
//
//   image -> 512x512 RGB, normalized to [-1, 1], CHW -> et_run_vision -> 64 x 960 image embeddings
//   prompt ids -> rows of the int8 token-embedding table, dequantized on the host,
//     with the 64 image rows spliced in where the <image> placeholders sit
//   et_prefill_batch fills the KV cache for the whole prompt in one pass (from
//     position 0, at most `prefillMax` rows) and returns the first token;
//     et_prefill_step_embed / et_step_embed continue one row at a time from any position.
//
// Text embedding happens on the host because the decoder programs take
// embeddings, not token ids. The KV cache survives between questions, so a
// follow-up only prefills the new turn.
import { fetchWithProgress } from '../loader/fetch-progress.js';
import { hasExport, instantiate } from './emscripten.js';
import { SHARED } from '../config/shared.js';
import { importVerified } from '../loader/import-verified.js';

const FAKE_IMAGE = 49189; // <fake_token_around_image>
const GLOBAL_IMAGE = 49152; // <global-img>
const END_OF_UTTERANCE = 49279;
// "<|im_start|>User:" and "<end_of_utterance>\nAssistant:" as the chat template tokenizes them.
const USER_OPEN = [1, 11126, 42];
const ASSISTANT_OPEN = [END_OF_UTTERANCE, 198, 9519, 9531, 42];

async function loadTokenizer(lib, rootUrl, dir) {
  const tj = await importVerified(lib);
  tj.env.allowLocalModels = true;
  tj.env.allowRemoteModels = false;
  tj.env.useBrowserCache = false;
  tj.env.localModelPath = rootUrl;
  return tj.AutoTokenizer.from_pretrained(dir.replace(/\/$/, ''));
}

const aborted = () => new DOMException('Aborted', 'AbortError');

// RGBA bytes -> planar RGB floats, rescaled and normalized as the model's image processor does.
export function normalizeRGBA(rgba, plane, { rescale, mean, std }) {
  const px = new Float32Array(3 * plane);
  const k = rescale / std;
  const b = mean / std;
  for (let p = 0; p < plane; p++) {
    px[p] = rgba[p * 4] * k - b;
    px[plane + p] = rgba[p * 4 + 1] * k - b;
    px[2 * plane + p] = rgba[p * 4 + 2] * k - b;
  }
  return px;
}

// "<|im_start|>User:<image>{question}<end_of_utterance>\nAssistant:", where <image>
// expands to the wrapped run of placeholders the image embeddings replace.
export function firstTurnIds(meta, questionIds) {
  const image = [FAKE_IMAGE, GLOBAL_IMAGE, ...Array(meta.n_image_tokens).fill(meta.image_token_id), FAKE_IMAGE];
  return { ids: [...USER_OPEN, ...image, ...questionIds, ...ASSISTANT_OPEN], imageStart: USER_OPEN.length + 2 };
}

// "<end_of_utterance>\nUser: {question}<end_of_utterance>\nAssistant:" after a previous answer.
export function followUpIds(pendingToken, userTurnIds) {
  return [...(pendingToken == null ? [] : [pendingToken]), END_OF_UTTERANCE, ...userTurnIds, ...ASSISTANT_OPEN];
}

export class VisionLanguageModel {
  // stages reported through onProgress: runtime, download (bytes over all files), gpu
  static async load(source, { onProgress = () => {}, signal } = {}) {
    const { spec, url } = source;
    const logs = [];
    onProgress({ stage: 'runtime', fraction: 0 });
    const module = await instantiate(url(spec.runtime.script), spec.runtime.factory, {
      wasmUrl: url(spec.runtime.wasm),
      onLog: (kind, m) => (logs.push(m), console.debug(`[ExecuTorch] ${m}`)),
    });
    if (!module.ccall('et_init', 'number', [], [])) throw new Error('No WebGPU device available to the ExecuTorch runtime');
    onProgress({ stage: 'runtime', fraction: 1 });

    const files = Object.entries(spec.files);
    const total = files.reduce((s, [, f]) => s + (f.bytes ?? 0), 0);
    const loaded = Object.fromEntries(files.map(([k]) => [k, 0]));
    let fromCache = true;
    const report = () => {
      const got = Object.values(loaded).reduce((a, b) => a + b, 0);
      onProgress({ stage: 'download', fraction: total ? Math.min(1, got / total) : 0, loaded: got, total, fromCache });
    };
    const libUrl = SHARED.tokenizerLib;
    if (!libUrl) throw new Error('No tokenizer library is configured for this build');
    const [blobs, meta, tokenizer] = await Promise.all([
      Promise.all(
        files.map(([key, f]) =>
          fetchWithProgress(url(f.path), {
            signal,
            expectedBytes: f.bytes,
            onProgress: (p) => {
              loaded[key] = p.loaded;
              fromCache &&= p.fromCache;
              report();
            },
          }).then((bytes) => [key, bytes]),
        ),
      ).then(Object.fromEntries),
      fetch(url(spec.meta), { signal }).then((r) => {
        if (!r.ok) throw new Error(`Could not read the model description (HTTP ${r.status})`);
        return r.json();
      }),
      loadTokenizer(libUrl, url(''), spec.tokenizer),
    ]);
    if (signal?.aborted) throw aborted();

    onProgress({ stage: 'gpu', fraction: 0 });
    for (const [fn, types, values] of spec.configure ?? []) {
      if (hasExport(module, fn)) module.ccall(fn, null, types, values);
    }
    const programs = [
      ['vision', 'et_load_vision', [], []],
      ['decoder', 'et_load_decoder', [], []],
      ['prefill', 'et_load_prefill', ['number'], [spec.prefillMax]],
    ];
    let hasPrefill = false;
    for (const [i, [key, fn, extraTypes, extraArgs]] of programs.entries()) {
      if (!blobs[key] || !hasExport(module, fn)) continue;
      const path = `/${key}.pte`;
      module.FS.writeFile(path, blobs[key], { canOwn: true });
      delete blobs[key];
      const ok = await module.ccall(fn, 'number', ['string', ...extraTypes], [path, ...extraArgs], { async: true });
      try {
        module.FS.unlink(path);
      } catch {
        // The runtime may already have taken ownership of the file.
      }
      if (key === 'prefill') hasPrefill = ok === 1;
      else if (!ok) throw new Error(`The runtime could not load the ${key} program${logs.length ? `: ${logs.at(-1)}` : ''}`);
      onProgress({ stage: 'gpu', fraction: (i + 1) / programs.length });
      if (signal?.aborted) throw aborted();
    }
    return new VisionLanguageModel(module, tokenizer, spec, meta, {
      embed: new Int8Array(blobs.embed.buffer, blobs.embed.byteOffset, blobs.embed.byteLength),
      scale: new Float32Array(blobs.embedScale.buffer.slice(blobs.embedScale.byteOffset, blobs.embedScale.byteOffset + blobs.embedScale.byteLength)),
      hasPrefill,
    });
  }

  constructor(module, tokenizer, spec, meta, { embed, scale, hasPrefill }) {
    this.m = module;
    this.tokenizer = tokenizer;
    this.hidden = meta.hidden;
    this.size = meta.image_size;
    this.meta = meta;
    this.embed = embed;
    this.scale = scale;
    this.hasPrefill = hasPrefill;
    this.prefillMax = hasPrefill ? spec.prefillMax : 0;
    this.maxCtx = spec.maxCtx ?? 512;
    this.eos = new Set(meta.eos_token_ids);
    this.imageTokens = meta.n_image_tokens;
    this.imageStart = firstTurnIds(meta, []).imageStart;

    const plane = this.size * this.size;
    this.pixPtr = module._malloc(3 * plane * 4);
    this.rowPtr = module._malloc(this.hidden * 4);
    this.batchPtr = this.prefillMax ? module._malloc(this.prefillMax * this.hidden * 4) : 0;
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.canvas.height = this.size;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });

    this.pixels = null;
    this.imageEmbeds = null;
    this.pos = 0;
    this.pending = null;
    this.busy = Promise.resolve();
  }

  #enqueue(fn) {
    const run = this.busy.then(fn);
    this.busy = run.catch(() => {});
    return run;
  }

  // Accepts anything drawImage does (ImageBitmap, canvas, <img>, <video>).
  // Starts encoding right away; resolves with { visionMs } once the image is
  // encoded, or { superseded: true } if another image replaced it first.
  setImage(source) {
    const { size, ctx } = this;
    ctx.drawImage(source, 0, 0, size, size);
    const px = normalizeRGBA(ctx.getImageData(0, 0, size, size).data, size * size, this.meta);
    this.pixels = px;
    // The reset waits its turn: an ask still running reads the old embeddings and advances `pos`.
    return this.#enqueue(() => {
      if (this.pixels !== px) return { superseded: true };
      this.imageEmbeds = null;
      this.pos = 0;
      this.pending = null;
      return this.#encode();
    });
  }

  get hasImage() {
    return this.pixels != null;
  }

  // Forget the conversation but keep the encoded image.
  resetConversation() {
    return this.#enqueue(() => {
      this.pos = 0;
      this.pending = null;
    });
  }

  async #encode() {
    const t0 = performance.now();
    this.m.HEAPF32.set(this.pixels, this.pixPtr >> 2);
    const ptr = await this.m.ccall('et_run_vision', 'number', ['number', 'number', 'number', 'number'], [this.pixPtr, 3, this.size, this.size], { async: true });
    if (!ptr) throw new Error('The vision encoder failed to run');
    const n = this.m.ccall('et_image_tokens', 'number', [], []);
    if (n !== this.imageTokens) throw new Error(`The vision encoder returned ${n} image tokens, expected ${this.imageTokens}`);
    this.imageEmbeds = this.m.HEAPF32.slice(ptr >> 2, (ptr >> 2) + n * this.hidden);
    return { visionMs: performance.now() - t0 };
  }

  #writeTokenRow(token, dst) {
    const heap = this.m.HEAPF32;
    const { hidden, embed } = this;
    const src = token * hidden;
    const s = this.scale[token];
    for (let j = 0; j < hidden; j++) heap[dst + j] = embed[src + j] * s;
  }

  // Row `pos` of the prompt: an image embedding inside the placeholder run, a token embedding elsewhere.
  #writeRow(ids, i, absPos, dst) {
    const img = absPos - this.imageStart;
    if (img >= 0 && img < this.imageTokens && ids[i] === this.meta.image_token_id) {
      this.m.HEAPF32.set(this.imageEmbeds.subarray(img * this.hidden, (img + 1) * this.hidden), dst);
    } else {
      this.#writeTokenRow(ids[i], dst);
    }
  }

  // Fills KV positions [this.pos, this.pos + ids.length) and returns the token
  // predicted after the last one.
  async #prefill(ids, signal) {
    const start = this.pos;
    let i = 0;
    let first = -1;
    if (start === 0 && this.prefillMax) {
      const count = Math.min(ids.length, this.prefillMax);
      const base = this.batchPtr >> 2;
      for (let r = 0; r < count; r++) this.#writeRow(ids, r, r, base + r * this.hidden);
      first = await this.m.ccall('et_prefill_batch', 'number', ['number', 'number', 'number'], [this.batchPtr, count, this.hidden], { async: true });
      if (first < 0) throw new Error(`Batched prefill failed (${first})`);
      i = count;
    }
    for (; i < ids.length; i++) {
      if (signal?.aborted) throw aborted();
      this.#writeRow(ids, i, start + i, this.rowPtr >> 2);
      if (i < ids.length - 1) {
        await this.m.ccall('et_prefill_step_embed', null, ['number', 'number', 'number'], [this.rowPtr, this.hidden, start + i], { async: true });
      } else {
        first = await this.m.ccall('et_step_embed', 'number', ['number', 'number', 'number'], [this.rowPtr, this.hidden, start + i], { async: true });
      }
    }
    this.pos = start + ids.length;
    return first;
  }

  // Streams an answer about the current image. Follow-ups continue the same
  // conversation; when it no longer fits the context it restarts from the image.
  ask(question, { onText, onStage, signal, maxNewTokens = 160 } = {}) {
    const t0 = performance.now();
    return this.#enqueue(() => this.#ask(question, { onText, onStage, signal, maxNewTokens, t0 }));
  }

  async #ask(question, { onText, onStage, signal, maxNewTokens, t0 }) {
    if (!this.pixels) throw new Error('Choose an image first');
    let visionMs = 0;
    if (!this.imageEmbeds) {
      onStage?.('vision');
      ({ visionMs } = await this.#encode());
    }
    if (signal?.aborted) throw aborted();

    const encode = (text) => Array.from(this.tokenizer.encode(text, { add_special_tokens: false })).map(Number);
    let ids = null;
    let restarted = false;
    if (this.pos > 0) {
      const turn = followUpIds(this.pending, encode(`\nUser: ${question.trim()}`));
      if (this.pos + turn.length + Math.min(maxNewTokens, 32) < this.maxCtx) ids = turn;
      else restarted = true;
    }
    this.pending = null;
    if (!ids) {
      this.pos = 0;
      ids = firstTurnIds(this.meta, encode(question.trim())).ids;
      if (ids.length + 8 > this.maxCtx) throw new Error('That question is too long for this model’s context');
    }

    onStage?.('prefill');
    const promptTokens = ids.length;
    let tok;
    try {
      tok = await this.#prefill(ids, signal);
    } catch (err) {
      // A partial prefill leaves the cache out of step with `pos`.
      this.pos = 0;
      throw err;
    }
    const ttftMs = performance.now() - t0;
    onStage?.('decode');

    const budget = Math.max(1, Math.min(maxNewTokens, this.maxCtx - this.pos - 1));
    const out = [];
    let text = '';
    let stop = 'eos';
    let steps = 0;
    const decodeStart = performance.now();
    for (;;) {
      if (tok < 0) throw new Error('The decoder failed');
      if (this.eos.has(tok)) break;
      out.push(tok);
      const full = this.tokenizer.decode(out, { skip_special_tokens: true }).replace(/^\s+/, '');
      if (full.length > text.length && !full.endsWith('\ufffd')) {
        text = full;
        onText?.(text);
      }
      if (out.length >= budget || signal?.aborted) {
        stop = signal?.aborted ? 'aborted' : 'length';
        // Not yet in the cache; a follow-up feeds it first.
        this.pending = tok;
        break;
      }
      this.#writeTokenRow(tok, this.rowPtr >> 2);
      tok = await this.m.ccall('et_step_embed', 'number', ['number', 'number', 'number'], [this.rowPtr, this.hidden, this.pos], { async: true });
      this.pos++;
      steps++;
    }
    const decodeMs = performance.now() - decodeStart;
    const final = this.tokenizer.decode(out, { skip_special_tokens: true }).trim();
    if (final !== text) onText?.(final);
    return {
      text: final,
      stop,
      restarted,
      visionMs,
      ttftMs,
      promptTokens,
      newTokens: out.length,
      tps: steps > 0 ? steps / (decodeMs / 1000) : null,
      contextUsed: this.pos,
    };
  }
}
