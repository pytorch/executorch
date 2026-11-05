// A real ExecuTorch LLM on WebGPU: one .pte with an in-graph KV cache, driven
// token by token from JS through the shared runtime.
//
// The program's forward takes `tokens` [1, n] and `input_pos` [1] and returns
// the logits of the last token. Prefill feeds the prompt in chunks of up to
// `prefillMax` tokens; decode feeds one token at a time and samples in JS.
import { fetchWithProgress } from '../loader/fetch-progress.js';
import { i64, Program, startRuntime, withRuntime } from './executorch.js';
import { importVerified } from '../loader/import-verified.js';

async function loadTokenizer(lib, rootUrl, dir) {
  const tj = await importVerified(lib);
  tj.env.allowLocalModels = true;
  tj.env.allowRemoteModels = false;
  tj.env.useBrowserCache = false;
  tj.env.localModelPath = rootUrl;
  return tj.AutoTokenizer.from_pretrained(dir.replace(/\/$/, ''));
}

export class ExecuTorchLLM {
  // stages reported through onProgress: runtime, weights (bytes), tokenizer, gpu
  static async load(source, { onProgress = () => {}, signal } = {}) {
    const { spec, url } = source;
    onProgress({ stage: 'runtime', fraction: 0 });
    const [, pte, tokenizer] = await Promise.all([
      startRuntime().then(() => onProgress({ stage: 'runtime', fraction: 1 })),
      fetchWithProgress(url(spec.pte), {
        signal,
        expectedBytes: spec.pteBytes,
        onProgress: (p) => onProgress({ stage: 'weights', fraction: p.total ? p.loaded / p.total : 0, ...p }),
      }),
      loadTokenizer(source.shared.tokenizerLib, url(''), spec.tokenizer).then((t) => {
        onProgress({ stage: 'tokenizer', fraction: 1 });
        return t;
      }),
    ]);
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

    onProgress({ stage: 'gpu', fraction: 0 });
    const program = await withRuntime((et) => Program.load(et, pte, 'model'));
    onProgress({ stage: 'gpu', fraction: 1 });
    return new ExecuTorchLLM(program, tokenizer, spec);
  }

  constructor(program, tokenizer, spec) {
    this.program = program;
    this.tokenizer = tokenizer;
    this.spec = spec;
    this.maxCtx = spec.maxCtx;
    this.prefillMax = spec.prefillMax || 1;
    this.eos = new Set(spec.eos);
    this.committed = [];
    this.busy = Promise.resolve();
  }

  get pos() {
    return this.committed.length;
  }

  encodeChat(messages) {
    const chat = this.spec.chat ?? 'template';
    let ids = null;
    if (chat === 'template') {
      try {
        const enc = this.tokenizer.apply_chat_template(messages, { add_generation_prompt: true, tokenize: true, return_tensor: false });
        ids = Array.isArray(enc) ? enc.flat(Infinity).map(Number) : null;
      } catch {
        ids = null;
      }
      if (!ids?.length) throw new Error('The chat template could not render this conversation.');
    }
    if (!ids?.length) {
      // ChatML, the format of the Qwen family. "nothink" pre-closes Qwen 3's
      // reasoning block, which is what its template does with thinking disabled.
      let s = '';
      for (const m of messages) s += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
      s += '<|im_start|>assistant\n';
      if (chat === 'chatml-nothink') s += '<think>\n\n</think>\n\n';
      ids = Array.from(this.tokenizer.encode(s, { add_special_tokens: false })).map(Number);
    }
    return ids;
  }

  countTokens(text) {
    return this.tokenizer.encode(text, { add_special_tokens: false }).length;
  }

  reset() {
    this.committed = [];
  }

  // Feeds tokens at positions [pos, pos + tokens.length) and returns the logits
  // that follow the last one.
  #forward(tokens, pos) {
    return withRuntime(async (et) => {
      if (!this.program) throw new Error('This model was unloaded');
      const inputs = [i64(et, [1, tokens.length], tokens), i64(et, [1], [pos])];
      let outputs;
      try {
        outputs = await this.program.forward(inputs);
      } finally {
        for (const t of inputs) t.delete();
      }
      const logits = outputs[0].data.slice();
      for (const t of outputs) t.delete();
      return logits;
    });
  }

  // Fills KV positions [from, to) and returns the logits after the last token.
  async #prefill(ids, from, to, onChunk) {
    let logits = null;
    for (let off = from; off < to; ) {
      const count = Math.min(this.prefillMax, to - off);
      logits = await this.#forward(ids.slice(off, off + count), off);
      off += count;
      onChunk?.(off - from, to - from);
    }
    return logits;
  }

  // Frees the program on the GPU once any running generation has finished.
  dispose() {
    this.busy = this.busy.then(() =>
      withRuntime(() => {
        this.program?.delete();
        this.program = null;
      }),
    );
  }

  // Streams a reply to `messages`. The KV cache persists across turns, so only the
  // part of the prompt that differs from what is already resident is prefilled.
  async generate(messages, { maxNewTokens = 512, sampling = {}, signal, onPrefill, onText } = {}) {
    const run = this.busy.then(() => this.#generate(messages, { maxNewTokens, sampling, signal, onPrefill, onText }));
    this.busy = run.catch(() => {});
    return run;
  }

  async #generate(messages, { maxNewTokens, sampling, signal, onPrefill, onText }) {
    const ids = this.encodeChat(messages);
    if (ids.length + 16 > this.maxCtx) throw new Error(`The conversation (${ids.length} tokens) no longer fits the ${this.maxCtx}-token context.`);
    const budget = Math.min(maxNewTokens, this.maxCtx - ids.length - 1);

    let keep = 0;
    while (keep < this.committed.length && keep < ids.length - 1 && this.committed[keep] === ids[keep]) keep++;
    const t0 = performance.now();
    this.committed = ids.slice(0, keep);
    let logits = await this.#prefill(ids, keep, ids.length, onPrefill);
    this.committed = ids.slice();

    const sampler = makeSampler(sampling);
    const out = [];
    let text = '';
    let ttft = 0;
    let stop = 'length';
    // `committed` always lists exactly the tokens resident in the KV cache; the last
    // sampled token is only fed back when another one is wanted.
    const decodeStart = { t: 0 };
    for (let n = 0; n < budget; n++) {
      const next = sampler(logits, this.committed);
      if (next < 0) throw new Error('The model produced no usable logits (all values were non-finite).');
      if (n === 0) {
        ttft = performance.now() - t0;
        decodeStart.t = performance.now();
      }
      if (this.eos.has(next)) {
        stop = 'eos';
        break;
      }
      out.push(next);
      const full = this.tokenizer.decode(out, { skip_special_tokens: true });
      if (full.length > text.length && !full.endsWith('�')) {
        onText?.(full.slice(text.length), full);
        text = full;
      }
      if (signal?.aborted) {
        stop = 'aborted';
        break;
      }
      if (n + 1 < budget) {
        logits = await this.#forward([next], this.committed.length);
        this.committed.push(next);
      }
    }
    const final = this.tokenizer.decode(out, { skip_special_tokens: true });
    if (final.length > text.length) onText?.(final.slice(text.length), final);
    const decodeMs = performance.now() - decodeStart.t;
    return {
      text: final,
      stop,
      promptTokens: ids.length,
      prefilledTokens: ids.length - keep,
      newTokens: out.length,
      ttftMs: ttft,
      decodeTps: out.length > 1 ? (out.length - 1) / (decodeMs / 1000) : null,
    };
  }
}

// Temperature + top-k + top-p with a repetition penalty over recent context.
export function makeSampler({ temperature = 0.7, topK = 40, topP = 0.9, repetitionPenalty = 1.1, window = 96 } = {}) {
  return (logits, context) => {
    if (repetitionPenalty !== 1) {
      const seen = new Set(context.slice(-window));
      for (const t of seen) {
        if (t < logits.length) logits[t] = logits[t] > 0 ? logits[t] / repetitionPenalty : logits[t] * repetitionPenalty;
      }
    }
    if (temperature <= 0) {
      let best = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
      return best;
    }
    const k = Math.min(topK, logits.length);
    const top = [];
    for (let i = 0; i < logits.length; i++) {
      const v = logits[i];
      if (!Number.isFinite(v)) continue;
      if (top.length < k) {
        top.push([v, i]);
        if (top.length === k) top.sort((a, b) => a[0] - b[0]);
      } else if (v > top[0][0]) {
        top[0] = [v, i];
        for (let j = 0; j + 1 < k && top[j][0] > top[j + 1][0]; j++) [top[j], top[j + 1]] = [top[j + 1], top[j]];
      }
    }
    if (!top.length) return -1;
    top.sort((a, b) => b[0] - a[0]);
    const max = top[0][0];
    let sum = 0;
    const probs = top.map(([v]) => {
      const p = Math.exp((v - max) / temperature);
      sum += p;
      return p;
    });
    let cut = 0;
    let acc = 0;
    while (cut < probs.length) {
      acc += probs[cut] / sum;
      cut++;
      if (acc >= topP) break;
    }
    let r = Math.random() * acc * sum;
    for (let i = 0; i < cut; i++) {
      r -= probs[i];
      if (r <= 0) return top[i][1];
    }
    return top[cut - 1][1];
  };
}
