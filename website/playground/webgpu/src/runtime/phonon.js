/**
 * Phonon 31M (English) text to speech on the ExecuTorch WebGPU runtime.
 *
 *   const tts = await loadPhonon({ onProgress, signal });   // one shared instance per page
 *   const stats = await tts.speak(text, { voice, temperature, seed, signal, onAudio });
 *
 * Two programs (exported by the phonon-31m-en recipe):
 * - flow_lm: forward(embeds [1,S,512], input_pos [1], noise [1,32]) -> latent [1,32],
 *   eos_logit [1,1], next_embeds [1,1,512]. Its KV cache lives in the program: the voice
 *   prompt fills positions 0..124 once per voice, each text chunk is prompted at 125, then
 *   one call per 80 ms frame feeds back the previous call's next_embeds.
 * - mimi: forward(latent [1,1,32], ...state) -> pcm [1,1,1920], ...state. One frame of the
 *   streaming audio decoder; its state goes in and out each call (zeros at a chunk start).
 *
 * The host does the rest: text normalization and tokenization (src/audio/tts-text.js,
 * src/audio/unigram.js), the text-embedding lookup (a bf16 table), the noise, and the
 * end-of-speech rule. onAudio(pcm Float32Array at 24 kHz, info) streams every frame.
 */
import { getModel } from '../config/models.js';
import { fetchWithProgress } from '../loader/fetch-progress.js';
import { EosPolicy, gaussian, maxTokensFor, planChunks } from '../audio/tts-text.js';
import { UnigramTokenizer } from '../audio/unigram.js';
import { mulberry32 } from '../lib/prng.js';
import { f32, i64, Program, readOutputs, startRuntime, withRuntime } from './executorch.js';
import { resolveModelSource } from './sources.js';

export const PHONON_MODEL_ID = 'phonon-31m-en';
const DIM = 512;
const LATENT = 32;

export class PhononUnavailableError extends Error {
  constructor(reasons = []) {
    super('Phonon is not available in this build.');
    this.name = 'PhononUnavailableError';
    this.reasons = reasons;
  }
}

const aborted = () => new DOMException('Aborted', 'AbortError');
const floats = (bytes) => new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));

// One chunk's frames: step(k) runs frame k and resolves { eosLogit }. Stops at the
// frame budget, or `framesAfterEos` frames after the first EOS (the EOS frame and
// that tail are kept, as the official runtime does). -> frames generated.
export async function runChunk({ step, frameBudget, framesAfterEos, eosThreshold, signal }) {
  const eos = new EosPolicy(framesAfterEos);
  let frames = 0;
  for (let k = 0; k < frameBudget; k++) {
    if (signal?.aborted) throw aborted();
    const { eosLogit } = await step(k);
    frames++;
    if (eos.shouldStop(eosLogit > eosThreshold)) break;
  }
  return frames;
}

// bf16 rows (as stored in the checkpoint) to f32: the bf16 bits are the high half.
export function embedRows(table, ids, dim = DIM) {
  const out = new Float32Array(ids.length * dim);
  const bits = new Uint32Array(out.buffer);
  ids.forEach((id, r) => {
    for (let j = 0; j < dim; j++) bits[r * dim + j] = table[id * dim + j] << 16;
  });
  return out;
}

class PhononTTS {
  static async load(emit, signal) {
    const source = await resolveModelSource(getModel(PHONON_MODEL_ID));
    if (source.kind === 'unavailable') throw new PhononUnavailableError(source.reasons);
    if (!globalThis.navigator?.gpu) throw new Error('WebGPU is not available in this browser.');
    const { spec, url } = source;

    emit({ stage: 'runtime', fraction: 0 });
    await startRuntime();
    emit({ stage: 'runtime', fraction: 1 });
    if (signal.aborted) throw aborted();

    const entries = [
      ...['flowLm', 'mimi', 'textEmbed', 'bosEmbed', 'tokenizer'].map((k) => [k, spec.files[k]]),
      ...spec.voices.map((v) => [`voice:${v.id}`, v]),
    ];
    const loaded = {};
    const totals = Object.fromEntries(entries.map(([k, f]) => [k, f.bytes ?? 0]));
    const report = () => {
      const l = entries.reduce((s, [k]) => s + (loaded[k] ?? 0), 0);
      const t = entries.reduce((s, [k]) => s + totals[k], 0);
      emit({ stage: 'download', fraction: t ? Math.min(1, l / t) : 0, loaded: l, total: t });
    };
    report();
    const files = Object.fromEntries(
      await Promise.all(
        entries.map(async ([k, f]) => [
          k,
          await fetchWithProgress(url(f.path), {
            signal,
            expectedBytes: totals[k],
            onProgress: (p) => {
              loaded[k] = p.loaded;
              if (p.total) totals[k] = p.total;
              report();
            },
          }),
        ]),
      ),
    );
    if (signal.aborted) throw aborted();

    const programs = {};
    try {
      emit({ stage: 'gpu', fraction: 0, detail: 'speech model' });
      programs.flow = await withRuntime((et) => Program.load(et, files.flowLm, 'speech model'));
      files.flowLm = null;
      if (signal.aborted) throw aborted();
      emit({ stage: 'gpu', fraction: 0.6, detail: 'audio decoder' });
      programs.mimi = await withRuntime((et) => Program.load(et, files.mimi, 'audio decoder'));
      files.mimi = null;
      if (signal.aborted) throw aborted();
    } catch (err) {
      withRuntime(() => Object.values(programs).forEach((p) => p.delete()));
      throw err;
    }
    emit({ stage: 'gpu', fraction: 1 });

    const voices = new Map(spec.voices.map((v) => [v.id, floats(files[`voice:${v.id}`])]));
    return new PhononTTS(programs, spec, {
      table: new Uint16Array(files.textEmbed.buffer, files.textEmbed.byteOffset, files.textEmbed.byteLength / 2),
      bos: floats(files.bosEmbed),
      tokenizer: new UnigramTokenizer(JSON.parse(new TextDecoder().decode(files.tokenizer))),
      voices,
    });
  }

  constructor(programs, spec, { table, bos, tokenizer, voices }) {
    this.p = programs;
    this.spec = spec;
    this.table = table;
    this.bos = bos;
    this.tokenizer = tokenizer;
    this.voices = voices;
    this.stateShapes = spec.mimiState.map(([, shape]) => shape);
    this.voiceLen = spec.voiceLen;
    this.primed = null; // the voice whose prompt is in the KV cache
    this.busy = Promise.resolve();
  }

  get sampleRate() {
    return this.spec.sampleRate;
  }

  // Text chunks as the model will see them (normalized, tokenized, budgeted).
  plan(text) {
    const room = this.spec.maxContext - this.voiceLen;
    return planChunks(this.tokenizer, text, { maxTokens: 50, fitTo: Math.min(this.spec.prefillMax, maxTokensFor(room)) });
  }

  speak(text, opts = {}) {
    const run = this.busy.then(() => this.#speak(text, opts));
    this.busy = run.catch(() => {});
    return run;
  }

  // One flow-LM call; must run inside withRuntime(). -> outputs as { data, shape }.
  async #flow(et, embeds, rows, pos, noise) {
    const inputs = [f32(et, [1, rows, DIM], embeds), i64(et, [1], [pos]), f32(et, [1, LATENT], noise)];
    try {
      return readOutputs(await this.p.flow.forward(inputs));
    } finally {
      for (const t of inputs) t.delete();
    }
  }

  async #prompt(embeds, rows, pos) {
    await withRuntime((et) => this.#flow(et, embeds, rows, pos, new Float32Array(LATENT)));
  }

  async #speak(text, { voice = this.spec.voices[0].id, temperature = this.spec.temperature, seed = 1, signal, onAudio } = {}) {
    const t0 = performance.now();
    const chunks = this.plan(text);
    if (!chunks.length) throw new Error('Nothing to say: the text is empty.');
    const emb = this.voices.get(voice);
    if (!emb) throw new Error(`Unknown voice: ${voice}`);
    const rng = mulberry32(seed);
    const stats = { chunks: chunks.length, tokens: 0, frames: 0, samples: 0, promptMs: 0, firstAudioMs: null, flowMs: 0, mimiMs: 0 };
    if (this.primed !== voice) {
      const t = performance.now();
      this.primed = null;
      await this.#prompt(emb, this.voiceLen, 0);
      this.primed = voice;
      stats.promptMs += performance.now() - t;
    }
    for (let c = 0; c < chunks.length; c++) {
      if (signal?.aborted) throw aborted();
      const chunk = chunks[c];
      stats.tokens += chunk.tokens.length;
      let pos = this.voiceLen;
      const tp = performance.now();
      await this.#prompt(embedRows(this.table, chunk.tokens), chunk.tokens.length, pos);
      stats.promptMs += performance.now() - tp;
      pos += chunk.tokens.length;
      let embed = this.bos;
      let state = this.stateShapes.map((s) => new Float32Array(s.reduce((a, b) => a * b, 1)));
      const step = (k) =>
        withRuntime(async (et) => {
          const t1 = performance.now();
          const [latent, eos, next] = await this.#flow(et, embed, 1, pos++, gaussian(rng, LATENT, temperature));
          const t2 = performance.now();
          const inputs = [f32(et, [1, 1, LATENT], latent.data), ...state.map((s, i) => f32(et, this.stateShapes[i], s))];
          let outs;
          try {
            outs = readOutputs(await this.p.mimi.forward(inputs));
          } finally {
            for (const t of inputs) t.delete();
          }
          stats.flowMs += t2 - t1;
          stats.mimiMs += performance.now() - t2;
          embed = next.data;
          state = outs.slice(1).map((o) => o.data);
          stats.frames++;
          stats.samples += outs[0].data.length;
          stats.firstAudioMs ??= performance.now() - t0;
          onAudio?.(outs[0].data, { chunk: c, chunks: chunks.length, frame: k, frameBudget: chunk.frameBudget, text: chunk.text });
          return { eosLogit: eos.data[0] };
        });
      await runChunk({ step, frameBudget: chunk.frameBudget, framesAfterEos: chunk.framesAfterEos, eosThreshold: this.spec.eosThreshold, signal });
    }
    stats.ms = performance.now() - t0;
    stats.audioSec = stats.samples / this.sampleRate;
    return stats;
  }
}

let ready = null;
let pending = null;

/** Loads (once) and returns the shared synthesizer. See the module comment. */
export function loadPhonon({ onProgress, signal } = {}) {
  if (ready) return Promise.resolve(ready);
  if (!pending) {
    const controller = new AbortController();
    const listeners = new Set();
    let last = null;
    const emit = (e) => {
      last = e;
      for (const l of listeners) l(e);
    };
    pending = { controller, listeners, last: () => last };
    pending.promise = PhononTTS.load(emit, controller.signal).then(
      (tts) => (ready = tts),
      (err) => {
        pending = null;
        throw err;
      },
    );
  }
  const job = pending;
  if (onProgress) {
    job.listeners.add(onProgress);
    if (job.last()) onProgress(job.last());
  }
  signal?.addEventListener('abort', () => job.controller.abort(), { once: true });
  return job.promise.finally(() => onProgress && job.listeners.delete(onProgress));
}

export function isPhononLoaded() {
  return Boolean(ready);
}
