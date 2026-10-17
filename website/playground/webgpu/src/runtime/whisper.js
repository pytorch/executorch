/**
 * Whisper base speech recognition on the ExecuTorch WebGPU runtime.
 *
 *   import { isWhisperAvailable, loadWhisper } from './runtime/whisper.js';
 *   if (await isWhisperAvailable()) {
 *     const asr = await loadWhisper({ onProgress, signal });   // one shared instance per page
 *     const { text, ms } = await asr.transcribe(pcm, { onPartial, signal, language: 'en' });
 *   }
 *
 * - `pcm` is a Float32Array of mono samples at 16 kHz (see src/audio/: recordMic, decodeAudioFile).
 * - onProgress({ stage: 'runtime' | 'download' | 'gpu', fraction, loaded?, total?, detail? })
 * - onPartial(textSoFar, { window, windows }) streams text while decoding.
 * - transcribe() resolves { text, ms, audioSec, segments: [{ start, end, text }] } (seconds).
 *   Calls are serialised, so concurrent callers queue. Audio longer than one window is split
 *   at quiet points; silent windows are skipped.
 * - Load errors are Error (WhisperUnavailableError when no model source resolves). Aborting
 *   one caller's signal rejects that caller with an AbortError; the shared download only
 *   stops when every waiting caller has aborted.
 */
import { getModel } from '../config/models.js';
import { fetchWithProgress } from '../loader/fetch-progress.js';
import { planWindows, rms, SPEECH_RATE } from '../audio/pcm.js';
import { f32, i64, Program, readOutputs, startRuntime, withRuntime } from './executorch.js';
import { resolveModelSource } from './sources.js';

export const WHISPER_MODEL_ID = 'whisper-base';

// The mel program's input: 30 s of audio as [3000, 400] reflect-padded STFT
// frames (hop 160). The Hann window and mel filterbank are baked into the
// program, so the host frames the audio before it and takes the log after it.
const FRAMES = 3000;
const N_FFT = 400;
const HOP = 160;
const PAD = N_FFT >> 1;
const N_SAMPLES = FRAMES * HOP;
const N_MELS = 80;
// Decoder positions per window: the rows of `decoder`, and the prefix plus
// about four tokens per second of fast speech.
const MAX_DEC = 64;
const SILENCE_RMS = 4e-4;

// VOCAB is the real vocabulary; the output head pads its logits beyond it.
const TOK = { SOT: 50258, EOT: 50257, TRANSCRIBE: 50359, NOTIMESTAMPS: 50363, FIRST_SPECIAL: 50257, VOCAB: 51865 };

export const WHISPER_LANGUAGES = [
  { code: 'en', label: 'English', id: 50259 },
  { code: 'es', label: 'Spanish', id: 50262 },
  { code: 'fr', label: 'French', id: 50265 },
  { code: 'de', label: 'German', id: 50261 },
  { code: 'it', label: 'Italian', id: 50274 },
  { code: 'pt', label: 'Portuguese', id: 50267 },
  { code: 'nl', label: 'Dutch', id: 50271 },
  { code: 'ru', label: 'Russian', id: 50263 },
  { code: 'hi', label: 'Hindi', id: 50276 },
  { code: 'zh', label: 'Chinese', id: 50260 },
  { code: 'ja', label: 'Japanese', id: 50266 },
  { code: 'ko', label: 'Korean', id: 50264 },
];

// Programs, in load order. The decoder step is `decodeStream` (one token per
// call against an in-graph KV cache) when the spec has it, else `decoder`
// (recomputes the whole prefix every step); only the one used is downloaded.
const PROGRAMS = [
  ['mel', 'log-mel front end'],
  ['encoder', 'encoder'],
  ['crossKv', 'cross-attention K/V'],
  ['decodeStream', 'streaming decoder'],
  ['decoder', 'decoder'],
  ['lmHead', 'output head'],
];

export class WhisperUnavailableError extends Error {
  constructor(reasons = []) {
    super('Whisper is not available in this build.');
    this.name = 'WhisperUnavailableError';
    this.reasons = reasons;
  }
}

const aborted = () => new DOMException('Aborted', 'AbortError');

// GPT-2 style byte-level BPE: every byte maps to a printable code point.
function unicodeToByte() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n++);
    }
  }
  return new Map(cs.map((c, i) => [String.fromCodePoint(c), bs[i]]));
}

// Detokenizer only (the decoder emits ids; nothing is ever encoded). Special
// tokens (>= 50257) are dropped, as with skip_special_tokens.
export class WhisperDetokenizer {
  constructor(vocab) {
    this.tokens = [];
    for (const [tok, id] of Object.entries(vocab)) this.tokens[id] = tok;
    this.bytes = unicodeToByte();
    this.utf8 = new TextDecoder('utf-8');
  }

  decode(ids, { partial = false } = {}) {
    const out = [];
    for (const id of ids) {
      if (id >= TOK.FIRST_SPECIAL) continue;
      for (const ch of this.tokens[id] ?? '') {
        const b = this.bytes.get(ch);
        if (b !== undefined) out.push(b);
      }
    }
    const text = this.utf8.decode(new Uint8Array(out));
    // A partial sequence may stop mid code point.
    return partial ? text.replace(/�+$/, '') : text;
  }
}

// Whisper's classic failure is a short loop ("you you you ..."). True when the
// tail repeats with period 1..4 at least four times (needs >= 8 tokens).
export function isLooping(gen) {
  const n = gen.length;
  if (n < 8) return false;
  for (let p = 1; p <= 4; p++) {
    if (n < p * 4) continue;
    let cyc = true;
    for (let i = 1; i <= p * 3 && cyc; i++) if (gen[n - i] !== gen[n - i - p]) cyc = false;
    if (cyc) return true;
  }
  return false;
}

// Collapses a trailing repeated cycle to one instance.
export function trimLoop(gen) {
  const n = gen.length;
  if (!isLooping(gen)) return gen;
  for (let p = 1; p <= 4; p++) {
    let cyc = n >= p * 4;
    for (let i = 1; i <= p * 3 && cyc; i++) if (gen[n - i] !== gen[n - i - p]) cyc = false;
    if (!cyc) continue;
    let end = n;
    while (end - 2 * p >= 0) {
      let same = true;
      for (let i = 0; i < p && same; i++) if (gen[end - 1 - i] !== gen[end - 1 - i - p]) same = false;
      if (!same) break;
      end -= p;
    }
    return gen.slice(0, end);
  }
  return gen;
}

// Raw framed audio [FRAMES, N_FFT]: zero-pad to 30 s, reflect-pad n_fft/2 on
// both ends (torch.stft center=True), then hop-160 windows.
export function buildFrames(audio, out = new Float32Array(FRAMES * N_FFT)) {
  const base = new Float32Array(N_SAMPLES + 2 * PAD);
  base.set(audio.subarray(0, Math.min(audio.length, N_SAMPLES)), PAD);
  for (let i = 0; i < PAD; i++) {
    base[PAD - 1 - i] = base[PAD + 1 + i];
    base[PAD + N_SAMPLES + i] = base[PAD + N_SAMPLES - 2 - i];
  }
  for (let t = 0; t < FRAMES; t++) out.set(base.subarray(t * HOP, t * HOP + N_FFT), t * N_FFT);
  return out;
}

// Whisper's log-mel features from the mel program's power spectrum
// [frames, mels]: log10 with a floor, clamped to 8 below the peak, scaled to
// about [-1, 1], and transposed to the encoder's [mels, frames].
export function logMel(power, frames = FRAMES, mels = N_MELS) {
  const x = new Float32Array(power.length);
  let max = -Infinity;
  for (let i = 0; i < power.length; i++) {
    x[i] = Math.log10(Math.max(power[i], 1e-10));
    if (x[i] > max) max = x[i];
  }
  const floor = Math.fround(max - 8);
  const out = new Float32Array(power.length);
  for (let t = 0; t < frames; t++) for (let m = 0; m < mels; m++) out[m * frames + t] = (Math.max(x[t * mels + m], floor) + 4) / 4;
  return out;
}

export function argmax(v, n = v.length) {
  let best = 0;
  for (let i = 1; i < n; i++) if (v[i] > v[best]) best = i;
  return best;
}

// Runs a program on `temps` (freed afterwards) followed by `keep` (left alone),
// and returns copies of its outputs. Must run inside withRuntime().
async function once(program, temps, keep = []) {
  try {
    return readOutputs(await program.forward([...temps, ...keep]));
  } finally {
    for (const t of temps) t.delete();
  }
}

const floats = (bytes) => new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));

class WhisperRecognizer {
  static async load(emit, signal) {
    const source = await resolveModelSource(getModel(WHISPER_MODEL_ID));
    if (source.kind === 'unavailable') throw new WhisperUnavailableError(source.reasons);
    if (!globalThis.navigator?.gpu) throw new Error('WebGPU is not available in this browser.');
    const { spec, url } = source;
    const stream = Boolean(spec.files.decodeStream && spec.files.posEmbed);
    const programs = PROGRAMS.filter(([k]) => (k === 'decodeStream' ? stream : k === 'decoder' ? !stream : true));

    emit({ stage: 'runtime', fraction: 0 });
    await startRuntime();
    emit({ stage: 'runtime', fraction: 1 });
    if (signal.aborted) throw aborted();

    const keys = [...programs.map(([k]) => k), 'embed', 'embedScale', ...(stream ? ['posEmbed'] : []), 'vocab'];
    const loaded = {};
    const totals = {};
    for (const k of keys) totals[k] = spec.bytes?.[k] ?? 0;
    const report = () => {
      const l = keys.reduce((s, k) => s + (loaded[k] ?? 0), 0);
      const t = keys.reduce((s, k) => s + (totals[k] ?? 0), 0);
      emit({ stage: 'download', fraction: t ? Math.min(1, l / t) : 0, loaded: l, total: t });
    };
    report();
    const files = Object.fromEntries(
      await Promise.all(
        keys.map(async (k) => [
          k,
          await fetchWithProgress(url(spec.files[k]), {
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

    const ready = {};
    try {
      for (let i = 0; i < programs.length; i++) {
        const [key, label] = programs[i];
        emit({ stage: 'gpu', fraction: i / programs.length, detail: label });
        ready[key] = await withRuntime((et) => Program.load(et, files[key], label));
        files[key] = null;
        if (signal.aborted) throw aborted();
      }
    } catch (err) {
      withRuntime(() => Object.values(ready).forEach((p) => p.delete()));
      throw err;
    }
    emit({ stage: 'gpu', fraction: 1 });

    const vocab = JSON.parse(new TextDecoder().decode(files.vocab));
    return new WhisperRecognizer(ready, {
      embed: new Int8Array(files.embed.buffer, files.embed.byteOffset, files.embed.byteLength),
      embedScale: floats(files.embedScale),
      posEmbed: stream ? floats(files.posEmbed) : null,
      detok: new WhisperDetokenizer(vocab),
    });
  }

  constructor(programs, { embed, embedScale, posEmbed, detok }) {
    this.p = programs;
    this.embed = embed;
    this.embedScale = embedScale;
    this.posEmbed = posEmbed;
    this.detok = detok;
    this.stream = Boolean(programs.decodeStream);
    this.hidden = embed.length / embedScale.length;
    this.windowSec = Math.max(8, Math.min(30, (MAX_DEC - 4) / 4));
    this.frames = new Float32Array(FRAMES * N_FFT);
    this.emb = new Float32Array((this.stream ? 1 : MAX_DEC) * this.hidden);
    this.busy = Promise.resolve();
  }

  get info() {
    return { model: WHISPER_MODEL_ID, streaming: this.stream, windowSec: this.windowSec, maxTokens: MAX_DEC };
  }

  transcribe(audio, opts = {}) {
    const run = this.busy.then(() => this.#transcribe(audio, opts));
    this.busy = run.catch(() => {});
    return run;
  }

  async #transcribe(audio, { onPartial, signal, language = 'en' } = {}) {
    const t0 = performance.now();
    const lang = (WHISPER_LANGUAGES.find((l) => l.code === language) ?? WHISPER_LANGUAGES[0]).id;
    const windows = planWindows(audio, { maxSec: this.windowSec });
    const segments = [];
    const joined = (extra = '') => [...segments.map((s) => s.text), extra].filter(Boolean).join(' ');
    for (let i = 0; i < windows.length; i++) {
      if (signal?.aborted) throw aborted();
      const { start, end } = windows[i];
      if (rms(audio, start, end) < SILENCE_RMS) continue;
      const text = await this.#window(audio.subarray(start, end), lang, signal, (partial) => onPartial?.(joined(partial), { window: i, windows: windows.length }));
      if (text) segments.push({ start: start / SPEECH_RATE, end: end / SPEECH_RATE, text });
      onPartial?.(joined(), { window: i + 1, windows: windows.length });
    }
    return { text: joined(), ms: performance.now() - t0, audioSec: audio.length / SPEECH_RATE, segments };
  }

  // Token embedding (dequantized int8 row), plus the positional row when pos >= 0.
  #writeEmbed(id, pos, dst, at) {
    const H = this.hidden;
    const s = id * H;
    const sc = this.embedScale[id];
    if (pos < 0) {
      for (let j = 0; j < H; j++) dst[at + j] = this.embed[s + j] * sc;
      return;
    }
    const p = pos * H;
    for (let j = 0; j < H; j++) dst[at + j] = this.embed[s + j] * sc + this.posEmbed[p + j];
  }

  async #window(pcm, lang, signal, onPartial) {
    const H = this.hidden;
    buildFrames(pcm, this.frames);
    // Encoder and cross-attention K/V once per window; the K/V stay in the
    // runtime as inputs to every decoder step.
    const kv = await withRuntime(async (et) => {
      const [power] = await once(this.p.mel, [f32(et, [1, FRAMES, N_FFT], this.frames)]);
      const [hidden] = await once(this.p.encoder, [f32(et, [1, N_MELS, FRAMES], logMel(power.data))]);
      const outs = await once(this.p.crossKv, [f32(et, hidden.shape, hidden.data)]);
      return outs.map((o) => f32(et, o.shape, o.data));
    });

    const ids = [TOK.SOT, lang, TOK.TRANSCRIBE, TOK.NOTIMESTAMPS];
    const prefix = ids.length;
    const emit = () => onPartial(this.detok.decode(ids.slice(prefix), { partial: true }).trim());
    try {
      for (let pos = this.stream ? 0 : prefix - 1; pos < MAX_DEC; pos++) {
        let row;
        if (this.stream) {
          this.#writeEmbed(ids[pos], pos, this.emb, 0);
        } else {
          for (let k = 0; k <= pos; k++) this.#writeEmbed(ids[k], -1, this.emb, k * H);
          this.emb.fill(0, (pos + 1) * H);
        }
        const logits = await withRuntime(async (et) => {
          if (this.stream) {
            [row] = await once(this.p.decodeStream, [f32(et, [1, 1, H], this.emb), i64(et, [1], [pos])], kv);
            if (pos < prefix - 1) return null;
          } else {
            const [all] = await once(this.p.decoder, [f32(et, [1, MAX_DEC, H], this.emb)], kv);
            row = { data: all.data.subarray(pos * H, (pos + 1) * H) };
          }
          const [out] = await once(this.p.lmHead, [f32(et, [1, 1, H], row.data)]);
          return out.data;
        });
        if (!logits) continue;
        const next = argmax(logits, TOK.VOCAB);
        if (next === TOK.EOT) break;
        ids.push(next);
        emit();
        if (ids.length >= MAX_DEC || isLooping(ids.slice(prefix)) || signal?.aborted) break;
      }
    } finally {
      withRuntime(() => kv.forEach((t) => t.delete()));
    }
    return this.detok.decode(trimLoop(ids.slice(prefix))).trim();
  }
}

let ready = null;
let pending = null;
let settled = Promise.resolve();

function startLoad() {
  const job = { listeners: new Set(), last: null, callers: 0, controller: new AbortController() };
  const emit = (e) => {
    job.last = e;
    for (const l of job.listeners) l(e);
  };
  // A retry after an abort must not touch the runtime until the aborted load's
  // in-flight call has returned: the runtime is single-threaded and shared.
  job.promise = settled.then(() => WhisperRecognizer.load(emit, job.controller.signal)).then(
    (r) => (ready = r),
    (err) => {
      if (pending === job) pending = null;
      throw err;
    },
  );
  settled = job.promise.catch(() => {});
  return job;
}

/** Loads (once) and returns the shared recognizer. See the module comment. */
export async function loadWhisper({ onProgress, signal } = {}) {
  if (ready) return ready;
  if (signal?.aborted) throw aborted();
  pending ??= startLoad();
  const job = pending;
  if (onProgress) {
    job.listeners.add(onProgress);
    if (job.last) onProgress(job.last);
  }
  job.callers++;
  let onAbort = null;
  try {
    return await new Promise((resolve, reject) => {
      job.promise.then(resolve, reject);
      onAbort = () => reject(aborted());
      signal?.addEventListener('abort', onAbort, { once: true });
    });
  } finally {
    if (onAbort) signal?.removeEventListener('abort', onAbort);
    if (onProgress) job.listeners.delete(onProgress);
    job.callers--;
    if (signal?.aborted && job.callers === 0 && !ready) {
      job.controller.abort();
      if (pending === job) pending = null;
    }
  }
}

/** True when a model source resolves (dev assets on loopback, or cleared hosting). */
export async function isWhisperAvailable() {
  try {
    const source = await resolveModelSource(getModel(WHISPER_MODEL_ID));
    return source.kind !== 'unavailable';
  } catch {
    return false;
  }
}

export function isWhisperLoaded() {
  return Boolean(ready);
}
