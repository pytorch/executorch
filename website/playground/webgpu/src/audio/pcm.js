// Pure PCM helpers (no DOM), shared by the mic, file decoding and the
// recognizer. Everything downstream of capture is 16 kHz mono Float32.
export const SPEECH_RATE = 16000;

// Band-limited resampling: Hann-windowed sinc, normalised to unity DC gain.
// When downsampling the cutoff tracks the target Nyquist so nothing aliases.
export function resample(input, fromRate, toRate = SPEECH_RATE) {
  if (fromRate === toRate) return Float32Array.from(input);
  const ratio = fromRate / toRate;
  const out = new Float32Array(Math.floor(input.length / ratio));
  const cutoff = Math.min(1, 1 / ratio) * 0.94;
  const half = Math.ceil(8 * Math.max(1, ratio));
  const last = input.length - 1;
  for (let i = 0; i < out.length; i++) {
    const center = i * ratio;
    const lo = Math.max(0, Math.ceil(center - half));
    const hi = Math.min(last, Math.floor(center + half));
    let acc = 0;
    let wsum = 0;
    for (let j = lo; j <= hi; j++) {
      const x = j - center;
      const t = Math.PI * cutoff * x;
      const k = (x === 0 ? 1 : Math.sin(t) / t) * (0.5 + 0.5 * Math.cos((Math.PI * x) / half));
      acc += input[j] * k;
      wsum += k;
    }
    out[i] = wsum ? acc / wsum : 0;
  }
  return out;
}

export function downmix(channels) {
  if (channels.length === 1) return Float32Array.from(channels[0]);
  const out = new Float32Array(channels[0].length);
  for (const ch of channels) for (let i = 0; i < out.length; i++) out[i] += ch[i];
  for (let i = 0; i < out.length; i++) out[i] /= channels.length;
  return out;
}

export function concat(chunks) {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

export function rms(samples, start = 0, end = samples.length) {
  let s = 0;
  for (let i = start; i < end; i++) s += samples[i] * samples[i];
  return end > start ? Math.sqrt(s / (end - start)) : 0;
}

// Splits audio into windows of at most `maxSec`, cutting at the quietest
// `frameMs` frame in the last stretch of each window so words stay whole.
export function planWindows(samples, { rate = SPEECH_RATE, maxSec = 15, minFrac = 0.6, frameMs = 40 } = {}) {
  const length = samples.length;
  const max = Math.round(maxSec * rate);
  const min = Math.round(max * minFrac);
  const frame = Math.max(1, Math.round((frameMs / 1000) * rate));
  const out = [];
  let pos = 0;
  while (length - pos > max) {
    let cut = pos + max;
    let best = Infinity;
    for (let f = pos + min; f + frame <= pos + max; f += frame) {
      const e = rms(samples, f, f + frame);
      if (e < best) {
        best = e;
        cut = f + (frame >> 1);
      }
    }
    out.push({ start: pos, end: cut });
    pos = cut;
  }
  if (length > pos) out.push({ start: pos, end: length });
  return out;
}

// Peak envelope in `buckets` columns, for drawing a waveform.
export function peaks(samples, buckets) {
  const out = new Float32Array(buckets);
  const step = samples.length / buckets;
  for (let b = 0; b < buckets; b++) {
    const s = Math.floor(b * step);
    const e = Math.max(s + 1, Math.floor((b + 1) * step));
    let p = 0;
    for (let i = s; i < e && i < samples.length; i++) p = Math.max(p, Math.abs(samples[i]));
    out[b] = p;
  }
  return out;
}

// 16-bit PCM WAV, for playing a clip back.
export function encodeWav(samples, rate = SPEECH_RATE) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF');
  v.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, 'data');
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const x = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true);
  }
  return new Uint8Array(buf);
}

export function formatClock(seconds, { tenths = false } = {}) {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  const sec = tenths ? rest.toFixed(1).padStart(4, '0') : String(Math.floor(rest)).padStart(2, '0');
  return `${String(m).padStart(2, '0')}:${sec}`;
}
