import assert from 'node:assert/strict';
import test from 'node:test';
import { downmix, encodeWav, formatClock, peaks, planWindows, resample, rms } from '../src/audio/pcm.js';
import { argmax, buildFrames, isLooping, logMel, trimLoop, WhisperDetokenizer } from '../src/runtime/whisper.js';

const tone = (hz, rate, sec, amp = 0.5) => Float32Array.from({ length: Math.round(rate * sec) }, (_, i) => amp * Math.sin((2 * Math.PI * hz * i) / rate));

test('resampling keeps an in-band tone and removes an out-of-band one', () => {
  const kept = resample(tone(440, 48000, 0.5), 48000, 16000);
  assert.equal(kept.length, 8000);
  assert.ok(Math.abs(rms(kept, 500, 7500) - 0.5 / Math.SQRT2) < 0.01);
  // 12 kHz is above the 8 kHz Nyquist of the target rate: it must not alias in.
  const removed = resample(tone(12000, 48000, 0.5), 48000, 16000);
  assert.ok(rms(removed, 500, 7500) < 0.01);
  assert.deepEqual(Array.from(resample(new Float32Array([1, 2, 3]), 16000, 16000)), [1, 2, 3]);
});

test('downmix averages channels', () => {
  assert.deepEqual(Array.from(downmix([new Float32Array([1, 0]), new Float32Array([0, 1])])), [0.5, 0.5]);
});

test('planWindows cuts long audio at the quietest point and covers every sample', () => {
  const rate = 16000;
  const audio = tone(300, rate, 40);
  // A silent gap at 12.0–12.2 s should attract the first cut.
  audio.fill(0, 12 * rate, 12.2 * rate);
  const w = planWindows(audio, { maxSec: 15 });
  assert.equal(w[0].start, 0);
  assert.ok(w[0].end > 12 * rate && w[0].end < 12.2 * rate, `cut at ${w[0].end / rate}`);
  for (let i = 1; i < w.length; i++) assert.equal(w[i].start, w[i - 1].end);
  assert.equal(w.at(-1).end, audio.length);
  for (const x of w) assert.ok(x.end - x.start <= 15 * rate);
  assert.deepEqual(planWindows(new Float32Array(100)), [{ start: 0, end: 100 }]);
});

test('peaks, WAV header and clock formatting', () => {
  assert.deepEqual(Array.from(peaks(new Float32Array([0.1, -0.9, 0.2, 0.3]), 2)).map((v) => +v.toFixed(2)), [0.9, 0.3]);
  const wav = encodeWav(new Float32Array([0, 1, -1]), 16000);
  const v = new DataView(wav.buffer);
  assert.equal(String.fromCharCode(...wav.slice(0, 4)), 'RIFF');
  assert.equal(v.getUint32(24, true), 16000);
  assert.equal(v.getInt16(46, true), 32767);
  assert.equal(v.getInt16(48, true), -32768);
  assert.equal(formatClock(65.25, { tenths: true }), '01:05.3');
  assert.equal(formatClock(9), '00:09');
});

test('the byte-level detokenizer restores spaces and multi-byte text and skips specials', () => {
  // "Ġ" is the byte-level stand-in for a space; "Ã©" are the two UTF-8 bytes of "é".
  const d = new WhisperDetokenizer({ 'ĠHello': 0, ',': 1, 'Ġcaf': 2, 'Ã©': 3, 'Ã': 4 });
  assert.equal(d.decode([50258, 0, 1, 2, 3, 50257]), ' Hello, café');
  assert.equal(d.decode([2, 4], { partial: true }), ' caf');
});

test('loop guard catches short repeating tails and trims them to one instance', () => {
  assert.equal(isLooping([1, 2, 3, 4, 5, 6, 7, 8]), false);
  const looped = [9, 8, 7, 5, 6, 5, 6, 5, 6, 5, 6];
  assert.equal(isLooping(looped), true);
  assert.deepEqual(trimLoop(looped), [9, 8, 7, 5, 6]);
});

test('frames are reflect-padded hop-160 windows over a 30 s buffer', () => {
  const audio = Float32Array.from({ length: 1000 }, (_, i) => i + 1);
  const f = buildFrames(audio);
  assert.equal(f.length, 3000 * 400);
  // Frame 0 starts 200 samples before the audio, mirrored about sample 0.
  assert.equal(f[199], audio[1]);
  assert.equal(f[200], audio[0]);
  assert.equal(f[201], audio[1]);
  // Frame 1 starts at sample 160 - 200 = -40.
  assert.equal(f[400 + 40], audio[0]);
  assert.equal(f[400 + 41], audio[1]);
});

test('log-mel takes log10, clamps 8 below the peak, rescales, and transposes to [mels, frames]', () => {
  // 2 frames x 3 mels, frame-major as the mel program returns it.
  const power = new Float32Array([1, 100, 1e-12, 10, 1e-3, 0]);
  const out = logMel(power, 2, 3);
  // log10: [0, 2, -12, 1, -3, -10]; floor = 2 - 8 = -6; then (x + 4) / 4.
  const want = [[0, 1], [2, -3], [-6, -6]].flatMap(([a, b]) => [(a + 4) / 4, (Math.max(b, -6) + 4) / 4]);
  assert.deepEqual(Array.from(out), want);
});

test('argmax ignores the padded tail of the output head', () => {
  const logits = new Float32Array([0.1, 2, -1, 9]);
  assert.equal(argmax(logits), 3);
  assert.equal(argmax(logits, 3), 1);
});
