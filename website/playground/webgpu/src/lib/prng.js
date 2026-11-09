// Deterministic pseudo-random helpers shared by the CPU references, the
// tokenizer's quick-fill text, and the benchmarks.

export function hashString(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomFloats(n, seed, scale = 1) {
  const rng = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (rng() * 2 - 1) * scale;
  return out;
}

// Mirrors `pcg` in the WGSL init kernel bit-for-bit, so GPU-initialised
// parameters can be checked against the CPU.
export function pcg(v) {
  const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

// Sum of three uniforms: mean 0, variance 1. A cheap stand-in for a Gaussian.
export function initValue(index, seed, scale) {
  const s = pcg(seed);
  const i3 = Math.imul(index >>> 0, 3) >>> 0;
  const u0 = pcg((i3 + s) >>> 0) / 4294967296;
  const u1 = pcg((i3 + 1 + s) >>> 0) / 4294967296;
  const u2 = pcg((i3 + 2 + s) >>> 0) / 4294967296;
  return Math.fround((u0 + u1 + u2 - 1.5) * 2 * scale);
}
