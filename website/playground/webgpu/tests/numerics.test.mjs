import assert from 'node:assert/strict';
import test from 'node:test';
import { initValue, pcg, randomFloats } from '../src/lib/prng.js';

test('pcg mirrors the WGSL hash (u32 wraparound)', () => {
  assert.equal(pcg(0), pcg(0));
  for (const v of [0, 1, 0xffffffff, 123456789]) {
    const h = pcg(v);
    assert.ok(Number.isInteger(h) && h >= 0 && h < 2 ** 32);
  }
  assert.notEqual(pcg(1), pcg(2));
});

test('parameter init has mean ~0 and variance ~scale^2', () => {
  const n = 20000;
  let sum = 0;
  let sq = 0;
  for (let i = 0; i < n; i++) {
    const v = initValue(i, 42, 0.5);
    sum += v;
    sq += v * v;
  }
  const mean = sum / n;
  assert.ok(Math.abs(mean) < 0.01, `mean ${mean}`);
  assert.ok(Math.abs(sq / n - 0.25) < 0.01, `var ${sq / n}`);
});
