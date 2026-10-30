import assert from 'node:assert/strict';
import test from 'node:test';
import { bestMask, composeMasks, encodePromptTokens, LABEL, maskContours, planPrompt, scoreMasks, upsampleMask } from '../src/runtime/sam.js';

const disk = (w, h, cx, cy, r) => Float32Array.from({ length: w * h }, (_, i) => r - Math.hypot((i % w) - cx, Math.floor(i / w) - cy));

test('a fixed two-token decoder gets one decode per include point, paired with each exclude', () => {
  const points = [
    { x: 10, y: 10, label: LABEL.POSITIVE },
    { x: 20, y: 20, label: LABEL.POSITIVE },
    { x: 5, y: 5, label: LABEL.NEGATIVE },
  ];
  const units = planPrompt({ points, fixedTokens: 2 });
  assert.equal(units.length, 2);
  for (const u of units) {
    assert.equal(u.keep.length, 1);
    assert.deepEqual(u.keep[0].map((t) => t.kind), ['pos', 'neg']);
  }
  assert.deepEqual(planPrompt({ points: [points[0]], fixedTokens: 2 })[0].keep[0].map((t) => t.kind), ['pos', 'pad']);
});

test('a box decodes alone and exclude points cut from it; exclude-only prompts decode nothing', () => {
  const units = planPrompt({ box: { x0: 30, y0: 40, x1: 10, y1: 5 }, points: [{ x: 1, y: 2, label: LABEL.NEGATIVE }], fixedTokens: 2 });
  assert.equal(units.length, 1);
  assert.deepEqual(units[0].keep[0], [{ kind: 'tl', x: 10, y: 5 }, { kind: 'br', x: 30, y: 40 }]);
  assert.deepEqual(units[0].cut[0].map((t) => t.kind), ['pos', 'pad']);
  assert.deepEqual(planPrompt({ points: [{ x: 1, y: 1, label: LABEL.NEGATIVE }], fixedTokens: 2 }), []);
});

test('a flexible decoder gets SAM’s native prompt with one padding token, boxes included', () => {
  const units = planPrompt({ points: [{ x: 1, y: 1, label: 1 }, { x: 2, y: 2, label: 0 }] });
  assert.deepEqual(units[0].keep[0].map((t) => t.kind), ['pos', 'neg', 'pad']);
  assert.deepEqual(planPrompt({ box: { x0: 0, y0: 0, x1: 4, y1: 4 } })[0].keep[0].map((t) => t.kind), ['tl', 'br', 'pad']);
});

test('a three-token decoder decodes a box or a click with an exclude natively, padded to size', () => {
  const kinds = (args) => planPrompt({ ...args, fixedTokens: 3 }).map((u) => u.keep.map((k) => k.map((t) => t.kind)));
  assert.deepEqual(kinds({ box: { x0: 0, y0: 0, x1: 4, y1: 4 } }), [[['tl', 'br', 'pad']]]);
  assert.deepEqual(kinds({ points: [{ x: 1, y: 1, label: 1 }, { x: 2, y: 2, label: 0 }] }), [[['pos', 'neg', 'pad']]]);
  assert.deepEqual(kinds({ points: [{ x: 1, y: 1, label: 1 }] }), [[['pos', 'pad', 'pad']]]);
});

test('prompt tokens add the kind embedding to the Fourier encoding; padding is the bare embedding', () => {
  const hidden = 4;
  const gaussian = new Float32Array([0, 0, 0, 0]);
  const embeds = [0, 1, 2, 3].map((k) => new Float32Array(hidden).fill(k * 10));
  const notAPoint = new Float32Array([7, 7, 7, 7]);
  const data = encodePromptTokens([{ kind: 'pos', x: 3, y: 4 }, { kind: 'pad' }], { width: 8, height: 8, size: 8, hidden, gaussian, embeds, notAPoint });
  // With a zero projection, sin = 0 and cos = 1 regardless of the point.
  assert.deepEqual([...data.subarray(0, 4)], [10, 10, 11, 11]);
  assert.deepEqual([...data.subarray(4)], [7, 7, 7, 7]);
});

test('mask scoring prefers a stable object over noise and ignores empty masks', () => {
  const w = 32;
  const plane = w * w;
  const all = new Float32Array(3 * plane);
  all.fill(-5, 0, plane);
  all.set(disk(w, w, 16, 16, 9).map((v) => v * 4), plane);
  for (let i = 0; i < plane; i++) all[2 * plane + i] = Math.sin(i * 1.7) * 0.8;
  const c = scoreMasks(all, 3, plane);
  assert.equal(c[0].area, 0);
  assert.ok(c[1].stability > 0.8);
  assert.equal(bestMask(c), 1);
});

test('composition unions units and carves cuts in logit space', () => {
  const w = 16;
  const plane = w * w;
  const unit = (logits) => ({ all: logits, n: 1, lw: w, lh: w, candidates: scoreMasks(logits, 1, plane) });
  const a = unit(disk(w, w, 4, 8, 3));
  const b = unit(disk(w, w, 12, 8, 3));
  const cut = unit(disk(w, w, 12, 8, 1.5));
  const r = composeMasks([{ keep: [a], cut: [] }, { keep: [b], cut: [cut] }], { width: 32, height: 32 });
  const at = (x, y) => r.logits[y * w + x] > 0;
  assert.ok(at(4, 8) && at(14, 8), 'both disks present');
  assert.ok(!at(12, 8), 'centre of the second disk cut out');
  assert.equal(r.mask.length, 32 * 32);
  assert.equal(composeMasks([], { width: 4, height: 4 }).area, 0);
});

test('upsampling keeps the zero crossing in place', () => {
  const m = upsampleMask(new Float32Array([-1, 1, -1, 1]), 2, 2, 4, 2);
  assert.deepEqual([...m], [0, 0, 1, 1, 0, 0, 1, 1]);
});

test('contours close around blobs and holes with sub-cell precision', () => {
  const w = 24;
  const ring = Float32Array.from(disk(w, w, 12, 12, 8), (v, i) => Math.min(v, Math.hypot((i % w) - 12, Math.floor(i / w) - 12) - 3));
  const loops = maskContours(ring, w, w);
  assert.equal(loops.length, 2);
  const radii = loops.map((l) => l.reduce((s, [x, y]) => s + Math.hypot(x - 12, y - 12), 0) / l.length).sort((p, q) => p - q);
  assert.ok(Math.abs(radii[0] - 3) < 0.25 && Math.abs(radii[1] - 8) < 0.25, `radii ${radii}`);
  // A blob touching the border still closes.
  assert.equal(maskContours(new Float32Array(16).fill(1), 4, 4).length, 1);
});
