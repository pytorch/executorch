import assert from 'node:assert/strict';
import test from 'node:test';
import { colormapLut } from '../src/vision/colormap.js';
import { COCO_CLASSES } from '../src/vision/coco.js';
import { decodeDetections, nms } from '../src/vision/detect.js';
import { tilePlan } from '../src/vision/superres.js';

test('tile plans cover every pixel exactly once and stay inside the image', () => {
  for (const length of [64, 65, 100, 128, 200, 256, 257]) {
    const plan = tilePlan(length, 64, 8);
    let next = 0;
    for (const t of plan) {
      assert.ok(t.at >= 0 && t.at + 64 <= Math.max(64, length), `tile at ${t.at} fits ${length}`);
      assert.equal(t.from, next);
      assert.ok(t.from >= t.at && t.to <= t.at + 64, 'interior lies inside its tile');
      next = t.to;
    }
    assert.equal(next, length);
  }
});

test('interior seams keep at least the overlap margin from tile edges', () => {
  const plan = tilePlan(256, 64, 8);
  for (const [i, t] of plan.entries()) {
    if (i > 0) assert.ok(t.from - t.at >= 4);
    if (i < plan.length - 1) assert.ok(t.at + 64 - t.to >= 4);
  }
});

test('detections decode from the [1, 84, anchors] layout and undo the letterbox', () => {
  const anchors = 3;
  const data = new Float32Array(84 * anchors);
  const set = (row, a, v) => (data[row * anchors + a] = v);
  // anchor 1: a dog (class 16) centered at (320, 200) in model pixels, 100 x 50
  set(0, 1, 320); set(1, 1, 200); set(2, 1, 100); set(3, 1, 50); set(4 + 16, 1, 0.9);
  set(4 + 0, 0, 0.01);
  const boxes = decodeDetections(data, [1, 84, anchors], { floor: 0.1, scale: [0.5, 0.5], offset: [0, 80] });
  assert.equal(boxes.length, 1);
  assert.equal(COCO_CLASSES[boxes[0].cls], 'dog');
  assert.deepEqual([boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h], [540, 190, 200, 100]);
});

test('NMS drops same-class overlaps but keeps other classes', () => {
  const a = { x: 0, y: 0, w: 10, h: 10, score: 0.9, cls: 0 };
  const b = { x: 1, y: 1, w: 10, h: 10, score: 0.8, cls: 0 };
  const c = { x: 1, y: 1, w: 10, h: 10, score: 0.7, cls: 2 };
  assert.deepEqual(nms([a, b, c]), [a, c]);
});

test('colormaps span dark to bright', () => {
  for (const name of ['turbo', 'inferno', 'gray']) {
    const lut = colormapLut(name);
    assert.equal(lut.length, 768);
    const lum = (i) => lut[i * 3] * 0.3 + lut[i * 3 + 1] * 0.59 + lut[i * 3 + 2] * 0.11;
    assert.ok(lum(0) < 80, `${name} starts dark`);
    assert.ok(lum(160) > lum(0), `${name} brightens`);
  }
});
