import assert from 'node:assert/strict';
import test from 'node:test';
import { firstTurnIds, followUpIds, normalizeRGBA } from '../src/runtime/vlm.js';

const META = { n_image_tokens: 64, image_token_id: 49190, rescale: 1 / 255, mean: 0.5, std: 0.5 };

test('first turn wraps 64 image placeholders between the user tag and the question', () => {
  const { ids, imageStart } = firstTurnIds(META, [11, 12]);
  const expected = [1, 11126, 42, 49189, 49152, ...Array(64).fill(49190), 49189, 11, 12, 49279, 198, 9519, 9531, 42];
  assert.deepEqual(ids, expected);
  assert.equal(ids[imageStart], 49190);
  assert.equal(ids[imageStart - 1], 49152);
  assert.equal(ids[imageStart + 64], 49189);
});

test('a short question fits the 83-row batched prefill', () => {
  assert.ok(firstTurnIds(META, Array(8).fill(5)).ids.length <= 83);
});

test('follow-ups close the previous answer, feeding any token not yet in the cache first', () => {
  assert.deepEqual(followUpIds(null, [7]), [49279, 7, 49279, 198, 9519, 9531, 42]);
  assert.deepEqual(followUpIds(99, [7]), [99, 49279, 7, 49279, 198, 9519, 9531, 42]);
});

test('pixels become planar RGB in [-1, 1]', () => {
  const rgba = new Uint8ClampedArray([0, 255, 51, 255, 255, 0, 204, 255]);
  const px = normalizeRGBA(rgba, 2, META);
  const r = (x) => Math.round(x * 1000) / 1000;
  assert.deepEqual(Array.from(px, r), [-1, 1, 1, -1, -0.6, 0.6]);
});
