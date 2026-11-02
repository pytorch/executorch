import assert from 'node:assert/strict';
import test from 'node:test';
import { checkWiring } from '../scripts/check-wiring.mjs';

test('imports resolve to real exports and every applied class is styled', () => {
  const { problems } = checkWiring();
  assert.deepEqual(problems, []);
});
