import assert from 'node:assert/strict';
import test from 'node:test';
import { CATEGORIES, computeBadge, DEMOS } from '../src/config/catalog.js';
import { getModel } from '../src/config/models.js';

test('every demo has a compute badge generated from the table', () => {
  for (const d of DEMOS) assert.ok(computeBadge(d).label, d.id);
});

test('every category band has at least one demo, and every demo is in a band', () => {
  const ids = new Set(CATEGORIES.map((c) => c.id));
  for (const c of CATEGORIES) assert.ok(DEMOS.some((d) => d.category === c.id), c.id);
  for (const d of DEMOS) assert.ok(ids.has(d.category), d.id);
});

test('the launch tiles are present', () => {
  for (const id of ['chat', 'effects', 'segment', 'vlm', 'speech-to-text']) assert.ok(DEMOS.find((d) => d.id === id), id);
});

test('every referenced model exists in the single config; chat models carry architecture', () => {
  for (const d of DEMOS) for (const m of d.models) assert.ok(getModel(m));
  for (const m of DEMOS.find((d) => d.id === 'chat').models) assert.ok(getModel(m).arch.layers > 0);
});

test('visitors only see demos whose models are all published', async () => {
  const { DEMOS, LISTED_DEMOS, LISTED_CATEGORIES, getDemo } = await import('../src/config/catalog.js');
  const { getModel, isPublished } = await import('../src/config/models.js');
  for (const d of LISTED_DEMOS) {
    assert.ok(!d.unlisted, d.id);
    for (const id of d.models) assert.ok(isPublished(getModel(id)), `${d.id} needs unpublished ${id}`);
  }
  for (const d of DEMOS.filter((x) => !LISTED_DEMOS.includes(x))) assert.equal(getDemo(d.id), null, d.id);
  for (const c of LISTED_CATEGORIES) assert.ok(LISTED_DEMOS.some((d) => d.category === c.id), c.id);
});
