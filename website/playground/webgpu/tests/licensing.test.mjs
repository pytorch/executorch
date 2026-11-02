import assert from 'node:assert/strict';
import test from 'node:test';
import { LICENSE_STATUS, MODELS } from '../src/config/models.js';
import { resolveWeights, WEIGHT_SOURCE } from '../src/loader/resolve.js';

const base = MODELS.find((m) => m.license.status !== LICENSE_STATUS.CLEARED) ?? MODELS[0];
const withLicense = (status, baseUrl) => ({ ...base, license: { ...base.license, status }, assets: { ...base.assets, baseUrl } });

test('every model is either cleared with a pinned, checksummed URL, or ships with no URL', () => {
  for (const m of MODELS) {
    assert.ok(m.license.upstream?.length > 0, `${m.id} needs the upstream license it would inherit`);
    assert.ok(m.license.provenance?.length > 0, `${m.id} needs a provenance note`);
    if (m.license.status === LICENSE_STATUS.CLEARED) {
      assert.match(m.assets.baseUrl, /^https:\/\/huggingface\.co\/[\w.-]+\/[\w.-]+\/resolve\/[0-9a-f]{40}\/$/, `${m.id} must pin a commit`);
      assert.ok(m.license.reviewNote, `${m.id} must record the review decision`);
      for (const f of m.assets.files) assert.match(m.assets.sha256?.[f] ?? '', /^[0-9a-f]{64}$/, `${m.id}: ${f} needs a sha256`);
      assert.ok(m.assets.spec, `${m.id} needs a runtime spec`);
    } else {
      assert.equal(m.assets.baseUrl, null, `${m.id} must not ship a download URL until cleared`);
      assert.equal(m.license.status, LICENSE_STATUS.TODO, `${m.id} must ship as license status todo`);
    }
  }
});

test('the AGPL detector is never cleared', () => {
  for (const m of MODELS.filter((x) => /AGPL/.test(x.license.upstream))) assert.notEqual(m.license.status, LICENSE_STATUS.CLEARED, m.id);
});

test('uncleared models do not resolve to remote weights', () => {
  for (const m of MODELS.filter((x) => x.license.status !== LICENSE_STATUS.CLEARED)) assert.equal(resolveWeights(m).source, WEIGHT_SOURCE.UNAVAILABLE, m.id);
});

test('a stray URL without a cleared license does not resolve', () => {
  const r = resolveWeights(withLicense(LICENSE_STATUS.TODO, 'https://example.org/models/llama/')); // @lint-ignore (test fixture, not a real link)
  assert.equal(r.source, WEIGHT_SOURCE.UNAVAILABLE);
  assert.deepEqual(r.reasons.map((x) => x.code), ['license-not-cleared']);
});

test('a cleared license without a URL does not resolve', () => {
  const r = resolveWeights(withLicense(LICENSE_STATUS.CLEARED, null));
  assert.equal(r.source, WEIGHT_SOURCE.UNAVAILABLE);
  assert.deepEqual(r.reasons.map((x) => x.code), ['no-asset-url']);
});

test('a blocked license with a URL does not resolve', () => {
  assert.equal(resolveWeights(withLicense(LICENSE_STATUS.BLOCKED, 'https://example.org/m/')).source, WEIGHT_SOURCE.UNAVAILABLE); // @lint-ignore (test fixture, not a real link)
});

test('a non-https URL does not resolve even when cleared', () => {
  assert.equal(resolveWeights(withLicense(LICENSE_STATUS.CLEARED, 'http://example.org/m/')).source, WEIGHT_SOURCE.UNAVAILABLE); // @lint-ignore (test fixture, not a real link)
});

test('cleared license AND https URL resolves to per-file URLs', () => {
  const r = resolveWeights(withLicense(LICENSE_STATUS.CLEARED, 'https://example.org/models/llama')); // @lint-ignore (test fixture, not a real link)
  assert.equal(r.source, WEIGHT_SOURCE.REMOTE);
  assert.deepEqual(r.files.map((f) => f.url), base.assets.files.map((f) => `https://example.org/models/llama/${f}`)); // @lint-ignore (test fixture, not a real link)
});
