import assert from 'node:assert/strict';
import test from 'node:test';
import { renderMarkdown } from '../src/lib/markdown.js';
import { makeSampler } from '../src/runtime/llm.js';

test('greedy sampling picks the argmax', () => {
  const pick = makeSampler({ temperature: 0, repetitionPenalty: 1 });
  assert.equal(pick(new Float32Array([0.1, 3, -2, 2.9]), []), 1);
});

test('sampling ignores non-finite logits and reports when nothing is usable', () => {
  const pick = makeSampler({ temperature: 0.7, topK: 4, topP: 1, repetitionPenalty: 1 });
  for (let i = 0; i < 50; i++) assert.equal(pick(new Float32Array([NaN, 5, -Infinity, NaN]), []), 1);
  assert.equal(pick(new Float32Array([NaN, NaN]), []), -1);
});

test('top-p keeps only the head of the distribution', () => {
  const pick = makeSampler({ temperature: 1, topK: 10, topP: 0.5, repetitionPenalty: 1 });
  const logits = () => new Float32Array([10, 0, 0, 0, 0]);
  for (let i = 0; i < 50; i++) assert.equal(pick(logits(), []), 0);
});

test('repetition penalty pushes recently used tokens down', () => {
  const pick = makeSampler({ temperature: 0, repetitionPenalty: 2 });
  assert.equal(pick(new Float32Array([4, 3]), [0]), 1);
});

test('markdown escapes HTML and renders the supported subset', () => {
  const html = renderMarkdown('**Hi** <img src=x onerror=alert(1)>\n\n- one\n- `two`\n\n```js\nlet a = "<b>";\n```');
  assert.ok(!html.includes('<img'));
  assert.ok(html.includes('&lt;img'));
  assert.ok(html.includes('<strong>Hi</strong>'));
  assert.ok(html.includes('<ul><li>one</li><li><code>two</code></li></ul>'));
  assert.ok(html.includes('<pre class="md-pre"><code>let a = &quot;&lt;b&gt;&quot;;</code></pre>'));
});

test('a download whose SHA-256 does not match is rejected and not cached', async () => {
  const { expectSha256, fetchWithProgress } = await import('../src/loader/fetch-progress.js');
  const body = new TextEncoder().encode('weights');
  const put = [];
  const deps = {
    fetch: async () => new Response(body, { headers: { 'content-length': String(body.length) } }),
    caches: { open: async () => ({ match: async () => undefined, put: async (u) => put.push(u) }) },
  };
  expectSha256('https://example.org/m/a.pte', '0'.repeat(64)); // @lint-ignore (test fixture, not a real link)
  await assert.rejects(fetchWithProgress('https://example.org/m/a.pte', { deps }), /Checksum mismatch/); // @lint-ignore (test fixture, not a real link)
  assert.deepEqual(put, []);
  // sha256("weights")
  expectSha256('https://example.org/m/b.pte', '9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c'); // @lint-ignore (test fixture, not a real link)
  const bytes = await fetchWithProgress('https://example.org/m/b.pte', { deps }); // @lint-ignore (test fixture, not a real link)
  assert.equal(new TextDecoder().decode(bytes), 'weights');
  assert.deepEqual(put, ['https://example.org/m/b.pte']); // @lint-ignore (test fixture, not a real link)
});

test('a cleared model loads from its pinned repo with the site runtime and registers checksums', async () => {
  const { resolveModelSource } = await import('../src/runtime/sources.js');
  const { fetchWithProgress } = await import('../src/loader/fetch-progress.js');
  const base = 'https://huggingface.co/org/Model-ExecuTorch-WebGPU/resolve/abc123/'; // @lint-ignore (test fixture, not a real link)
  const model = {
    id: 'm',
    license: { status: 'cleared' },
    assets: { baseUrl: base, files: ['tokenizer/tokenizer.json', 'model.pte'], sha256: { 'model.pte': 'ab'.repeat(32) }, spec: { pte: 'model.pte', tokenizer: 'tokenizer' } },
  };
  const src = await resolveModelSource(model);
  assert.equal(src.kind, 'remote');
  assert.equal(src.url('model.pte'), `${base}model.pte`);
  assert.equal(src.url('tokenizer/tokenizer.json'), `${base}tokenizer/tokenizer.json`);
  assert.equal(src.repoUrl, 'https://huggingface.co/org/Model-ExecuTorch-WebGPU'); // @lint-ignore (test fixture, not a real link)
  const deps = { fetch: async () => new Response(new Uint8Array([1])), caches: { open: async () => ({ match: async () => undefined, put: async () => {} }) } };
  await assert.rejects(fetchWithProgress(`${base}model.pte`, { deps }), /Checksum mismatch/);
  const bare = await resolveModelSource({ ...model, assets: { ...model.assets, spec: null } });
  assert.deepEqual(bare.reasons.map((r) => r.code), ['no-spec']);
  assert.equal((await resolveModelSource({ ...model, license: { status: 'todo' } })).kind, 'unavailable');
});

test('a dev config entry has the models.js assets shape, relative to the dev server', async () => {
  const { assetsSource } = await import('../src/runtime/sources.js');
  const src = assetsSource('dev', { baseUrl: 'reexport/model/', spec: { pte: 'model.pte' } }, 'http://localhost:5173/__dev_assets__/');
  assert.equal(src.kind, 'dev');
  assert.equal(src.url(src.spec.pte), 'http://localhost:5173/__dev_assets__/reexport/model/model.pte');
  assert.equal(src.repoUrl, null);
});
