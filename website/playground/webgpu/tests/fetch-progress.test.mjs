import assert from 'node:assert/strict';
import test from 'node:test';
import { evictFromCache, fetchWithProgress, isCached } from '../src/loader/fetch-progress.js';

function fakeCaches() {
  const store = new Map();
  return {
    store,
    open: async () => ({
      match: async (url) => store.get(url),
      put: async (url, res) => store.set(url, new Response(await res.arrayBuffer(), { headers: res.headers })),
      delete: async (url) => store.delete(url),
    }),
  };
}

function fakeFetch(bytes, { chunk = 1000, contentLength = String(bytes.length) } = {}) {
  let calls = 0;
  const fn = async () => {
    calls++;
    let offset = 0;
    const body = new ReadableStream({
      pull(c) {
        if (offset >= bytes.length) return c.close();
        c.enqueue(bytes.slice(offset, offset + chunk));
        offset += chunk;
      },
    });
    return new Response(body, { headers: contentLength ? { 'content-length': contentLength } : {} });
  };
  return { fn, calls: () => calls };
}

const payload = Uint8Array.from({ length: 4500 }, (_, i) => i % 251);

test('streams with monotonic byte-level progress and caches the result', async () => {
  const caches = fakeCaches();
  const net = fakeFetch(payload);
  const events = [];
  const bytes = await fetchWithProgress('https://example.org/m.pte', { onProgress: (e) => events.push(e), deps: { fetch: net.fn, caches } }); // @lint-ignore (test fixture, not a real link)
  assert.deepEqual(bytes, payload);
  assert.ok(events.length >= 5);
  for (let i = 1; i < events.length; i++) assert.ok(events[i].loaded >= events[i - 1].loaded);
  assert.equal(events.at(-1).loaded, payload.length);
  assert.equal(events.at(-1).total, payload.length);
  assert.equal(await isCached('https://example.org/m.pte', { caches }), true); // @lint-ignore (test fixture, not a real link)
});

test('second request is served from Cache Storage without touching the network', async () => {
  const caches = fakeCaches();
  const net = fakeFetch(payload);
  await fetchWithProgress('https://example.org/m.pte', { deps: { fetch: net.fn, caches } }); // @lint-ignore (test fixture, not a real link)
  const events = [];
  const again = await fetchWithProgress('https://example.org/m.pte', { onProgress: (e) => events.push(e), deps: { fetch: net.fn, caches } }); // @lint-ignore (test fixture, not a real link)
  assert.equal(net.calls(), 1);
  assert.deepEqual(again, payload);
  assert.equal(events[0].fromCache, true);
});

test('a compressed Content-Length overrun switches to indeterminate instead of >100%', async () => {
  const events = [];
  await fetchWithProgress('https://example.org/x', { onProgress: (e) => events.push(e), deps: { fetch: fakeFetch(payload, { contentLength: '2000' }).fn, caches: fakeCaches() } }); // @lint-ignore (test fixture, not a real link)
  for (const e of events.slice(0, -1)) assert.ok(e.total === 0 || e.loaded <= e.total);
});

test('HTTP errors throw and nothing is cached', async () => {
  const caches = fakeCaches();
  await assert.rejects(fetchWithProgress('https://example.org/404', { deps: { fetch: async () => new Response('no', { status: 404 }), caches } })); // @lint-ignore (test fixture, not a real link)
  assert.equal(caches.store.size, 0);
});

test('eviction removes cached files (clearing a model is a config change)', async () => {
  const caches = fakeCaches();
  await fetchWithProgress('https://example.org/m.pte', { deps: { fetch: fakeFetch(payload).fn, caches } }); // @lint-ignore (test fixture, not a real link)
  assert.equal(await evictFromCache(['https://example.org/m.pte'], { caches }), 1); // @lint-ignore (test fixture, not a real link)
  assert.equal(await isCached('https://example.org/m.pte', { caches }), false); // @lint-ignore (test fixture, not a real link)
});
