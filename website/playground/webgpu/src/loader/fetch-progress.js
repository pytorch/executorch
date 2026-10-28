// Streaming fetch with byte-level progress and a Cache Storage tier, so each
// model file downloads at most once per device. Every hosted model file, and the
// runtime's .wasm, takes this path.

export const MODEL_CACHE = 'executorch-webgpu-models-v1';

// Expected SHA-256 per URL, registered by the model source for hosted files.
// A download that doesn't match is rejected before it is cached or used.
const expected = new Map();

export function expectSha256(url, hex) {
  expected.set(url, hex.toLowerCase());
}

async function verify(url, bytes) {
  const want = expected.get(url);
  if (!want) return;
  if (!globalThis.crypto?.subtle) {
    throw new Error(`Cannot verify ${url}: this page is not a secure context (HTTPS or localhost)`);
  }
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const got = Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
  if (got !== want) throw new Error(`Checksum mismatch for ${url}: the download is corrupt or was changed`);
}

export async function fetchWithProgress(url, { onProgress, signal, expectedBytes = 0, deps = {} } = {}) {
  const fetchImpl = deps.fetch ?? globalThis.fetch;
  const cacheStorage = deps.caches ?? globalThis.caches;
  const cache = cacheStorage ? await cacheStorage.open(MODEL_CACHE) : null;

  const hit = cache ? await cache.match(url) : undefined;
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    onProgress?.({ loaded: bytes.byteLength, total: bytes.byteLength, fromCache: true });
    return bytes;
  }

  const response = await fetchImpl(url, { signal });
  if (!response.ok) throw new Error(`Download failed: HTTP ${response.status} for ${url}`);

  // Content-Length is the compressed size when the server applies Content-Encoding,
  // so it is only a hint; an overrun switches the bar to indeterminate.
  const headerBytes = Number(response.headers.get('content-length')) || 0;
  let total = headerBytes || expectedBytes;
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    if (total && loaded > total) total = 0;
    onProgress?.({ loaded, total, fromCache: false });
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.byteLength;
  }
  onProgress?.({ loaded, total: loaded, fromCache: false });
  await verify(url, bytes);

  if (cache) {
    // Caching is best-effort: a full quota or a private window must not fail a
    // download that already succeeded.
    try {
      await cache.put(
        url,
        new Response(bytes, {
          headers: {
            'content-type': response.headers.get('content-type') ?? 'application/octet-stream',
            'content-length': String(loaded),
          },
        }),
      );
    } catch (err) {
      console.warn(`Could not cache ${url}: ${err.message}`);
    }
  }
  return bytes;
}

export async function isCached(url, deps = {}) {
  const cacheStorage = deps.caches ?? globalThis.caches;
  if (!cacheStorage) return false;
  const cache = await cacheStorage.open(MODEL_CACHE);
  return Boolean(await cache.match(url));
}

export async function evictFromCache(urls, deps = {}) {
  const cacheStorage = deps.caches ?? globalThis.caches;
  if (!cacheStorage) return 0;
  const cache = await cacheStorage.open(MODEL_CACHE);
  let removed = 0;
  for (const url of urls) if (await cache.delete(url)) removed++;
  return removed;
}
