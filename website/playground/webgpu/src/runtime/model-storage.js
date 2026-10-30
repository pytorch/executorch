// What each model has stored on this device, and removing it. Model files live
// in the browser's Cache Storage (written by fetchWithProgress), keyed by URL.
import { evictFromCache, MODEL_CACHE } from '../loader/fetch-progress.js';
import { resolveModelSource } from './sources.js';

const FILE = /\.(pte|ptd|bin|json|txt|model|gguf|onnx|wasm)$/i;

// Runtime specs nest file paths in different shapes (a `pte` string, a `files`
// map, a list of `{ path }`), so collect every string that looks like a file.
function filePaths(value, out = new Set()) {
  if (typeof value === 'string') {
    if (FILE.test(value)) out.add(value);
  } else if (Array.isArray(value)) {
    for (const v of value) filePaths(v, out);
  } else if (value && typeof value === 'object') {
    // The runtime build is shared by several models and isn't part of any one of them.
    for (const [k, v] of Object.entries(value)) if (k !== 'runtime') filePaths(v, out);
  }
  return out;
}

export async function modelFileUrls(model) {
  const source = await resolveModelSource(model);
  if (source.kind === 'remote' || source.kind === 'dev') return [...filePaths(source.spec)].map((p) => source.url(p));
  return [];
}

// -> { urls: string[] (the ones actually cached), bytes }
export async function storedFiles(model) {
  if (!globalThis.caches) return { urls: [], bytes: 0 };
  const cache = await caches.open(MODEL_CACHE);
  const urls = [];
  let bytes = 0;
  for (const url of await modelFileUrls(model)) {
    const hit = await cache.match(url);
    if (!hit) continue;
    urls.push(url);
    bytes += Number(hit.headers.get('content-length')) || 0;
  }
  return { urls, bytes };
}

export async function deleteStoredModel(model) {
  const { urls, bytes } = await storedFiles(model);
  await evictFromCache(urls);
  return { removed: urls.length, bytes };
}

const WEIGHTS = /\.(pte|ptd|bin)$/i;

// True when every weight file the model needs is already in this browser.
export async function isOnDevice(model) {
  if (!globalThis.caches) return false;
  const urls = (await modelFileUrls(model)).filter((u) => WEIGHTS.test(new URL(u).pathname));
  if (!urls.length) return false;
  const cache = await caches.open(MODEL_CACHE);
  for (const url of urls) if (!(await cache.match(url))) return false;
  return true;
}
