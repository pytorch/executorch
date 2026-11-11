// Imports an ES module from a pinned URL after checking its SHA-256. The bytes go
// through the same verified, cached download as model files, then load from a
// blob: URL, so the module's MIME type doesn't depend on the host. A blob: URL
// can't be the base of a relative URL, so the module's import.meta.url is
// pointed back at the URL it was downloaded from.
import { expectSha256, fetchWithProgress } from './fetch-progress.js';

const modules = new Map();

export function importVerified({ url, sha256, bytes }) {
  if (!modules.has(url)) {
    const loading = (async () => {
      expectSha256(url, sha256);
      const code = await fetchWithProgress(url, { expectedBytes: bytes });
      const source = new TextDecoder().decode(code).replaceAll('import.meta.url', JSON.stringify(url));
      const blobUrl = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
      try {
        return await import(blobUrl);
      } finally {
        URL.revokeObjectURL(blobUrl);
      }
    })();
    loading.catch(() => modules.delete(url));
    modules.set(url, loading);
  }
  return modules.get(url);
}
