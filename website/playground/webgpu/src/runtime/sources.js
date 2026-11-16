// Where a real ExecuTorch model's files come from.
//
// Shipped builds resolve weights only through resolveWeights() (license cleared
// AND an https URL). For local review there is a second source: the dev server's
// /__dev__/config.json, which maps model ids to an `assets` block shaped like the
// one in models.js, with `baseUrl` relative to the directory the dev server
// serves from outside the site. A dev entry wins over the hosted one, so a new
// export can be tried before it is published. Production hosts have no such
// route, and the lookup is skipped unless the page is on a loopback host or the
// dev server marked it. Either way the runtime itself is this site's runtime/.
import { expectSha256 } from '../loader/fetch-progress.js';
import { resolveWeights, WEIGHT_SOURCE } from '../loader/resolve.js';
import { SHARED } from '../config/shared.js';

const LOOPBACK = new Set(['localhost', '127.0.0.1', '[::1]']);
let devConfig = null;

export function getDevConfig() {
  if (!devConfig) {
    const page = typeof document !== 'undefined';
    const marked = page && document.querySelector('meta[name="executorch-dev-assets"]');
    devConfig = page && (LOOPBACK.has(location.hostname) || marked)
      ? fetch('/__dev__/config.json', { cache: 'no-store' })
          .then((r) => (r.ok ? r.json() : null))
          .catch(() => null)
      : Promise.resolve(null);
  }
  return devConfig;
}

// -> { kind, spec, url(path), shared, repoUrl? } for an `assets` block
// ({ baseUrl, spec, sha256 }) whose baseUrl is resolved against `base`.
// Registers each file's sha256 so a download that doesn't match is rejected.
export function assetsSource(kind, { baseUrl, spec, sha256 = {} }, base) {
  const root = new URL(baseUrl ?? '', base).href;
  const dir = root.endsWith('/') ? root : `${root}/`;
  const url = (path) => new URL(path, dir).href;
  for (const [path, hex] of Object.entries(sha256)) expectSha256(url(path), hex);
  const repoUrl = kind === 'remote' && dir.includes('/resolve/') ? dir.split('/resolve/')[0] : null;
  return { kind, spec, url, shared: SHARED, repoUrl };
}

// -> { kind: 'dev' | 'remote', spec, url(path), shared, repoUrl } | { kind: 'unavailable', reasons }
export async function resolveModelSource(model) {
  const dev = await getDevConfig();
  const local = dev?.models?.[model.id];
  if (local?.spec) return assetsSource('dev', local, new URL(dev.base, location.origin));
  const weights = resolveWeights(model);
  if (weights.source !== WEIGHT_SOURCE.REMOTE) return { kind: 'unavailable', reasons: weights.reasons };
  if (!model.assets.spec) return { kind: 'unavailable', reasons: [{ code: 'no-spec', text: 'This model has no ExecuTorch program description yet' }] };
  return assetsSource('remote', model.assets);
}
