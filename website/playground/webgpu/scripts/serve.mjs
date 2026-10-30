// Zero-dependency static server for local review. Binds every interface so the
// no-WebGPU fallback can be exercised over a LAN IP (an insecure context).
//
// Local-assets dev mode (`--dev-assets <dir> --dev-config <file>`) additionally
// serves model files from a directory outside this tree, so real models can be
// tried before any checkpoint is cleared for hosting. It binds loopback only and
// is never part of the deployed site: production has no /__dev__ routes.
// `--share --tls-cert <crt> --tls-key <key>` adds an HTTPS listener on every
// interface (port --share-port, default 5174) for an authenticated HTTPS proxy in
// front of this machine, and marks pages served there so they use the dev config
// on that proxy's hostname too. Plain HTTP stays on loopback.
import { createReadStream } from 'node:fs';
import { readdir, readFile, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { createServer as createTlsServer } from 'node:https';
import { networkInterfaces } from 'node:os';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SITE_ROOT = resolve(ROOT, '..', '..');
const args = parseArgs(process.argv.slice(2));
const PORT = Number(args.port ?? process.env.PORT) || 5173;
const DEV_ASSETS = args['dev-assets'] ? resolve(args['dev-assets']) : null;
const DEV_CONFIG = args['dev-config'] ? resolve(args['dev-config']) : null;
const DEV = Boolean(DEV_ASSETS && DEV_CONFIG);
const SHARE = DEV && args.share === 'true';
const SHARE_PORT = Number(args['share-port']) || 5174;
const HOST = DEV ? '127.0.0.1' : '0.0.0.0';
const SHARE_MARK = '<meta name="executorch-dev-assets" content="shared">';
if (SHARE && !(args['tls-cert'] && args['tls-key'])) {
  console.error('--share needs --tls-cert and --tls-key: the proxy only talks to HTTPS backends');
  process.exit(1);
}

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.wasm': 'application/wasm',
  '.txt': 'text/plain; charset=utf-8',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
};

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const m = /^--([\w-]+)(?:=(.*))?$/.exec(argv[i]);
    if (!m) continue;
    out[m[1]] = m[2] ?? (argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true');
  }
  return out;
}

// --dev-config is a JSON file, or a directory of JSON fragments merged by key.
async function readDevConfig() {
  const files = (await stat(DEV_CONFIG)).isDirectory()
    ? (await readdir(DEV_CONFIG)).filter((f) => f.endsWith('.json')).sort().map((f) => join(DEV_CONFIG, f))
    : [DEV_CONFIG];
  const merged = { models: {}, shared: {} };
  for (const f of files) {
    const c = JSON.parse(await readFile(f, 'utf8'));
    Object.assign(merged.models, c.models);
    Object.assign(merged.shared, c.shared);
  }
  return merged;
}

function inside(base, path) {
  const file = normalize(join(base, path));
  return file === base || file.startsWith(base + sep) ? file : null;
}

async function sendFile(req, res, file, cacheControl, shared) {
  let info = await stat(file);
  if (info.isDirectory()) {
    file = join(file, 'index.html');
    info = await stat(file);
  }
  if (shared && extname(file) === '.html') {
    const html = (await readFile(file, 'utf8')).replace('<head>', `<head>\n  ${SHARE_MARK}`);
    res.writeHead(200, { 'content-type': TYPES['.html'], 'cache-control': cacheControl });
    return res.end(req.method === 'HEAD' ? undefined : html);
  }
  res.writeHead(200, {
    'content-type': TYPES[extname(file).toLowerCase()] ?? 'application/octet-stream',
    'content-length': String(info.size),
    'cache-control': cacheControl,
  });
  if (req.method === 'HEAD') return res.end();
  createReadStream(file).pipe(res);
}

function requestPath(url) {
  try {
    return decodeURIComponent(new URL(url, 'http://localhost/').pathname);
  } catch {
    return null;
  }
}

const handler = (shared) => async (req, res) => {
  const path = requestPath(req.url);
  if (path == null) return res.writeHead(400).end();
  try {
    if (DEV && path === '/__dev__/config.json') {
      const body = JSON.stringify({ ...(await readDevConfig()), base: '/__dev_assets__/' });
      res.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store' });
      return res.end(body);
    }
    if (DEV && path.startsWith('/__dev_assets__/')) {
      const file = inside(DEV_ASSETS, path.slice('/__dev_assets__/'.length));
      if (!file) return res.writeHead(403).end();
      return await sendFile(req, res, file, 'no-cache', shared);
    }
    // The whole site is served as deployed; / opens the playground hub.
    if (path === '/') {
      res.writeHead(302, { location: '/playground/' });
      return res.end();
    }
    const file = inside(SITE_ROOT, path);
    if (!file) return res.writeHead(403).end();
    await sendFile(req, res, file, 'no-store', shared);
  } catch {
    if (!res.headersSent) res.writeHead(404, { 'content-type': 'text/plain' }).end('not found');
  }
};

if (SHARE) {
  const readTls = async () => ({ cert: await readFile(args['tls-cert']), key: await readFile(args['tls-key']) });
  const tlsServer = createTlsServer(await readTls(), handler(true));
  // Host certificates rotate under a running server; pick up the new pair hourly.
  setInterval(async () => {
    try {
      tlsServer.setSecureContext(await readTls());
    } catch (err) {
      console.error(`could not reload TLS certificate: ${err.message}`);
    }
  }, 60 * 60 * 1000).unref();
  tlsServer.listen(SHARE_PORT, '::', () => {
    console.log(`shared over HTTPS on port ${SHARE_PORT} (all interfaces) for an authenticated proxy`);
  });
}

// A malformed request must never take the server down for everyone else.
process.on('uncaughtException', (err) => console.error(`request error: ${err.message}`));

createServer(handler(false)).listen(PORT, HOST, () => {
  console.log(`secure context   http://localhost:${PORT}/playground/   (site home: http://localhost:${PORT}/index.html)`);
  if (DEV) {
    console.log('local-assets dev mode: loopback only; model files are served from outside the site tree');
    return;
  }
  for (const addrs of Object.values(networkInterfaces())) {
    for (const a of addrs ?? []) {
      if (a.family === 'IPv4' && !a.internal) console.log(`insecure (LAN)   http://${a.address}:${PORT}/playground/`);
    }
  }
});
