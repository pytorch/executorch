import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, extname, join, relative, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

// The whole site is scanned, not just the playground: the homepage links into it.
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const SCANNED = new Set(['.js', '.mjs', '.css', '.html', '.json', '.md', '.toml', '.svg', '.txt']);

// Patterns are assembled from fragments so this file does not match itself.
const j = (...parts) => parts.join('');
const FORBIDDEN = [
  { name: 'internal corp hostnames', re: new RegExp(j('intern', '(?:fb|alfb|almeta)\\.com'), 'i') },
  { name: 'internal short links', re: new RegExp(j('\\bfb', 'url\\.com|\\bfb\\.', 'me/|\\b', 'our\\.intern'), 'i') },
  { name: 'internal tooling hosts', re: new RegExp(j('\\b(?:phab', 'ricator|work', 'place)\\.(?:intern|facebook)'), 'i') },
  { name: 'internal monorepo paths', re: new RegExp(j('\\bfb', 'code/|\\bfb', 'source\\b'), 'i') },
  { name: 'internal blob storage handles', re: new RegExp(j('\\b(?:mani', 'fold|ever', 'store)://'), 'i') },
  { name: 'internal blob storage names', re: new RegExp(j('\\bmani', 'fold\\b(?:\\s+bucket|\\.|:)'), 'i') },
  { name: 'internal image hosting', re: new RegExp(j('\\bpxl', '\\.cl\\b'), 'i') },
  { name: 'review/diff IDs', re: /\bD\d{7,9}\b/ },
  { name: 'task IDs', re: /\bT\d{7,10}\b/ },
  { name: 'incident IDs', re: /\bS\d{6,7}\b/ },
  { name: 'devserver hostnames', re: new RegExp(j('\\.', 'facebook\\.com\\b|\\bdev', 'vm\\d+'), 'i') },
  { name: 'local user paths', re: /\/Users\/[a-z][\w.-]*\//i },
  { name: 'devserver home paths', re: new RegExp(j('/data/', 'users/|/home/[a-z][\\w-]*/'), 'i') },
  { name: 'internal app hosts', re: new RegExp(j('intern', 'meta\\.com|\\bintern', 'cache\\b|\\bx2p\\b'), 'i') },
  { name: 'internal asset tree', re: new RegExp(j('models/', 'webgpu|candidate-', 'builds|demo_', 'assets/'), 'i') },
  { name: 'third-party script CDNs', re: new RegExp(j('cdn\\.js', 'delivr|\\besm\\.(?:run|sh)\\b|\\bun', 'pkg\\.com'), 'i') },
];

const BINARY = new Set(['.pte', '.ptd', '.wasm', '.bin', '.gguf', '.onnx', '.safetensors']);

function files(dir, exts = SCANNED) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((d) => {
    if (d.name === 'node_modules' || d.name.startsWith('.')) return [];
    const p = join(dir, d.name);
    return d.isDirectory() ? files(p, exts) : exts.has(extname(p)) ? [p] : [];
  });
}

test('the site ships no model weights and no binaries (the runtime .wasm is hosted with the models)', () => {
  assert.deepEqual(files(ROOT, BINARY).map((f) => relative(ROOT, f)), []);
});

test('source contains no internal or corporate references', () => {
  const hits = [];
  for (const f of files(ROOT)) {
    const lines = readFileSync(f, 'utf8').split('\n');
    lines.forEach((line, i) => {
      for (const { name, re } of FORBIDDEN) if (re.test(line)) hits.push(`${relative(ROOT, f)}:${i + 1} [${name}] ${line.trim().slice(0, 120)}`);
    });
  }
  assert.deepEqual(hits, []);
});

test('the scanner actually catches what it claims to', () => {
  const samples = [
    j('https://www.intern', 'fb.com/x'), // @lint-ignore (test fixture, not a real link)
    j('see manif', 'old://bucket/x'),
    j('fixed in D', '12345678'),
    j('tracked as T', '123456789'),
    j('fbcod', 'e/foo/bar'),
    j('/data/', 'users/someone/x'),
    j('https://app.intern', 'meta.com/'), // @lint-ignore (test fixture, not a real link)
    j('dashboard/candidate-', 'builds/x'),
    j('https://cdn.js', 'delivr.net/npm/x'), // @lint-ignore (test fixture, not a real link)
  ];
  for (const s of samples) assert.ok(FORBIDDEN.some(({ re }) => re.test(s)), `not caught: ${s}`);
});
