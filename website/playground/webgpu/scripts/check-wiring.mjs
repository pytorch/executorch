// Static wiring check. Two classes of bug are decidable from source and easy to
// miss in review: an import that names something the target never exports (it
// takes down the whole module graph at load), and a class applied in JS that the
// stylesheet never styles (it renders unstyled while looking fine in the diff).
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ENTRY = join(ROOT, 'src/main.js');
const STYLESHEETS = readdirSync(join(ROOT, 'styles')).filter((f) => f.endsWith('.css')).map((f) => join(ROOT, 'styles', f));
const HTML = [join(ROOT, 'index.html')];

// Modules that are intentionally not imported by the app yet, with the reason.
const UNREFERENCED_OK = {
  'src/loader/fetch-progress.js': 'download path for cleared models; no model resolves to a URL yet',
};

const stripComments = (src) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"`\\])\/\/.*$/gm, '$1');

function parseImports(src) {
  const out = [];
  const re = /import\s+(?:([\s\S]*?)\s+from\s+)?['"]([^'"]+)['"]/g;
  for (const m of src.matchAll(re)) {
    const clause = m[1]?.trim() ?? '';
    const entry = { spec: m[2], named: [], namespace: null };
    const ns = clause.match(/\*\s+as\s+(\w+)/);
    if (ns) entry.namespace = ns[1];
    const braces = clause.match(/\{([\s\S]*)\}/);
    if (braces) {
      for (const part of braces[1].split(',').map((s) => s.trim()).filter(Boolean)) {
        const [name, alias] = part.split(/\s+as\s+/);
        entry.named.push({ name: name.trim(), local: (alias ?? name).trim() });
      }
    }
    out.push(entry);
  }
  return out;
}

function parseExports(src) {
  const names = new Set();
  for (const m of src.matchAll(/export\s+(?:async\s+)?(?:function\*?|const|let|var|class)\s+(\w+)/g)) names.add(m[1]);
  for (const m of src.matchAll(/export\s*\{([^}]*)\}/g)) {
    for (const part of m[1].split(',').map((s) => s.trim()).filter(Boolean)) {
      const pieces = part.split(/\s+as\s+/);
      names.add((pieces[1] ?? pieces[0]).trim());
    }
  }
  return names;
}

function walk(entry) {
  const problems = [];
  const seen = new Map();
  const queue = [entry];
  while (queue.length) {
    const file = queue.shift();
    if (seen.has(file)) continue;
    const src = stripComments(readFileSync(file, 'utf8'));
    seen.set(file, src);
    for (const imp of parseImports(src)) {
      if (!imp.spec.startsWith('.')) continue;
      const target = resolve(dirname(file), imp.spec);
      const rel = relative(ROOT, file);
      if (!existsSync(target)) {
        problems.push(`${rel}: imports missing file ${imp.spec}`);
        continue;
      }
      const exports = parseExports(stripComments(readFileSync(target, 'utf8')));
      for (const { name } of imp.named) {
        if (!exports.has(name)) problems.push(`${rel}: imports { ${name} } but ${relative(ROOT, target)} does not export it`);
      }
      if (imp.namespace) {
        for (const m of src.matchAll(new RegExp(`\\b${imp.namespace}\\.(\\w+)`, 'g'))) {
          if (!exports.has(m[1])) problems.push(`${rel}: uses ${imp.namespace}.${m[1]} but ${relative(ROOT, target)} does not export it`);
        }
      }
      queue.push(target);
    }
  }
  return { problems, modules: seen };
}

function listFiles(dir, ext) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((d) => {
    const p = join(dir, d.name);
    return d.isDirectory() ? listFiles(p, ext) : p.endsWith(ext) ? [p] : [];
  });
}

const quoted = /'([^'\\]*)'|"([^"\\]*)"|`([^`\\]*)`/g;

// Returns { static: Set<class>, prefixes: Set<prefix> } for classes applied in JS/HTML.
function appliedClasses(sources) {
  const found = new Map();
  const prefixes = new Map();
  const add = (cls, where) => {
    if (!cls || /[^\w-]/.test(cls)) return;
    if (!found.has(cls)) found.set(cls, where);
  };
  const addValue = (value, where) => {
    for (const token of value.split(/\s+/)) {
      if (!token) continue;
      const dyn = token.indexOf('${');
      if (dyn === 0) continue;
      if (dyn > 0) prefixes.set(token.slice(0, dyn), where);
      else add(token, where);
    }
  };
  // In `cond ? 'a' : 'b'`, only the branches are classes; strings in the condition are not.
  const branches = (expr) => (expr.includes('?') ? expr.slice(expr.indexOf('?') + 1) : expr);
  for (const [file, src] of sources) {
    const where = relative(ROOT, file);
    for (const m of src.matchAll(/\bclass:\s*([^,}\n]+)/g)) {
      for (const q of branches(m[1]).matchAll(quoted)) addValue(q[1] ?? q[2] ?? q[3], where);
    }
    for (const m of src.matchAll(/classList\.(?:add|remove|toggle|contains)\(([^)]*)\)/g)) {
      const firstArg = m[1].split(/,(?=\s*[^'"`\s])/)[0];
      for (const q of branches(firstArg).matchAll(quoted)) addValue(q[1] ?? q[2] ?? q[3], where);
    }
    for (const m of src.matchAll(/class="([^"]*)"/g)) addValue(m[1], where);
  }
  return { found, prefixes };
}

function definedClasses(cssFiles) {
  const set = new Set();
  for (const f of cssFiles) {
    const css = readFileSync(f, 'utf8')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/url\([^)]*\)/g, '')
      .replace(/"[^"]*"/g, '');
    for (const m of css.matchAll(/\.(-?[_a-zA-Z][\w-]*)/g)) set.add(m[1]);
  }
  return set;
}

export function checkWiring() {
  const { problems, modules } = walk(ENTRY);

  for (const file of listFiles(join(ROOT, 'src'), '.js')) {
    const rel = relative(ROOT, file);
    if (!modules.has(file) && !UNREFERENCED_OK[rel]) problems.push(`${rel}: not reachable from src/main.js (dead module or missing import)`);
  }

  const sources = [...modules].concat(HTML.map((f) => [f, readFileSync(f, 'utf8')]));
  const { found, prefixes } = appliedClasses(sources);
  const defined = definedClasses(STYLESHEETS);
  for (const [cls, where] of found) {
    if (!defined.has(cls)) problems.push(`${where}: applies class "${cls}" but no stylesheet rule mentions .${cls}`);
  }
  for (const [prefix, where] of prefixes) {
    if (![...defined].some((c) => c.startsWith(prefix))) problems.push(`${where}: builds classes "${prefix}…" but no stylesheet rule starts with .${prefix}`);
  }
  return { problems, modules: modules.size, classes: found.size };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { problems, modules, classes } = checkWiring();
  console.log(`checked ${modules} modules and ${classes} applied classes`);
  for (const p of problems) console.log(`  ✗ ${p}`);
  if (problems.length) {
    console.log(`${problems.length} wiring problem(s)`);
    process.exit(1);
  }
  console.log('wiring OK');
}
