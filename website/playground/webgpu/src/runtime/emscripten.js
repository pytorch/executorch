// Loads an Emscripten MODULARIZE runtime (a classic script defining a global
// factory) and instantiates it, with its .wasm passed in as bytes or resolved
// next to the script.
const scripts = new Map();

export function loadScript(url) {
  if (!scripts.has(url)) {
    scripts.set(
      url,
      new Promise((resolve, reject) => {
        const el = document.createElement('script');
        el.src = url;
        el.async = true;
        el.onload = () => resolve();
        el.onerror = () => {
          scripts.delete(url);
          reject(new Error(`Could not load runtime script ${url}`));
        };
        document.head.append(el);
      }),
    );
  }
  return scripts.get(url);
}

// Each runtime script declares its factory with `var`, so two builds that share a
// factory name overwrite each other. Capture the factory right after its own
// script loads, and give every build a unique script URL.
const factories = new Map();

export async function instantiate(scriptUrl, factoryName, { wasmUrl, wasmBinary, onLog } = {}) {
  if (!factories.has(scriptUrl)) {
    await loadScript(scriptUrl);
    const factory = globalThis[factoryName];
    if (typeof factory !== 'function') throw new Error(`Runtime ${scriptUrl} did not define ${factoryName}()`);
    factories.set(scriptUrl, factory);
  }
  const dir = scriptUrl.slice(0, scriptUrl.lastIndexOf('/') + 1);
  return factories.get(scriptUrl)({
    ...(wasmBinary ? { wasmBinary } : {}),
    locateFile: (p) => (p.endsWith('.wasm') && wasmUrl ? wasmUrl : dir + p),
    print: (m) => onLog?.('log', m),
    printErr: (m) => onLog?.('err', m),
  });
}

export function hasExport(module, name) {
  return typeof module[`_${name}`] === 'function';
}
