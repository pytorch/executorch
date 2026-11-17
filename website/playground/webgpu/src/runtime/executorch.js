// The ExecuTorch Wasm runtime with its WebGPU backend (built from the public
// ExecuTorch sources; see runtime/BUILD_INFO.md), shared by every model on the
// page: one instance, one WebGPU device, one heap. The loader script ships in
// runtime/; its .wasm downloads from Hugging Face like a model file.
//
// Calls into the runtime can suspend (Asyncify) while the GPU works, and a
// second call must not enter it meanwhile, so every use goes through
// withRuntime(), which runs one caller at a time.
import { RUNTIME_WASM } from '../config/runtime.js';
import { expectSha256, fetchWithProgress } from '../loader/fetch-progress.js';
import { instantiate } from './emscripten.js';

const SCRIPT = new URL('../../runtime/executorch_wasm_lib.js', import.meta.url).href;
const logs = []; // the last 50 runtime log lines, as { n, text }
let logged = 0;
let ready = null;
let queue = Promise.resolve();

function start() {
  ready ??= (async () => {
    if (!globalThis.navigator?.gpu) throw new Error('WebGPU is not available in this browser.');
    expectSha256(RUNTIME_WASM.url, RUNTIME_WASM.sha256);
    const wasmBinary = await fetchWithProgress(RUNTIME_WASM.url, { expectedBytes: RUNTIME_WASM.bytes });
    const et = await instantiate(SCRIPT, 'createExecuTorch', {
      wasmBinary,
      onLog: (kind, text) => {
        logs.push({ n: ++logged, text });
        if (logs.length > 50) logs.shift();
        console.debug(`[ExecuTorch] ${text}`);
      },
    });
    try {
      await et.initWebGPU();
    } catch (err) {
      throw new Error(`No WebGPU device available to the ExecuTorch runtime: ${err.message ?? err}`);
    }
    return et;
  })();
  ready.catch(() => (ready = null));
  return ready;
}

// Starts the runtime (once) without queueing behind other callers.
export function startRuntime() {
  return start().then(() => {});
}

// Runs fn(et) with exclusive use of the runtime. A failure carries the first
// error the runtime logged meanwhile, which usually names the failing op.
export function withRuntime(fn) {
  const run = queue.then(async () => {
    const et = await start();
    const seen = logged;
    try {
      return await fn(et);
    } catch (err) {
      const fresh = logs.filter((l) => l.n > seen).map((l) => l.text);
      const detail = fresh.find((t) => t.startsWith('E ')) ?? fresh.at(-1) ?? '';
      const message = describe(et, err);
      throw new Error(detail && !message.includes(detail) ? `${message} (${detail})` : message);
    }
  });
  queue = run.catch(() => {});
  return run;
}

// The runtime throws Error objects for its own checks, but C++ exceptions and
// file system errors arrive as other objects.
function describe(et, err) {
  if (err instanceof Error && err.message) return err.message;
  try {
    const [type, message] = et.getExceptionMessage(err);
    return message ? `${type}: ${message}` : type;
  } catch {
    return err?.message ?? (err?.errno != null ? `file system error ${err.errno}` : String(err));
  }
}

// Wasm-side tensors. Pass these to forward() and delete them when done; the
// runtime copies `data` in, so the typed array can be reused straight away.
export const f32 = (et, shape, data) => et.Tensor.fromArray(shape, data, et.ScalarType.Float);
export const i64 = (et, shape, values) => et.Tensor.fromArray(shape, BigInt64Array.from(values, BigInt), et.ScalarType.Long);

// Copies every output out of the runtime and frees it. Output memory belongs
// to the program and is overwritten by its next run.
export function readOutputs(outputs) {
  const out = outputs.map((t) => ({ data: t.data.slice(), shape: Array.from(t.sizes) }));
  for (const t of outputs) t.delete();
  return out;
}

// One .pte program with its `forward` method loaded on the GPU.
export class Program {
  static #files = 0;

  // Must run inside withRuntime(). The runtime takes ownership of `bytes`.
  static async load(et, bytes, name) {
    const path = `/program-${++Program.#files}.pte`;
    let module = null;
    try {
      et.FS.writeFile(path, bytes, { canOwn: true });
      module = et.Module.load(path);
      await module.loadMethod('forward');
    } catch (err) {
      module?.delete();
      throw new Error(`The runtime could not load the ${name}: ${describe(et, err)}`);
    } finally {
      // The loader keeps its own handle; the file only has to outlive loadMethod.
      try {
        et.FS.unlink(path);
      } catch {
        // Never written.
      }
    }
    return new Program(et, module, name);
  }

  constructor(et, module, name) {
    this.et = et;
    this.module = module;
    this.name = name;
  }

  // Shape of each tensor input, from the program itself.
  inputShapes() {
    return this.module.getMethodMeta('forward').inputTensorMeta.map((m) => (m ? Array.from(m.sizes) : null));
  }

  // Must run inside withRuntime(). Inputs are runtime tensors; the caller owns
  // and deletes them. Returns the runtime's output tensors (see readOutputs).
  async forward(inputs) {
    try {
      return await this.module.forward(inputs);
    } catch (err) {
      throw new Error(`The ${this.name} failed to run: ${describe(this.et, err)}`);
    }
  }

  delete() {
    this.module?.delete();
    this.module = null;
  }
}
