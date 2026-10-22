// A single-graph ExecuTorch model on WebGPU, run through the shared runtime:
// set inputs, call forward, read back every output.
import { fetchWithProgress } from '../loader/fetch-progress.js';
import { f32, Program, readOutputs, startRuntime, withRuntime } from './executorch.js';

export class ForwardModel {
  // onProgress receives {stage: 'runtime' | 'weights' | 'gpu', fraction, loaded?, total?, fromCache?}
  // Models loaded this session, by program URL, so revisiting a demo (or another
  // demo that uses the same model) skips loading entirely.
  static #loaded = new Map();

  static isLoaded(source) {
    return Boolean(source?.spec && ForwardModel.#loaded.get(source.url(source.spec.pte))?.alive);
  }

  static async load(source, { onProgress = () => {}, signal } = {}) {
    const { spec, url } = source;
    const key = url(spec.pte);
    const existing = ForwardModel.#loaded.get(key);
    if (existing?.alive) {
      for (const stage of ['runtime', 'weights', 'gpu']) onProgress({ stage, fraction: 1, fromCache: true });
      return existing;
    }
    const abortCheck = () => {
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    };
    onProgress({ stage: 'runtime', fraction: 0 });
    const [, bytes] = await Promise.all([
      startRuntime().then(() => onProgress({ stage: 'runtime', fraction: 1 })),
      fetchWithProgress(key, {
        signal,
        expectedBytes: spec.pteBytes,
        onProgress: (p) => onProgress({ stage: 'weights', fraction: p.total ? p.loaded / p.total : 0, ...p }),
      }),
    ]);
    abortCheck();
    onProgress({ stage: 'gpu', fraction: 0 });
    const program = await withRuntime((et) => Program.load(et, bytes, 'model'));
    const model = new ForwardModel(program);
    if (signal?.aborted) model.dispose();
    abortCheck();
    ForwardModel.#loaded.set(key, model);
    onProgress({ stage: 'gpu', fraction: 1 });
    return model;
  }

  constructor(program) {
    this.program = program;
    this.lastMs = null;
  }

  // inputs: [{data: Float32Array, shape: number[]}] in the program's argument order.
  // -> [{data: Float32Array, shape: number[]}] for every output.
  run(inputs) {
    return withRuntime(async (et) => {
      if (!this.program) throw new Error('This model was unloaded');
      const t0 = performance.now();
      const tensors = inputs.map((t) => f32(et, t.shape, t.data));
      const t1 = performance.now();
      let outputs;
      try {
        outputs = await this.program.forward(tensors);
      } finally {
        for (const t of tensors) t.delete();
      }
      const t2 = performance.now();
      const result = readOutputs(outputs);
      const t3 = performance.now();
      this.lastMs = t3 - t0;
      // Where one run's time went, for the on-page timing breakdown.
      this.lastTiming = { upload: t1 - t0, forward: t2 - t1, readback: t3 - t2 };
      return result;
    });
  }

  get alive() {
    return Boolean(this.program);
  }

  dispose() {
    const program = this.program;
    this.program = null;
    if (program) withRuntime(() => program.delete());
  }
}
