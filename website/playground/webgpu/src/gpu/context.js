import { createStore } from '../lib/store.js';
import { adapterLabel, detectGPU, GPU_FAILURE } from './adapter.js';

// status: 'probing' | 'ready' | 'unavailable' | 'lost'
export const gpuState = createStore({ status: 'probing', reason: null, label: '', info: null, device: null });

// Shader compile messages, validation errors, and self-test results all land
// here so the UI can surface them instead of burying them in the console.
export const diagnostics = createStore({ items: [] });

let nextId = 1;

export function reportDiagnostic(item) {
  const entry = { id: nextId++, time: Date.now(), ...item };
  diagnostics.set({ items: [...diagnostics.get().items, entry] });
  const log = item.level === 'error' ? console.error : item.level === 'warning' ? console.warn : console.info;
  log(`[webgpu:${item.source}] ${item.title}${item.detail ? `\n${item.detail}` : ''}`);
  return entry;
}

export function dismissDiagnostic(id) {
  diagnostics.set({ items: diagnostics.get().items.filter((d) => d.id !== id) });
}

export async function initGPU() {
  gpuState.set({ status: 'probing', reason: null, device: null });
  const result = await detectGPU();
  if (!result.ok) {
    gpuState.set({
      status: 'unavailable',
      reason: result.reason,
      detail: result.detail ?? '',
      info: result.info ?? null,
      label: result.info ? adapterLabel(result.info) : '',
      device: null,
    });
    return gpuState.get();
  }

  const { device } = result;
  device.addEventListener('uncapturederror', (event) => {
    reportDiagnostic({
      level: 'error',
      source: 'validation',
      title: 'WebGPU reported an error',
      detail: event.error?.message ?? String(event.error),
    });
  });
  device.lost.then((lost) => {
    if (gpuState.get().device !== device) return;
    // reason 'destroyed' means we tore it down on purpose.
    if (lost.reason === 'destroyed') return;
    gpuState.set({ status: 'lost', reason: GPU_FAILURE.DEVICE_LOST, detail: lost.message, device: null });
    reportDiagnostic({ level: 'error', source: 'device', title: 'GPU device lost', detail: lost.message });
  });

  gpuState.set({
    status: 'ready',
    reason: null,
    detail: '',
    device,
    info: result.info,
    label: result.label,
    features: result.features,
    limits: result.limits,
  });
  return gpuState.get();
}

export function requireDevice() {
  const { device, status } = gpuState.get();
  if (!device) throw new Error(`WebGPU device not available (status: ${status})`);
  return device;
}
