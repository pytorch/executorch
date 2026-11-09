// WebGPU detection without UA sniffing: ask for an adapter and report what the
// browser says. Failure modes are kept distinct because each has a different remedy.

export const GPU_FAILURE = Object.freeze({
  INSECURE_CONTEXT: 'insecure-context',
  NO_API: 'no-api',
  NO_ADAPTER: 'no-adapter',
  DEVICE_FAILED: 'device-failed',
  DEVICE_LOST: 'device-lost',
});

export const FAILURE_COPY = {
  [GPU_FAILURE.INSECURE_CONTEXT]: {
    chip: 'No WebGPU · insecure page',
    title: 'This page isn’t in a secure context',
    body: 'Browsers only expose WebGPU on HTTPS pages and on localhost. This copy was opened over plain HTTP, so the GPU is hidden from it.',
    remedies: ['Open the HTTPS version of this page.', 'If you are serving it yourself, use HTTPS or localhost.'],
  },
  [GPU_FAILURE.NO_API]: {
    chip: 'No WebGPU in this browser',
    title: 'Your browser doesn’t offer WebGPU yet',
    body: 'navigator.gpu is missing, so this browser (or this browser configuration) has no WebGPU implementation turned on.',
    remedies: [
      'Use a current Chrome or Edge on desktop or Android, or Safari 26 or later.',
      'Firefox ships WebGPU on some platforms. Check that it is enabled in your version.',
      'Enterprise policies can switch WebGPU off. Ask your administrator if it should be on.',
    ],
  },
  [GPU_FAILURE.NO_ADAPTER]: {
    chip: 'No WebGPU adapter',
    title: 'WebGPU is here, but no GPU was offered',
    body: 'The API exists, but requestAdapter() returned nothing. That usually means the GPU driver is blocklisted, or this is a virtual machine or remote desktop without GPU passthrough.',
    remedies: [
      'Update your graphics driver and restart the browser.',
      'Turn on hardware acceleration in your browser settings.',
      'On a VM or remote desktop, try a machine with direct GPU access.',
    ],
  },
  [GPU_FAILURE.DEVICE_FAILED]: {
    chip: 'GPU device unavailable',
    title: 'Found a GPU, but couldn’t open it',
    body: 'An adapter was offered, but creating a device on it failed. This often clears up after the GPU process restarts.',
    remedies: ['Close other GPU-heavy tabs and reload.', 'Restart the browser.', 'Update your graphics driver.'],
  },
  [GPU_FAILURE.DEVICE_LOST]: {
    chip: 'GPU connection lost',
    title: 'The GPU connection was lost',
    body: 'The browser reset the GPU device. This can happen after a driver crash, a long-running shader, or when the system reclaims GPU memory.',
    remedies: ['Reconnect below. Loaded demos will need to load again.'],
  },
};

export async function readAdapterInfo(adapter) {
  let info = adapter.info;
  if (!info && typeof adapter.requestAdapterInfo === 'function') {
    try {
      info = await adapter.requestAdapterInfo();
    } catch {
      info = null;
    }
  }
  const field = (k) => (info && typeof info[k] === 'string' ? info[k].trim() : '');
  return {
    vendor: field('vendor'),
    architecture: field('architecture'),
    device: field('device'),
    description: field('description'),
    isFallbackAdapter: Boolean(info?.isFallbackAdapter ?? adapter.isFallbackAdapter),
  };
}

// Browsers mask these strings to different degrees; blanks are normal.
export function adapterLabel(info) {
  const primary = [info.vendor, info.architecture].filter(Boolean).join(' ');
  const label = primary || info.description || info.device || 'WebGPU adapter (details hidden by browser)';
  return info.isFallbackAdapter ? `${label} · software` : label;
}

export async function detectGPU({ powerPreference = 'high-performance' } = {}) {
  if (!globalThis.isSecureContext) return { ok: false, reason: GPU_FAILURE.INSECURE_CONTEXT };
  if (!('gpu' in navigator) || !navigator.gpu) return { ok: false, reason: GPU_FAILURE.NO_API };

  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference });
  } catch (err) {
    return { ok: false, reason: GPU_FAILURE.NO_ADAPTER, detail: String(err?.message ?? err) };
  }
  if (!adapter) return { ok: false, reason: GPU_FAILURE.NO_ADAPTER };

  const info = await readAdapterInfo(adapter);
  let device;
  try {
    device = await adapter.requestDevice({ label: 'executorch-webgpu-examples' });
  } catch (err) {
    return { ok: false, reason: GPU_FAILURE.DEVICE_FAILED, info, detail: String(err?.message ?? err) };
  }
  return {
    ok: true,
    adapter,
    device,
    info,
    label: adapterLabel(info),
    features: [...adapter.features].sort(),
    limits: pickLimits(device.limits),
  };
}

function pickLimits(limits) {
  const keys = [
    'maxBufferSize',
    'maxStorageBufferBindingSize',
    'maxComputeWorkgroupStorageSize',
    'maxComputeInvocationsPerWorkgroup',
    'maxComputeWorkgroupsPerDimension',
  ];
  return Object.fromEntries(keys.map((k) => [k, limits[k]]));
}
