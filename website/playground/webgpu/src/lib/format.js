export function formatBytes(bytes, { approx = false } = {}) {
  if (bytes == null || !Number.isFinite(bytes)) return '—';
  const prefix = approx ? '~' : '';
  if (bytes >= 1e9) return `${prefix}${(bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 1)} GB`;
  if (bytes >= 1e6) return `${prefix}${Math.round(bytes / 1e6)} MB`;
  if (bytes >= 1e3) return `${prefix}${Math.round(bytes / 1e3)} KB`;
  return `${prefix}${bytes} B`;
}

export function formatMs(ms) {
  if (ms == null || !Number.isFinite(ms)) return '—';
  if (ms >= 10000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms >= 100) return `${Math.round(ms)} ms`;
  return `${ms.toFixed(1)} ms`;
}

export function formatRate(value, unit) {
  if (value == null || !Number.isFinite(value)) return '—';
  const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${unit}`;
}

export function formatInt(n) {
  return Number(n).toLocaleString('en-US');
}
