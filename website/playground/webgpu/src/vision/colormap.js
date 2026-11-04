// 256-entry RGB lookup tables for single-channel maps.

// Polynomial fit of the Turbo colormap (Mikhailov, 2019).
function turbo(x) {
  const r = 0.13572138 + x * (4.6153926 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
  const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
  const b = 0.1066733 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
  return [r * 255, g * 255, b * 255];
}

// Inferno sampled at eleven evenly spaced points.
const INFERNO = [
  [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110], [147, 38, 103], [188, 55, 84],
  [221, 81, 58], [243, 120, 25], [252, 165, 10], [246, 215, 70], [252, 255, 164],
];

function stops(table) {
  return (x) => {
    const f = x * (table.length - 1);
    const i = Math.min(table.length - 2, Math.floor(f));
    const t = f - i;
    return table[i].map((a, c) => a + (table[i + 1][c] - a) * t);
  };
}

const MAPS = { turbo, inferno: stops(INFERNO), gray: (x) => [x * 255, x * 255, x * 255] };
const cache = new Map();

export const COLORMAPS = [
  { id: 'turbo', label: 'Turbo' },
  { id: 'inferno', label: 'Inferno' },
  { id: 'gray', label: 'Mono' },
];

export function colormapLut(name) {
  if (!cache.has(name)) {
    const fn = MAPS[name] ?? turbo;
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) lut.set(fn(i / 255), i * 3);
    cache.set(name, lut);
  }
  return cache.get(name);
}

// CSS gradient of a colormap, for legends.
export function colormapCss(name) {
  const lut = colormapLut(name);
  const parts = [];
  for (let i = 0; i <= 8; i++) {
    const k = Math.round((i / 8) * 255) * 3;
    parts.push(`rgb(${lut[k]}, ${lut[k + 1]}, ${lut[k + 2]}) ${i * 12.5}%`);
  }
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}
