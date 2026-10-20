import { svg, uid } from '../lib/dom.js';

// One icon family: a superellipse tile in the category's pastel-to-mid
// gradient, a top sheen and inner rim, and a glyph built from a white surface
// gradient (W) plus a category accent gradient (D) on a shared 64-unit grid.
// The glyph's live area is roughly 12..52, so the optical size stays even.

const SQUIRCLE = (() => {
  const pts = [];
  for (let i = 0; i < 160; i++) {
    const t = (i / 160) * 2 * Math.PI;
    const c = Math.cos(t);
    const s = Math.sin(t);
    pts.push(`${(32 + 32 * Math.sign(c) * Math.abs(c) ** 0.4).toFixed(2)} ${(32 + 32 * Math.sign(s) * Math.abs(s) ** 0.4).toFixed(2)}`);
  }
  return `M${pts.join('L')}Z`;
})();

const TONES = {
  language: { light: '#F5ECFF', mid: '#C4A0F5', accent: '#B07CF7', deep: '#7B2FD8', surface: '#F1E8FD' },
  vision: { light: '#E6F6EC', mid: '#86CFA0', accent: '#45B46A', deep: '#137333', surface: '#E5F3EA' },
  generative: { light: '#FFEDE8', mid: '#F4A391', accent: '#F07A64', deep: '#C5321F', surface: '#FCE9E5' },
  audio: { light: '#E8F0FF', mid: '#8EB3F7', accent: '#5B95F5', deep: '#1A5BD6', surface: '#E6EEFD' },
  training: { light: '#FFF6DA', mid: '#F5C752', accent: '#F2AE1C', deep: '#A15C00', surface: '#FCF1D4' },
};

const DEMO_CATEGORY = {
  chat: 'language',
  effects: 'vision', segment: 'vision', vlm: 'vision', depth: 'vision', detect: 'vision',
  cppn: 'generative', bgremove: 'generative', superres: 'generative',
  'speech-to-text': 'audio',
  trainer: 'training', kernels: 'training',
};

const rr = (x, y, w, h, r) => `M${x + r} ${y}H${x + w - r}A${r} ${r} 0 0 1 ${x + w} ${y + r}V${y + h - r}A${r} ${r} 0 0 1 ${x + w - r} ${y + h}H${x + r}A${r} ${r} 0 0 1 ${x} ${y + h - r}V${y + r}A${r} ${r} 0 0 1 ${x + r} ${y}Z`;
const star = (cx, cy, r) => `M${cx} ${cy - r}C${cx + r * 0.14} ${cy - r * 0.14} ${cx + r * 0.14} ${cy - r * 0.14} ${cx + r} ${cy}C${cx + r * 0.14} ${cy + r * 0.14} ${cx + r * 0.14} ${cy + r * 0.14} ${cx} ${cy + r}C${cx - r * 0.14} ${cy + r * 0.14} ${cx - r * 0.14} ${cy + r * 0.14} ${cx - r} ${cy}C${cx - r * 0.14} ${cy - r * 0.14} ${cx - r * 0.14} ${cy - r * 0.14} ${cx} ${cy - r}Z`;

const GLYPHS = {
  chat: ({ W, D, t }) => `
    <path d="${rr(26, 13, 27, 20, 8)}" fill="#fff" fill-opacity=".55"/>
    <path d="M46 32l5 6.5-.5-7z" fill="#fff" fill-opacity=".55"/>
    <path d="${rr(11, 22, 33, 25, 10)}" fill="${W}"/>
    <path d="M17 45l-3.5 8 11-6.5z" fill="${W}"/>
    <circle cx="20" cy="34.5" r="2.7" fill="${D}"/><circle cx="27.5" cy="34.5" r="2.7" fill="${D}" fill-opacity=".75"/><circle cx="35" cy="34.5" r="2.7" fill="${D}" fill-opacity=".5"/>
    <path d="${star(47.5, 15, 5.2)}" fill="${t.deep}" fill-opacity=".85"/>`,

  effects: ({ W, D, id }) => `
    <defs><radialGradient id="ln-${id}" cx=".38" cy=".35" r=".75"><stop offset="0" stop-color="#5E6B7A"/><stop offset=".55" stop-color="#2A3340"/><stop offset="1" stop-color="#11161D"/></radialGradient></defs>
    <path d="${rr(21, 16, 15, 9, 3)}" fill="${W}"/>
    <path d="${rr(11, 21, 42, 29, 8)}" fill="${W}"/>
    <circle cx="32" cy="35.5" r="11" fill="${D}"/>
    <circle cx="32" cy="35.5" r="7.6" fill="url(#ln-${id})"/>
    <circle cx="29.3" cy="32.8" r="2.1" fill="#fff" fill-opacity=".85"/>
    <circle cx="34.6" cy="38.2" r="1" fill="#fff" fill-opacity=".4"/>
    <path d="${rr(43.5, 25, 5, 3, 1.5)}" fill="${D}" fill-opacity=".55"/>
    <path d="${star(50, 14.5, 6.2)}" fill="#FBBC04"/>`,

  segment: ({ W, D, t }) => `
    <path d="${rr(11, 11, 40, 40, 9)}" fill="${W}"/>
    <path d="M23.5 39.5C17.5 34 20.5 22.5 29.5 21.5 37.5 20.5 44.5 26.5 42.5 34 41.5 40.5 30.5 45.5 23.5 39.5Z" fill="${t.accent}" fill-opacity=".42" stroke="${D}" stroke-width="2.4" stroke-linejoin="round"/>
    ${[[19.5, 19.5], [44.5, 19.5], [19.5, 43], [44.5, 43]].map(([x, y]) => `<rect x="${x - 2.1}" y="${y - 2.1}" width="4.2" height="4.2" rx="1.1" fill="#fff" stroke="${t.deep}" stroke-width="1.3"/>`).join('')}
    <path d="M38.5 36.5V51.5l3.8-3.7 3.1 6.2 2.8-1.3-3-6h5.4Z" fill="#fff" stroke="#202124" stroke-width="1.5" stroke-linejoin="round"/>`,

  vlm: ({ W, D, id, t }) => `
    <defs><clipPath id="pc-${id}"><path d="${rr(10, 10, 36, 31, 8)}"/></clipPath></defs>
    <path d="${rr(10, 10, 36, 31, 8)}" fill="${W}"/>
    <g clip-path="url(#pc-${id})">
      <path d="M8 43 L20 26 L27 33.5 L33 25.5 L48 43Z" fill="${D}"/>
      <path d="M17 30.2 L20 26 L23.2 29.6 L21 28.6 L20 30 L18.8 28.8Z" fill="#fff"/>
      <circle cx="38.5" cy="18.5" r="3.6" fill="#FBBC04"/>
    </g>
    <path d="${rr(27, 31, 27, 19, 8)}" fill="${t.deep}"/>
    <path d="M45 49l7 5-1.5-7.5z" fill="${t.deep}"/>
    <circle cx="34" cy="40.5" r="2.1" fill="#fff"/><circle cx="40.5" cy="40.5" r="2.1" fill="#fff" fill-opacity=".8"/><circle cx="47" cy="40.5" r="2.1" fill="#fff" fill-opacity=".6"/>`,

  depth: ({ W, D }) => {
    const layer = (y) => `M32 ${y - 9.5}L51 ${y}L32 ${y + 9.5}L13 ${y}Z`;
    return `
    <path d="${layer(42)}" fill="${D}"/>
    <path d="${layer(33)}" fill="#fff" fill-opacity=".62"/>
    <path d="${layer(24)}" fill="${W}"/>
    <circle cx="32" cy="24" r="3.2" fill="#FBBC04"/>`;
  },

  detect: ({ W, D, t }) => `
    <g fill="none" stroke="${W}" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round">
      <path d="M11.5 21v-5a4.5 4.5 0 0 1 4.5-4.5h5M43 11.5h5a4.5 4.5 0 0 1 4.5 4.5v5M52.5 43v5a4.5 4.5 0 0 1-4.5 4.5h-5M21 52.5h-5a4.5 4.5 0 0 1-4.5-4.5v-5"/>
    </g>
    <path d="${rr(20, 24, 24, 21, 4)}" fill="${t.accent}" fill-opacity=".2" stroke="${D}" stroke-width="2.2"/>
    <path d="${rr(19, 17.5, 17, 7, 3)}" fill="${t.deep}"/>
    <path d="M22 21h8" stroke="#fff" stroke-width="1.6" stroke-linecap="round"/>
    <circle cx="32" cy="32" r="4.3" fill="${W}"/>
    <path d="M24.2 44.6c.8-5.6 3.8-7.9 7.8-7.9s7 2.3 7.8 7.9z" fill="${W}"/>`,

  bgremove: ({ W, D, id, t }) => {
    const checks = [];
    for (let r = 0; r < 8; r++) for (let c = 0; c < 4; c++) checks.push(`<rect x="${32 + c * 5}" y="${11 + r * 5}" width="5" height="5" fill="${(r + c) % 2 ? '#E9EBEF' : '#fff'}"/>`);
    return `
    <defs><clipPath id="bc-${id}"><path d="${rr(11, 11, 40, 40, 9)}"/></clipPath></defs>
    <path d="${rr(11, 11, 40, 40, 9)}" fill="${W}"/>
    <g clip-path="url(#bc-${id})">
      <rect x="11" y="11" width="21" height="40" fill="${t.mid}" fill-opacity=".55"/>
      ${checks.join('')}
      <circle cx="32" cy="27.5" r="7.2" fill="${D}"/>
      <path d="M17.5 52c1.2-10.5 6.6-14.8 14.5-14.8s13.3 4.3 14.5 14.8z" fill="${D}"/>
    </g>
    <path d="${star(50, 14, 5.4)}" fill="#fff"/>`;
  },

  superres: ({ W, D, t }) => {
    const op = [1, 0.7, 0.85, 0.55, 0.75, 1, 0.6, 0.9, 0.55, 0.85, 1, 0.7, 0.9, 0.6, 0.75, 1];
    const px = op.map((o, i) => `<rect x="${11 + (i % 4) * 7.5}" y="${11 + Math.floor(i / 4) * 7.5}" width="6.3" height="6.3" rx="1.4" fill="${W}" fill-opacity="${o}"/>`).join('');
    return `${px}
    <path d="M47 47l6.5 6.5" stroke="${t.deep}" stroke-width="4.6" stroke-linecap="round"/>
    <circle cx="38.5" cy="38.5" r="12.5" fill="${W}"/>
    <circle cx="38.5" cy="38.5" r="12.5" fill="none" stroke="${D}" stroke-width="2.8"/>
    <circle cx="38.5" cy="38.5" r="6" fill="${D}"/>
    <circle cx="36.4" cy="36.4" r="1.9" fill="#fff" fill-opacity=".8"/>`;
  },

  'speech-to-text': ({ W, D, t }) => `
    <path d="${rr(15, 11, 14, 23, 7)}" fill="${W}"/>
    <path d="M18.5 19h7M18.5 23h7M18.5 27h7" stroke="${t.accent}" stroke-opacity=".55" stroke-width="1.3" stroke-linecap="round"/>
    <path d="M11.5 27a10.5 10.5 0 0 0 21 0M22 38v6.5M16.5 46.5h11" stroke="${W}" stroke-width="2.8" fill="none" stroke-linecap="round"/>
    <path d="${rr(37, 15, 16, 4.6, 2.3)}" fill="${D}"/>
    <path d="${rr(37, 24, 13, 4.6, 2.3)}" fill="${W}"/>
    <path d="${rr(37, 33, 16, 4.6, 2.3)}" fill="${W}"/>
    <path d="${rr(37, 42, 9, 4.6, 2.3)}" fill="${W}" fill-opacity=".75"/>`,

  cppn: ({ id }) => {
    const colors = ['#FDE7A8', '#F9B9A4', '#F28B82', '#D7AEF5', '#B590EE', '#FDD663', '#F6AEA9'];
    const bands = colors.map((c, i) => {
      const y0 = 10 + i * 6.5;
      let d = `M10 ${y0}`;
      for (let x = 10; x <= 54; x += 2) d += `L${x} ${(y0 + 3.2 * Math.sin(x / 4.5 + i * 0.9) + 1.6 * Math.sin(x / 2.2 - i)).toFixed(2)}`;
      return `<path d="${d}L54 56L10 56Z" fill="${c}"/>`;
    }).join('');
    return `
    <defs><clipPath id="oc-${id}"><circle cx="32" cy="33" r="19"/></clipPath><radialGradient id="os-${id}" cx=".35" cy=".25" r=".8"><stop offset="0" stop-color="#fff" stop-opacity=".55"/><stop offset=".6" stop-color="#fff" stop-opacity="0"/></radialGradient></defs>
    <circle cx="32" cy="33" r="19" fill="#FDE7A8"/>
    <g clip-path="url(#oc-${id})">${bands}</g>
    <circle cx="32" cy="33" r="19" fill="url(#os-${id})"/>
    <circle cx="32" cy="33" r="18.3" fill="none" stroke="#fff" stroke-width="1.6"/>
    <path d="${star(49.5, 14.5, 5.6)}" fill="#fff"/>`;
  },

  trainer: ({ W }) => {
    const colors = ['#4285F4', '#EA4335', '#34A853'];
    const dots = [];
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < 5; i++) {
        const k = i / 5;
        const r = 4.5 + k * 14;
        const a = k * 3.4 + (c * 2 * Math.PI) / 3;
        dots.push(`<circle cx="${(32 + r * Math.cos(a)).toFixed(2)}" cy="${(32 + r * Math.sin(a)).toFixed(2)}" r="${(1.9 + k * 1.4).toFixed(2)}" fill="${colors[c]}"/>`);
      }
    }
    return `<circle cx="32" cy="32" r="20" fill="${W}"/>${dots.join('')}`;
  },

  kernels: ({ W, D, t }) => {
    const pins = [];
    for (const p of [23, 30.5, 38]) {
      pins.push(`<path d="${rr(p, 10, 3.2, 6, 1.4)}" fill="${W}"/>`, `<path d="${rr(p, 48, 3.2, 6, 1.4)}" fill="${W}"/>`);
      pins.push(`<path d="${rr(10, p, 6, 3.2, 1.4)}" fill="${W}"/>`, `<path d="${rr(48, p, 6, 3.2, 1.4)}" fill="${W}"/>`);
    }
    const cells = [];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const on = r === 1 && c === 1;
        const row = r === 1 || c === 1;
        cells.push(`<path d="${rr(21.5 + c * 7.5, 21.5 + r * 7.5, 6, 6, 1.4)}" fill="${on ? D : t.accent}" fill-opacity="${on ? 1 : row ? 0.6 : 0.3}"/>`);
      }
    }
    return `${pins.join('')}<path d="${rr(15, 15, 34, 34, 7)}" fill="${W}"/>${cells.join('')}`;
  },
};

export function appIcon(demoId, size = 28) {
  const t = TONES[DEMO_CATEGORY[demoId]] ?? TONES.language;
  const glyph = GLYPHS[demoId];
  const id = uid('i');
  const W = `url(#w-${id})`;
  const D = `url(#d-${id})`;
  return svg(`
<svg class="app-icon" width="${size}" height="${size}" viewBox="0 0 64 64" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g-${id}" x1="0" y1="0" x2=".35" y2="1"><stop offset="0" stop-color="${t.light}"/><stop offset="1" stop-color="${t.mid}"/></linearGradient>
    <radialGradient id="h-${id}" cx=".28" cy="0" r=".95"><stop offset="0" stop-color="#fff" stop-opacity=".6"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>
    <linearGradient id="w-${id}" gradientUnits="userSpaceOnUse" x1="0" y1="10" x2="0" y2="54"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="${t.surface}"/></linearGradient>
    <linearGradient id="d-${id}" gradientUnits="userSpaceOnUse" x1="0" y1="12" x2="0" y2="52"><stop offset="0" stop-color="${t.accent}"/><stop offset="1" stop-color="${t.deep}"/></linearGradient>
    <filter id="s-${id}" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="1.3" stdDeviation="1.3" flood-color="${t.deep}" flood-opacity=".3"/></filter>
    <clipPath id="c-${id}"><path d="${SQUIRCLE}"/></clipPath>
  </defs>
  <path d="${SQUIRCLE}" fill="url(#g-${id})"/>
  <path d="${SQUIRCLE}" fill="url(#h-${id})"/>
  <g filter="url(#s-${id})">${glyph ? glyph({ W, D, id, t }) : ''}</g>
  <path d="${SQUIRCLE}" fill="none" stroke="#fff" stroke-opacity=".6" stroke-width="1.6" clip-path="url(#c-${id})"/>
  <path d="${SQUIRCLE}" fill="none" stroke="${t.deep}" stroke-opacity=".12" stroke-width="1" />
</svg>`);
}
