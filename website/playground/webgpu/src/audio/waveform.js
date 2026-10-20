// Canvas waveform: a scrolling level meter while recording, or the envelope of
// a finished clip. Colours come from the theme tokens at draw time.
import { peaks } from './pcm.js';

const BAR = 3;
const GAP = 3;

export function createWaveform(canvas) {
  const levels = [];
  let clip = null;
  let progress = -1;
  let raf = 0;

  const size = () => {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const hgt = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== hgt) {
      canvas.width = w;
      canvas.height = hgt;
    }
    return dpr;
  };

  const colors = () => {
    const cs = getComputedStyle(canvas);
    return { live: cs.getPropertyValue('--accent').trim() || '#e2502a', idle: cs.getPropertyValue('--text-3').trim() || '#888' };
  };

  function draw() {
    raf = 0;
    const dpr = size();
    const ctx = canvas.getContext('2d');
    const { width: W, height: H } = canvas;
    ctx.clearRect(0, 0, W, H);
    const c = colors();
    const step = (BAR + GAP) * dpr;
    const n = Math.max(1, Math.floor(W / step));
    const mid = H / 2;
    const bar = (i, amp, color, alpha = 1) => {
      const hh = Math.max(BAR * dpr, amp * (H - 4 * dpr));
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(i * step + GAP * dpr * 0.5, mid - hh / 2, BAR * dpr, hh, BAR * dpr);
      ctx.fill();
    };
    if (clip) {
      const env = peaks(clip, n);
      let max = 0;
      for (const p of env) max = Math.max(max, p);
      const scale = max > 0 ? 1 / max : 0;
      for (let i = 0; i < n; i++) {
        const done = progress >= 0 && i / n <= progress;
        bar(i, Math.sqrt(env[i] * scale), done ? c.live : c.idle, done ? 1 : 0.55);
      }
      ctx.globalAlpha = 1;
      return;
    }
    const first = Math.max(0, levels.length - n);
    const offset = n - (levels.length - first);
    for (let i = 0; i < n; i++) {
      const k = i - offset;
      if (k < 0) bar(i, 0, c.idle, 0.35);
      else bar(i, Math.min(1, Math.sqrt(levels[first + k]) * 1.6), c.live);
    }
    ctx.globalAlpha = 1;
  }

  const schedule = () => {
    if (!raf) raf = requestAnimationFrame(draw);
  };
  const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(schedule) : null;
  ro?.observe(canvas);
  const themeWatch = typeof MutationObserver === 'function' ? new MutationObserver(schedule) : null;
  themeWatch?.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  schedule();

  return {
    push(level) {
      clip = null;
      levels.push(level);
      if (levels.length > 2000) levels.splice(0, 1000);
      schedule();
    },
    showClip(samples) {
      clip = samples;
      progress = -1;
      levels.length = 0;
      schedule();
    },
    setProgress(f) {
      progress = f;
      schedule();
    },
    clear() {
      clip = null;
      levels.length = 0;
      schedule();
    },
    redraw: schedule,
    dispose() {
      ro?.disconnect();
      themeWatch?.disconnect();
      cancelAnimationFrame(raf);
    },
  };
}
