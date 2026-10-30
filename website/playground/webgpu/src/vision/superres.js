// 4x super-resolution with a fixed-size model: the image is cut into overlapping
// tiles, each tile is upscaled, and only each tile's interior is kept so seams
// fall where both neighbors had context.
import { h } from '../lib/dom.js';
import { segmented } from './controls.js';
import { ctx2d, makeCanvas, sizeOf } from './image.js';

const PAD = 8;
const SIZES = [
  { id: 64, label: '64 px' },
  { id: 128, label: '128 px' },
  { id: 256, label: '256 px' },
];
const BEFORE = [
  { id: 'smooth', label: 'Smooth' },
  { id: 'pixels', label: 'Pixels' },
];

// Tile origins along one axis, plus the [from, to) interior each tile owns.
export function tilePlan(length, tile, pad = PAD) {
  if (length <= tile) return [{ at: 0, from: 0, to: length }];
  const starts = [];
  for (let x = 0; x + tile < length; x += tile - 2 * pad) starts.push(x);
  starts.push(length - tile);
  const uniq = [...new Set(starts)];
  return uniq.map((at, i) => ({
    at,
    from: i === 0 ? 0 : Math.round((uniq[i - 1] + tile + at) / 2),
    to: i === uniq.length - 1 ? length : Math.round((at + tile + uniq[i + 1]) / 2),
  }));
}

// The model input: the picture fit within `max` px, and at least one tile on its short side.
export function inputCanvas(el, max, tile) {
  const [w, h] = sizeOf(el);
  let s = Math.min(1, max / Math.max(w, h));
  if (Math.min(w, h) * s < tile) s = tile / Math.min(w, h);
  const c = makeCanvas(Math.round(w * s), Math.round(h * s));
  const g = c.getContext('2d');
  g.imageSmoothingQuality = 'high';
  g.drawImage(el, 0, 0, c.width, c.height);
  return c;
}

export function createSuperresTask(env) {
  const [T] = env.spec.inputSize ?? [64, 64];
  const S = env.spec.scale ?? 4;
  const state = { size: 256, before: 'smooth', lens: null };
  const lensCanvas = h('canvas', { class: 'vision-lens__canvas', width: 360, height: 360 });
  const lens = h('div', { class: 'vision-lens', hidden: true }, lensCanvas);
  let current = null;

  function drawBefore(g, before, w, hh, sx = 0, sy = 0, sw = before.width, sh = before.height) {
    g.imageSmoothingEnabled = state.before === 'smooth';
    g.imageSmoothingQuality = 'high';
    g.drawImage(before, sx, sy, sw, sh, 0, 0, w, hh);
  }

  function paintLens(canvas, result, view) {
    if (!state.lens || !result) {
      lens.hidden = true;
      return;
    }
    const { x, y } = state.lens;
    const zoom = 3;
    const L = lensCanvas.width;
    const src = L / zoom;
    const cx = x * result.after.width;
    const cy = y * result.after.height;
    const sx = Math.max(0, Math.min(result.after.width - src, cx - src / 2));
    const sy = Math.max(0, Math.min(result.after.height - src, cy - src / 2));
    const g = lensCanvas.getContext('2d');
    const divide = (view.split * result.after.width - sx) * zoom;
    g.clearRect(0, 0, L, L);
    g.imageSmoothingEnabled = false;
    g.drawImage(result.after, sx, sy, src, src, 0, 0, L, L);
    if (divide > 0) {
      g.save();
      g.beginPath();
      g.rect(0, 0, Math.min(L, divide), L);
      g.clip();
      drawBefore(g, result.before, L, L, sx / S, sy / S, src / S, src / S);
      g.restore();
    }
    const rect = canvas.getBoundingClientRect();
    lens.hidden = false;
    lens.style.left = `${x * rect.width}px`;
    lens.style.top = `${y * rect.height}px`;
  }

  return {
    live: false,
    accept: 'image/*',
    overlay: lens,

    async infer(model, frame, { onProgress, isStale }) {
      const before = inputCanvas(frame, state.size, T);
      const { width: w, height: hh } = before;
      const src = ctx2d(before).getImageData(0, 0, w, hh).data;
      const after = makeCanvas(w * S, hh * S);
      const out = new ImageData(w * S, hh * S);
      const cols = tilePlan(w, T);
      const rows = tilePlan(hh, T);
      const total = cols.length * rows.length;
      const plane = T * T;
      const input = new Float32Array(3 * plane);
      let done = 0;
      for (const r of rows) {
        for (const c of cols) {
          if (isStale()) return null;
          for (let y = 0; y < T; y++) {
            for (let x = 0; x < T; x++) {
              const p = ((r.at + y) * w + c.at + x) * 4;
              const i = y * T + x;
              input[i] = src[p] / 255;
              input[plane + i] = src[p + 1] / 255;
              input[2 * plane + i] = src[p + 2] / 255;
            }
          }
          const [res] = await model.run([{ data: input, shape: [1, 3, T, T] }]);
          const oT = T * S;
          const oPlane = oT * oT;
          for (let y = r.from * S; y < r.to * S; y++) {
            const ty = y - r.at * S;
            for (let x = c.from * S; x < c.to * S; x++) {
              const tx = x - c.at * S;
              const i = ty * oT + tx;
              const p = (y * w * S + x) * 4;
              out.data[p] = res.data[i] * 255;
              out.data[p + 1] = res.data[oPlane + i] * 255;
              out.data[p + 2] = res.data[2 * oPlane + i] * 255;
              out.data[p + 3] = 255;
            }
          }
          onProgress?.(++done / total, `${done} / ${total} tiles`);
        }
      }
      after.getContext('2d').putImageData(out, 0, 0);
      return { before, after, tiles: total };
    },

    split: () => ['Before', `After ${S}×`],
    animating: () => false,

    draw(canvas, frame, result, view) {
      current = result;
      if (!result) {
        const before = inputCanvas(frame, state.size, T);
        canvas.width = before.width * S;
        canvas.height = before.height * S;
        drawBefore(canvas.getContext('2d'), before, canvas.width, canvas.height);
        return;
      }
      const { before, after } = result;
      if (canvas.width !== after.width || canvas.height !== after.height) {
        canvas.width = after.width;
        canvas.height = after.height;
      }
      const g = canvas.getContext('2d');
      g.drawImage(after, 0, 0);
      const x = Math.round(view.split * canvas.width);
      g.save();
      g.beginPath();
      g.rect(0, 0, x, canvas.height);
      g.clip();
      drawBefore(g, before, canvas.width, canvas.height);
      g.restore();
      paintLens(canvas, result, view);
    },

    hover(pos, canvas, view) {
      state.lens = pos;
      paintLens(canvas, current, view);
    },

    controls() {
      return [
        segmented('Input size', SIZES, state.size, (v) => {
          state.size = v;
          env.rerun();
        }),
        segmented('Before', BEFORE, state.before, (v) => {
          state.before = v;
          env.redraw({ controls: true });
        }),
        h('p', { class: 'vision-panel__muted' }, `The picture is shrunk to the input size, then upscaled ${S}× in ${T} px tiles. Drag the divider to compare, or hover to magnify.`),
      ];
    },

    async exportImage(frame, result) {
      return result ? { canvas: result.after, name: 'upscaled-4x.png' } : null;
    },
  };
}
