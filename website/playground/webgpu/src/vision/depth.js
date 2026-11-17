// Monocular depth: Depth Anything V2 predicts relative inverse depth (larger is
// nearer) for an ImageNet-normalized square input.
import { h } from '../lib/dom.js';
import { colormapCss, colormapLut, COLORMAPS } from './colormap.js';
import { ctx2d, makeCanvas, toTensor } from './image.js';
import { segmented } from './controls.js';

const IMAGENET = { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };
const VIEWS = [
  { id: 'depth', label: 'Depth' },
  { id: 'split', label: 'Compare' },
  { id: '3d', label: '3D' },
];

// Percentile bounds from a strided sample: robust to a few extreme pixels.
function bounds(data) {
  const step = Math.max(1, Math.floor(data.length / 6000));
  const sample = [];
  for (let i = 0; i < data.length; i += step) sample.push(data[i]);
  sample.sort((a, b) => a - b);
  return [sample[Math.floor(sample.length * 0.01)], sample[Math.floor(sample.length * 0.99)]];
}

export function createDepthTask(env) {
  const [W, H] = env.spec.inputSize ?? [224, 224];
  const state = { view: 'depth', cmap: 'turbo', lo: null, hi: null, mapKey: null, yaw: 0, pitch: 0, dragYaw: 0, dragPitch: 0, dragging: false, t0: performance.now() };
  let mapCanvas = null;
  let splat = null;

  function paintMap(result) {
    const key = `${state.cmap}|${state.lo}|${state.hi}`;
    if (result.mapKey === key && mapCanvas) return mapCanvas;
    const { depth, w, h } = result;
    mapCanvas = mapCanvas && mapCanvas.width === w && mapCanvas.height === h ? mapCanvas : makeCanvas(w, h);
    const g = ctx2d(mapCanvas);
    const img = g.createImageData(w, h);
    const lut = colormapLut(state.cmap);
    const range = state.hi - state.lo || 1;
    for (let i = 0, p = 0; i < w * h; i++, p += 4) {
      let t = (depth[i] - state.lo) / range;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const k = Math.round(t * 255) * 3;
      img.data[p] = lut[k];
      img.data[p + 1] = lut[k + 1];
      img.data[p + 2] = lut[k + 2];
      img.data[p + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    result.mapKey = key;
    return mapCanvas;
  }

  // Depth resampled onto a grid finer than the model's output, normalized to [0, 1].
  function reliefGrid(result, gw, gh) {
    const key = `${gw}x${gh}|${state.lo}|${state.hi}`;
    if (result.grid?.key === key) return result.grid.d;
    const { depth, w, h } = result;
    const d = new Float32Array(gw * gh);
    const range = state.hi - state.lo || 1;
    for (let j = 0; j < gh; j++) {
      const fy = (j / (gh - 1)) * (h - 1);
      const y0 = Math.floor(fy);
      const y1 = Math.min(h - 1, y0 + 1);
      const ty = fy - y0;
      for (let i = 0; i < gw; i++) {
        const fx = (i / (gw - 1)) * (w - 1);
        const x0 = Math.floor(fx);
        const x1 = Math.min(w - 1, x0 + 1);
        const tx = fx - x0;
        const top = depth[y0 * w + x0] * (1 - tx) + depth[y0 * w + x1] * tx;
        const bottom = depth[y1 * w + x0] * (1 - tx) + depth[y1 * w + x1] * tx;
        const v = (top * (1 - ty) + bottom * ty - state.lo) / range;
        d[j * gw + i] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
    }
    result.grid = { key, d };
    return d;
  }

  // A photo-textured relief, splatted point by point into a z-buffer.
  function draw3d(canvas, frame, result, live) {
    const aspect = frame.width / frame.height;
    const cw = Math.round(Math.min(960, Math.max(480, frame.width)));
    const ch = Math.round(cw / aspect);
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw;
      canvas.height = ch;
    }
    const gw = Math.min(frame.width, Math.round(cw / 2));
    const gh = Math.max(2, Math.round(gw / aspect));
    if (!splat || splat.w !== cw || splat.h !== ch || splat.gw !== gw) {
      const img = new ImageData(cw, ch);
      splat = { w: cw, h: ch, gw, gh, img, px: new Uint32Array(img.data.buffer), z: new Float32Array(cw * ch), tex: makeCanvas(gw, gh), texFor: null };
    }
    if (live || splat.texFor !== frame) {
      const g = ctx2d(splat.tex);
      g.imageSmoothingQuality = 'high';
      g.drawImage(frame, 0, 0, gw, gh);
      splat.colors = new Uint32Array(g.getImageData(0, 0, gw, gh).data.buffer);
      splat.texFor = frame;
    }
    const d = reliefGrid(result, gw, gh);
    const t = (performance.now() - state.t0) / 1000;
    const yaw = state.dragging ? state.dragYaw : state.dragYaw + 0.3 * Math.sin(t * 0.55);
    const pitch = state.dragging ? state.dragPitch : state.dragPitch + 0.12 * Math.sin(t * 0.4);
    const { px, z, img, colors } = splat;
    px.fill(0xff120b0b);
    z.fill(-Infinity);
    const ax = aspect >= 1 ? 1 : aspect;
    const ay = aspect >= 1 ? 1 / aspect : 1;
    const relief = 0.42;
    const dist = 2.4;
    const scale = (cw / ax) * 0.86;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const size = Math.max(2, Math.ceil((cw / gw) * 1.5));
    const half = size >> 1;
    for (let j = 0; j < gh; j++) {
      const y = (j / (gh - 1) - 0.5) * ay;
      for (let i = 0; i < gw; i++) {
        const k = j * gw + i;
        const dk = d[k];
        const x = (i / (gw - 1) - 0.5) * ax;
        const zz = (dk - 0.5) * relief;
        const x1 = x * cy + zz * sy;
        const z1 = -x * sy + zz * cy;
        const y2 = y * cp - z1 * sp;
        const z2 = y * sp + z1 * cp;
        const f = dist / (dist - z2);
        const sx = Math.round(cw / 2 + x1 * scale * f) - half;
        const syy = Math.round(ch / 2 + y2 * scale * f) - half;
        const shade = 0.55 + 0.45 * dk;
        const c = colors[k];
        const color = 0xff000000 | (((c >>> 16) & 255) * shade) << 16 | (((c >>> 8) & 255) * shade) << 8 | ((c & 255) * shade);
        for (let v = 0; v < size; v++) {
          const yy = syy + v;
          if (yy < 0 || yy >= ch) continue;
          const row = yy * cw;
          for (let u = 0; u < size; u++) {
            const xx = sx + u;
            if (xx < 0 || xx >= cw) continue;
            const o = row + xx;
            if (z2 > z[o]) {
              z[o] = z2;
              px[o] = color;
            }
          }
        }
      }
    }
    canvas.getContext('2d').putImageData(img, 0, 0);
  }

  return {
    live: true,
    accept: 'image/*,video/*',

    async infer(model, frame, { live }) {
      const { tensor } = toTensor(frame, W, H, IMAGENET);
      const [out] = await model.run([tensor]);
      const [h, w] = out.shape.slice(-2);
      const [lo, hi] = bounds(out.data);
      // Live video eases the range toward each frame's so the colors don't flicker.
      const a = live && state.lo != null ? 0.25 : 1;
      state.lo = state.lo == null || a === 1 ? lo : state.lo + (lo - state.lo) * a;
      state.hi = state.hi == null || a === 1 ? hi : state.hi + (hi - state.hi) * a;
      return { depth: out.data, w, h };
    },

    split() {
      return state.view === 'split' ? ['Photo', 'Depth'] : null;
    },

    animating() {
      return state.view === '3d';
    },

    draw(canvas, frame, result, view) {
      if (state.view === '3d' && result) return draw3d(canvas, frame, result, view.live);
      if (canvas.width !== frame.width || canvas.height !== frame.height) {
        canvas.width = frame.width;
        canvas.height = frame.height;
      }
      const g = canvas.getContext('2d');
      g.imageSmoothingQuality = 'high';
      if (!result) return g.drawImage(frame, 0, 0);
      const map = paintMap(result);
      g.drawImage(map, 0, 0, canvas.width, canvas.height);
      if (state.view === 'split') {
        const x = Math.round(view.split * canvas.width);
        g.drawImage(frame, 0, 0, x, frame.height, 0, 0, x, canvas.height);
      }
    },

    drag(dx, dy, phase) {
      if (state.view !== '3d') return false;
      if (phase === 'start') {
        const t = (performance.now() - state.t0) / 1000;
        state.dragYaw += 0.3 * Math.sin(t * 0.55);
        state.dragPitch += 0.12 * Math.sin(t * 0.4);
        state.dragging = true;
      } else if (phase === 'move') {
        state.dragYaw = Math.max(-0.9, Math.min(0.9, state.dragYaw + dx * 2.2));
        state.dragPitch = Math.max(-0.6, Math.min(0.6, state.dragPitch + dy * 1.6));
      } else {
        state.dragging = false;
        state.t0 = performance.now();
        state.dragYaw *= 0.5;
        state.dragPitch *= 0.5;
      }
      return true;
    },

    legend() {
      if (state.view === '3d') return h('span', { class: 'vision-legend__text' }, 'Drag to tilt');
      return [
        h('span', { class: 'vision-legend__text' }, 'Far'),
        h('span', { class: 'vision-legend__ramp', style: { background: colormapCss(state.cmap) } }),
        h('span', { class: 'vision-legend__text' }, 'Near'),
      ];
    },

    controls() {
      return [
        segmented('View', VIEWS, state.view, (v) => {
          state.view = v;
          env.redraw({ controls: true });
        }),
        state.view === '3d'
          ? null
          : segmented('Colors', COLORMAPS, state.cmap, (v) => {
              state.cmap = v;
              env.redraw({ controls: true });
            }),
      ];
    },

    async exportImage(frame, result) {
      if (!result) return null;
      const c = makeCanvas(frame.width, frame.height);
      const g = c.getContext('2d');
      g.imageSmoothingQuality = 'high';
      g.drawImage(paintMap(result), 0, 0, c.width, c.height);
      return { canvas: c, name: 'depth.png' };
    },

    resetLive() {
      state.lo = state.hi = null;
    },
  };
}
