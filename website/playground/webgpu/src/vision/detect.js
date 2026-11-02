// Object detection: a letterboxed RGB input in [0, 1], and
// an output of [1, 4 + classes, anchors] holding center-size boxes in input pixels
// followed by per-class scores that are already sigmoid-activated.
import { h } from '../lib/dom.js';
import { COCO_CLASSES } from './coco.js';
import { slider } from './controls.js';
import { toTensor } from './image.js';

const FLOOR = 0.05;

export function decodeDetections(data, shape, { floor = FLOOR, scale = [1, 1], offset = [0, 0] } = {}) {
  const [rows, anchors] = shape.slice(-2);
  const classes = rows - 4;
  const out = [];
  for (let a = 0; a < anchors; a++) {
    let best = 0;
    let cls = -1;
    for (let c = 0; c < classes; c++) {
      const s = data[(4 + c) * anchors + a];
      if (s > best) {
        best = s;
        cls = c;
      }
    }
    if (best < floor) continue;
    const cx = (data[a] - offset[0]) / scale[0];
    const cy = (data[anchors + a] - offset[1]) / scale[1];
    const w = data[2 * anchors + a] / scale[0];
    const hh = data[3 * anchors + a] / scale[1];
    out.push({ x: cx - w / 2, y: cy - hh / 2, w, h: hh, score: best, cls });
  }
  return out.sort((p, q) => q.score - p.score);
}

function iou(a, b) {
  const iw = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const ih = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = iw * ih;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

// Class-wise non-maximum suppression over boxes sorted by descending score.
export function nms(boxes, threshold = 0.45) {
  const keep = [];
  for (const b of boxes) {
    if (!keep.some((k) => k.cls === b.cls && iou(k, b) > threshold)) keep.push(b);
  }
  return keep;
}

export function classColor(cls) {
  return `hsl(${(cls * 137.5 + 18) % 360} 88% 58%)`;
}

function roundRect(g, x, y, w, hh, r) {
  g.beginPath();
  if (g.roundRect) g.roundRect(x, y, w, hh, r);
  else g.rect(x, y, w, hh);
}

export function createDetectTask(env) {
  const [W, H] = env.spec.inputSize ?? [640, 640];
  const state = { conf: 0.35, boxes: [], boxesFor: null };

  function visible(result) {
    const key = `${state.conf}`;
    if (state.boxesFor !== result || state.key !== key) {
      state.boxes = nms(result.candidates.filter((b) => b.score >= state.conf));
      state.boxesFor = result;
      state.key = key;
    }
    return state.boxes;
  }

  return {
    live: true,
    accept: 'image/*,video/*',

    async infer(model, frame) {
      const { tensor, scale, offset } = toTensor(frame, W, H, { letterbox: true });
      const [out] = await model.run([tensor]);
      return { candidates: decodeDetections(out.data, out.shape, { scale, offset }) };
    },

    split: () => null,
    animating: () => false,

    draw(canvas, frame, result) {
      if (canvas.width !== frame.width || canvas.height !== frame.height) {
        canvas.width = frame.width;
        canvas.height = frame.height;
      }
      const g = canvas.getContext('2d');
      g.drawImage(frame, 0, 0);
      if (!result) return;
      const k = canvas.width / Math.max(1, canvas.clientWidth || canvas.width);
      const font = Math.round(13 * k);
      g.font = `600 ${font}px ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif`;
      g.textBaseline = 'middle';
      g.lineJoin = 'round';
      for (const b of visible(result)) {
        const color = classColor(b.cls);
        g.lineWidth = 2.5 * k;
        g.strokeStyle = color;
        g.fillStyle = color.replace(')', ' / 0.12)');
        roundRect(g, b.x, b.y, b.w, b.h, 6 * k);
        g.fill();
        g.stroke();
        const label = `${COCO_CLASSES[b.cls] ?? `class ${b.cls}`} ${Math.round(b.score * 100)}%`;
        const tw = g.measureText(label).width + 14 * k;
        const th = font + 10 * k;
        const lx = Math.min(Math.max(0, b.x - 1.25 * k), canvas.width - tw);
        const ly = b.y - th - 3 * k < 0 ? b.y + 3 * k : b.y - th - 3 * k;
        g.fillStyle = color;
        roundRect(g, lx, ly, tw, th, 6 * k);
        g.fill();
        g.fillStyle = '#0b0b10';
        g.fillText(label, lx + 7 * k, ly + th / 2 + 0.5 * k);
      }
    },

    controls() {
      return [
        slider('Confidence', {
          min: 0.05,
          max: 0.95,
          step: 0.05,
          value: state.conf,
          format: (v) => `${Math.round(v * 100)}%`,
          onInput: (v) => {
            state.conf = v;
            env.redraw({ summary: true });
          },
        }),
      ];
    },

    summary(result) {
      if (!result) return null;
      const counts = new Map();
      for (const b of visible(result)) counts.set(b.cls, (counts.get(b.cls) ?? 0) + 1);
      if (!counts.size) return h('p', { class: 'vision-panel__muted' }, 'Nothing above the confidence threshold. Try lowering it.');
      return h(
        'div',
        { class: 'vision-chips' },
        [...counts.entries()]
          .sort((a, b) => b[1] - a[1])
          .map(([cls, n]) =>
            h('span', { class: 'vision-chip' }, h('span', { class: 'vision-chip__dot', style: { background: classColor(cls) } }), COCO_CLASSES[cls] ?? `class ${cls}`, h('span', { class: 'vision-chip__n' }, String(n))),
          ),
      );
    },
  };
}
