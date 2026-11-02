// Segment Anything 2 on WebGPU: two .pte programs on the shared runtime.
//
// The heavy image encoder runs once per image, and its outputs stay in the
// runtime as decoder inputs. Every prompt after that runs only the small mask
// decoder, which is what makes clicking feel instant. The prompt encoder is
// tiny, so it runs here in JS: each point becomes a random-Fourier positional
// encoding plus a learned embedding for its kind (positive, negative, box
// corner, padding).
import { fetchWithProgress } from '../loader/fetch-progress.js';
import { f32, Program, readOutputs, startRuntime, withRuntime } from './executorch.js';

export const LABEL = { NEGATIVE: 0, POSITIVE: 1 };
const STABILITY_OFFSET = 1;

export class SegmentAnything {
  // stages reported through onProgress: runtime, weights (bytes), prompts, gpu
  static async load(source, { onProgress = () => {}, signal } = {}) {
    const { spec, url } = source;
    const aborted = () => {
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    };

    onProgress({ stage: 'runtime', fraction: 0 });
    const expected = (spec.encoderBytes ?? 0) + (spec.decoderBytes ?? 0);
    const got = { encoder: 0, decoder: 0 };
    const report = (key) => (p) => {
      got[key] = p.loaded;
      const loaded = got.encoder + got.decoder;
      onProgress({ stage: 'weights', fraction: expected ? Math.min(1, loaded / expected) : 0, loaded, total: expected, fromCache: p.fromCache });
    };
    const floats = (path) => fetchWithProgress(url(path), { signal }).then((b) => new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4));
    const p = spec.prompts;
    const [, encoderBytes, decoderBytes, tables] = await Promise.all([
      startRuntime().then(() => onProgress({ stage: 'runtime', fraction: 1 })),
      fetchWithProgress(url(spec.encoder), { signal, expectedBytes: spec.encoderBytes, onProgress: report('encoder') }),
      fetchWithProgress(url(spec.decoder), { signal, expectedBytes: spec.decoderBytes, onProgress: report('decoder') }),
      Promise.all([p.gaussian, p.pointNegative, p.pointPositive, p.boxTopLeft, p.boxBottomRight, p.notAPoint, p.imagePositional, p.dense].map(floats)).then((t) => {
        onProgress({ stage: 'prompts', fraction: 1 });
        return t;
      }),
    ]);
    aborted();

    onProgress({ stage: 'gpu', fraction: 0 });
    const [gaussian, negative, positive, boxTL, boxBR, notAPoint, imagePos, dense] = tables;
    const size = spec.imageSize ?? 512;
    const engine = await withRuntime(async (et) => {
      const encoder = await Program.load(et, encoderBytes, 'image encoder');
      onProgress({ stage: 'gpu', fraction: 0.5 });
      let decoder;
      try {
        decoder = await Program.load(et, decoderBytes, 'mask decoder');
      } catch (err) {
        encoder.delete();
        throw err;
      }
      // The decoder's inputs: image embeddings, positional grid, sparse prompt
      // [1, 1, tokens, hidden], dense prompt, then two high-resolution features.
      // The grid and the no-mask dense embedding are the same for every prompt.
      const shapes = decoder.inputShapes();
      const constants = { imagePos: f32(et, shapes[1], imagePos), dense: f32(et, shapes[3], dense) };
      return new SegmentAnything(encoder, decoder, constants, spec, { size, tokens: shapes[2][2] }, { gaussian, embeds: [negative, positive, boxTL, boxBR], notAPoint });
    });
    onProgress({ stage: 'gpu', fraction: 1 });
    if (signal?.aborted) engine.dispose();
    aborted();
    return engine;
  }

  constructor(encoder, decoder, constants, spec, { size, tokens }, prompt) {
    this.encoder = encoder;
    this.decoder = decoder;
    this.constants = constants;
    this.size = size;
    this.hidden = spec.hidden ?? 256;
    this.mean = spec.mean ?? [0.485, 0.456, 0.406];
    this.std = spec.std ?? [0.229, 0.224, 0.225];
    this.prompt = prompt;
    this.image = null;
    this.features = null;
    this.cache = new Map();
    // How many prompt tokens one decode takes; richer prompts are composed from
    // several decodes (see planPrompt).
    this.fixedTokens = tokens;
    this.decodeCount = 0;
    this.decodeMs = null;
    this.busy = Promise.resolve();
    this.pixels = new Float32Array(3 * size * size);
    const canvas = typeof OffscreenCanvas === 'function' ? new OffscreenCanvas(size, size) : Object.assign(document.createElement('canvas'), { width: size, height: size });
    this.ctx = canvas.getContext('2d', { willReadFrequently: true });
  }

  // One call into the runtime at a time; a failed call must not wedge the queue.
  #serial(fn) {
    const run = this.busy.then(fn);
    this.busy = run.catch(() => {});
    return run;
  }

  // Encodes an image (anything drawImage accepts). Prompts are in this image's
  // pixel coordinates until the next setImage.
  setImage(img, { width, height } = {}) {
    const w = width ?? img.naturalWidth ?? img.videoWidth ?? img.width;
    const hgt = height ?? img.naturalHeight ?? img.videoHeight ?? img.height;
    return this.#serial(async () => {
      this.image = null;
      this.cache = new Map();
      this.#preprocess(img);
      const t0 = performance.now();
      await withRuntime(async (et) => {
        this.#freeFeatures();
        const input = f32(et, [1, 3, this.size, this.size], this.pixels);
        let outputs;
        try {
          outputs = await this.encoder.forward([input]);
        } finally {
          input.delete();
        }
        // Copied out first: creating a tensor can grow the heap under a view.
        const [emb, hr0, hr1] = readOutputs(outputs);
        this.features = { emb: f32(et, emb.shape, emb.data), hr0: f32(et, hr0.shape, hr0.data), hr1: f32(et, hr1.shape, hr1.data) };
      });
      const ms = performance.now() - t0;
      this.image = { width: w, height: hgt };
      return { ms };
    });
  }

  #freeFeatures() {
    if (!this.features) return;
    for (const t of Object.values(this.features)) t.delete();
    this.features = null;
  }

  #preprocess(img) {
    const S = this.size;
    const ctx = this.ctx;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    // SAM 2 resizes to a square without keeping aspect; masks map back the same way.
    ctx.drawImage(img, 0, 0, S, S);
    const rgba = ctx.getImageData(0, 0, S, S).data;
    const plane = S * S;
    const px = this.pixels;
    const [m0, m1, m2] = this.mean;
    const [s0, s1, s2] = this.std;
    for (let i = 0; i < plane; i++) {
      px[i] = (rgba[i * 4] / 255 - m0) / s0;
      px[plane + i] = (rgba[i * 4 + 1] / 255 - m1) / s1;
      px[2 * plane + i] = (rgba[i * 4 + 2] / 255 - m2) / s2;
    }
  }

  // points: [{x, y, label}] with label 1 (include) or 0 (exclude); box: {x0, y0, x1, y1}
  // in image pixels. `detail` picks among the decoder's granularities: 'auto', or
  // 0..n-1 from smallest to largest. Returns combined logits at decoder resolution;
  // `mask` upsamples them to the image size on first access.
  segment({ points = [], box = null, detail = 'auto' } = {}) {
    return this.#serial(async () => {
      if (!this.image) throw new Error('Set an image before segmenting');
      const t0 = performance.now();
      const before = this.decodeCount;
      const units = planPrompt({ points, box, fixedTokens: this.fixedTokens });
      const decoded = [];
      for (const unit of units) {
        const keep = [];
        const cut = [];
        for (const t of unit.keep) keep.push(await this.#decode(t));
        for (const t of unit.cut) cut.push(await this.#decode(t));
        decoded.push({ keep, cut });
      }
      const result = composeMasks(decoded, { detail, ...this.image });
      result.ms = performance.now() - t0;
      result.decodes = this.decodeCount - before;
      return result;
    });
  }

  // Decodes one prompt (a list of tokens), memoised per image so undo and
  // re-picking a granularity never rerun the decoder.
  async #decode(tokens) {
    const key = JSON.stringify(tokens);
    const hit = this.cache.get(key);
    if (hit) return hit;
    if (tokens.length !== this.fixedTokens) throw new Error(`The mask decoder takes ${this.fixedTokens} prompt tokens, not ${tokens.length}`);
    const { width, height } = this.image;
    const data = encodePromptTokens(tokens, { width, height, size: this.size, hidden: this.hidden, ...this.prompt });
    const t0 = performance.now();
    const [masks] = await withRuntime(async (et) => {
      const { emb, hr0, hr1 } = this.features;
      const sparse = f32(et, [1, 1, tokens.length, this.hidden], data);
      try {
        return readOutputs(await this.decoder.forward([emb, this.constants.imagePos, sparse, this.constants.dense, hr0, hr1]));
      } finally {
        sparse.delete();
      }
    });
    this.decodeMs = performance.now() - t0;
    this.decodeCount++;
    // masks [1, n, h, w]: mask logits for mask tokens 1..n (part, object, group).
    const [n, lh, lw] = masks.shape.slice(-3);
    const out = { all: masks.data, n, lw, lh, candidates: scoreMasks(masks.data, n, lw * lh) };
    this.cache.set(key, out);
    return out;
  }

  dispose() {
    this.busy.finally(() =>
      withRuntime(() => {
        this.#freeFeatures();
        for (const t of Object.values(this.constants)) t.delete();
        this.encoder.delete();
        this.decoder.delete();
      }),
    );
  }
}

const tok = (kind, x = 0, y = 0) => ({ kind, x: Math.round(x * 10) / 10, y: Math.round(y * 10) / 10 });

// Turns a prompt into decoder calls. SAM 2's native prompt is every point and
// box corner, then one padding token ("not a point"). A decoder that takes that
// many tokens (or any number) gets exactly that, padded to its size. A smaller
// fixed decoder gets a composition of two-token prompts instead:
//   - each include point is decoded with each exclude point as its partner (no
//     excludes: with padding), and the pair masks are intersected;
//   - a box is decoded on its own, and each exclude point cuts out the smallest
//     confident segment under it;
//   - the object is the union of all of the above.
export function planPrompt({ points = [], box = null, fixedTokens = null }) {
  const pos = points.filter((p) => p.label !== LABEL.NEGATIVE).map((p) => tok('pos', p.x, p.y));
  const neg = points.filter((p) => p.label === LABEL.NEGATIVE).map((p) => tok('neg', p.x, p.y));
  const corners = box
    ? [tok('tl', Math.min(box.x0, box.x1), Math.min(box.y0, box.y1)), tok('br', Math.max(box.x0, box.x1), Math.max(box.y0, box.y1))]
    : [];
  if (!pos.length && !box) return [];
  const fill = (tokens) => (fixedTokens == null ? tokens : [...tokens, ...Array(Math.max(0, fixedTokens - tokens.length)).fill(tok('pad'))]);
  const native = [...pos, ...neg, ...corners, tok('pad')];
  if (fixedTokens == null || native.length <= fixedTokens) return [{ keep: [fill(native)], cut: [] }];
  const units = pos.map((p) => ({ keep: neg.length ? neg.map((n) => fill([p, n])) : [fill([p, tok('pad')])], cut: [] }));
  if (box) units.push({ keep: [fill(corners)], cut: neg.map((n) => fill([tok('pos', n.x, n.y), tok('pad')])) });
  return units;
}

// Combines decoded units in logit space (union = max, intersection = min,
// cut = min with the negated cut), which keeps a smooth zero crossing for the
// outline instead of a jagged boolean edge.
export function composeMasks(units, { detail = 'auto', width, height }) {
  const first = units[0]?.keep[0];
  const lw = first?.lw ?? 1;
  const lh = first?.lh ?? 1;
  const plane = lw * lh;
  const logits = new Float32Array(plane).fill(-20);
  const scores = [];
  for (const u of units) {
    const acc = new Float32Array(plane).fill(Infinity);
    for (const d of u.keep) {
      const c = pickMask(d.candidates, detail);
      scores.push(d.candidates[c].stability);
      const src = d.all.subarray(c * plane, (c + 1) * plane);
      for (let i = 0; i < plane; i++) if (src[i] < acc[i]) acc[i] = src[i];
    }
    for (const d of u.cut) {
      const c = smallestConfident(d.candidates);
      const src = d.all.subarray(c * plane, (c + 1) * plane);
      for (let i = 0; i < plane; i++) if (-src[i] < acc[i]) acc[i] = -src[i];
    }
    for (let i = 0; i < plane; i++) if (acc[i] > logits[i]) logits[i] = acc[i];
  }
  let on = 0;
  for (let i = 0; i < plane; i++) if (logits[i] > 0) on++;
  let mask = null;
  return {
    logits,
    logitsWidth: lw,
    logitsHeight: lh,
    granularities: first?.n ?? 0,
    score: scores.length ? Math.min(...scores) : 0,
    area: on / plane,
    width,
    height,
    get mask() {
      mask ??= upsampleMask(logits, lw, lh, width, height);
      return mask;
    },
  };
}

function pickMask(candidates, detail) {
  if (detail === 'auto' || detail == null) return bestMask(candidates);
  const bySize = [...candidates].sort((a, b) => a.area - b.area);
  return bySize[Math.max(0, Math.min(bySize.length - 1, detail))].index;
}

function smallestConfident(candidates) {
  const ok = candidates.filter((c) => c.area > 0.0005 && c.stability > 0.6).sort((a, b) => a.area - b.area);
  return (ok[0] ?? [...candidates].sort((a, b) => a.area - b.area)[0]).index;
}

const KIND = { neg: 0, pos: 1, tl: 2, br: 3 };

// Builds the decoder's sparse prompt tensor [1, 1, tokens, hidden]. Coordinates are
// mapped into the square encoder frame and encoded as [sin(2πcG), cos(2πcG)] plus
// the learned embedding for the token's kind; padding is "not a point" alone.
export function encodePromptTokens(tokens, { width, height, size, hidden, gaussian, embeds, notAPoint }) {
  const D = hidden / 2;
  const data = new Float32Array(tokens.length * hidden);
  tokens.forEach((t, row) => {
    const o = row * hidden;
    if (t.kind === 'pad') {
      data.set(notAPoint.subarray(0, hidden), o);
      return;
    }
    const embed = embeds[KIND[t.kind]];
    const cx = (2 * (t.x * (size / width) + 0.5)) / size - 1;
    const cy = (2 * (t.y * (size / height) + 0.5)) / size - 1;
    for (let k = 0; k < D; k++) {
      const a = 2 * Math.PI * (cx * gaussian[k] + cy * gaussian[D + k]);
      data[o + k] = Math.sin(a) + embed[k];
      data[o + D + k] = Math.cos(a) + embed[D + k];
    }
  });
  return data;
}

// Stability = IoU between the mask thresholded at +1 and at -1: a mask whose
// boundary barely moves when the threshold does is a confident one.
export function scoreMasks(all, n, plane) {
  const out = [];
  for (let c = 0; c < n; c++) {
    let pos = 0;
    let hi = 0;
    let lo = 0;
    const off = c * plane;
    for (let i = 0; i < plane; i++) {
      const v = all[off + i];
      if (v > 0) pos++;
      if (v > STABILITY_OFFSET) hi++;
      if (v > -STABILITY_OFFSET) lo++;
    }
    out.push({ index: c, area: pos / plane, stability: lo ? hi / lo : 0 });
  }
  return out;
}

// The decoder returns several granularities (part, object, group). Its IoU head
// often disagrees with upstream SAM 2 at this input size, so prefer stable masks
// and break ties toward the whole object rather than a sliver. Near-empty and
// near-full masks lose.
export function bestMask(candidates) {
  let best = 0;
  let bestScore = -Infinity;
  for (const c of candidates) {
    if (c.area < 0.0005 || c.area > 0.97) continue;
    const s = c.stability + Math.min(c.area, 0.5);
    if (s > bestScore) {
      bestScore = s;
      best = c.index;
    }
  }
  return best;
}

// Bilinear upsample of mask logits (cell-centred, like align_corners=false), then
// threshold at 0. Returns 0/1 per pixel.
export function upsampleMask(logits, lw, lh, width, height) {
  const out = new Uint8Array(width * height);
  const sx = lw / width;
  const sy = lh / height;
  for (let y = 0; y < height; y++) {
    const fy = Math.min(lh - 1, Math.max(0, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(lh - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < width; x++) {
      const fx = Math.min(lw - 1, Math.max(0, (x + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(lw - 1, x0 + 1);
      const tx = fx - x0;
      const top = logits[y0 * lw + x0] * (1 - tx) + logits[y0 * lw + x1] * tx;
      const bot = logits[y1 * lw + x0] * (1 - tx) + logits[y1 * lw + x1] * tx;
      out[y * width + x] = top * (1 - ty) + bot * ty > 0 ? 1 : 0;
    }
  }
  return out;
}

// Sub-pixel outlines of the zero level set, as closed polylines in logit-grid
// coordinates (cell centres at integers). Marching squares over a grid padded
// with background, so every contour closes; filled with the even-odd rule, the
// loops reproduce holes too.
export function maskContours(logits, w, h) {
  const W = w + 2;
  const H = h + 2;
  const v = (x, y) => (x <= 0 || y <= 0 || x > w || y > h ? -1 : logits[(y - 1) * w + (x - 1)]);
  // Edge ids: horizontal edge (x,y)-(x+1,y) -> 2*(y*W+x); vertical (x,y)-(x,y+1) -> +1.
  const point = new Map();
  const cross = (id, ax, ay, bx, by) => {
    if (!point.has(id)) {
      const a = v(ax, ay);
      const b = v(bx, by);
      const t = a === b ? 0.5 : a / (a - b);
      point.set(id, [ax + (bx - ax) * t - 1, ay + (by - ay) * t - 1]);
    }
    return id;
  };
  const links = new Map();
  const link = (a, b) => {
    (links.get(a) ?? links.set(a, []).get(a)).push(b);
    (links.get(b) ?? links.set(b, []).get(b)).push(a);
  };
  for (let y = 0; y < H - 1; y++) {
    for (let x = 0; x < W - 1; x++) {
      const tl = v(x, y) > 0;
      const tr = v(x + 1, y) > 0;
      const br = v(x + 1, y + 1) > 0;
      const bl = v(x, y + 1) > 0;
      const code = (tl ? 8 : 0) | (tr ? 4 : 0) | (br ? 2 : 0) | (bl ? 1 : 0);
      if (code === 0 || code === 15) continue;
      const top = () => cross(2 * (y * W + x), x, y, x + 1, y);
      const bottom = () => cross(2 * ((y + 1) * W + x), x, y + 1, x + 1, y + 1);
      const left = () => cross(2 * (y * W + x) + 1, x, y, x, y + 1);
      const right = () => cross(2 * (y * W + x + 1) + 1, x + 1, y, x + 1, y + 1);
      switch (code) {
        case 1: case 14: link(left(), bottom()); break;
        case 2: case 13: link(bottom(), right()); break;
        case 3: case 12: link(left(), right()); break;
        case 4: case 11: link(top(), right()); break;
        case 6: case 9: link(top(), bottom()); break;
        case 7: case 8: link(left(), top()); break;
        case 5: case 10: {
          const centre = (v(x, y) + v(x + 1, y) + v(x + 1, y + 1) + v(x, y + 1)) / 4 > 0;
          if ((code === 5) === centre) {
            link(left(), top());
            link(bottom(), right());
          } else {
            link(left(), bottom());
            link(top(), right());
          }
          break;
        }
      }
    }
  }
  const loops = [];
  const seen = new Set();
  for (const start of links.keys()) {
    if (seen.has(start)) continue;
    const loop = [];
    let prev = -1;
    let cur = start;
    while (cur !== undefined && !seen.has(cur)) {
      seen.add(cur);
      loop.push(point.get(cur));
      const next = links.get(cur).find((n) => n !== prev && !seen.has(n));
      prev = cur;
      cur = next;
    }
    if (loop.length >= 3) loops.push(loop);
  }
  return loops;
}
