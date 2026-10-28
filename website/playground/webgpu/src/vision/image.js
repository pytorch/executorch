// Pixel plumbing shared by the vision tasks: decoding inputs, resizing, and
// converting between canvases and CHW float tensors.

export function sizeOf(el) {
  return [el.videoWidth || el.naturalWidth || el.width, el.videoHeight || el.naturalHeight || el.height];
}

export function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(w));
  c.height = Math.max(1, Math.round(h));
  return c;
}

export function ctx2d(canvas) {
  return canvas.getContext('2d', { willReadFrequently: true });
}

export function fitWithin(w, h, max) {
  const s = Math.min(1, max / Math.max(w, h));
  return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
}

export function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('That image could not be decoded.'));
    img.src = url;
  });
}

// Draws any image-like source into a new canvas no larger than `max` on its longest side.
export function rasterize(el, max, { mirror = false } = {}) {
  const [w, h] = fitWithin(...sizeOf(el), max);
  const c = makeCanvas(w, h);
  const g = c.getContext('2d');
  if (mirror) {
    g.translate(w, 0);
    g.scale(-1, 1);
  }
  g.drawImage(el, 0, 0, w, h);
  return c;
}

const scratch = new Map();
function scratchCanvas(key, w, h) {
  let c = scratch.get(key);
  if (!c) scratch.set(key, (c = makeCanvas(w, h)));
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
  }
  return c;
}

// Resizes `el` to W x H (stretching, or letterboxing with a gray pad) and returns a
// [1,3,H,W] tensor of (pixel / 255 - mean) / std, the RGBA pixels it was made from,
// and the letterbox transform (frame px = (model px - offset) / scale).
export function toTensor(el, W, H, { mean = [0, 0, 0], std = [1, 1, 1], letterbox = false } = {}) {
  const c = scratchCanvas(`t${W}x${H}`, W, H);
  const g = ctx2d(c);
  const [sw, sh] = sizeOf(el);
  let scale = [W / sw, H / sh];
  let offset = [0, 0];
  if (letterbox) {
    const s = Math.min(W / sw, H / sh);
    scale = [s, s];
    offset = [(W - sw * s) / 2, (H - sh * s) / 2];
    g.fillStyle = 'rgb(114,114,114)';
    g.fillRect(0, 0, W, H);
  }
  g.imageSmoothingQuality = 'high';
  g.drawImage(el, offset[0], offset[1], sw * scale[0], sh * scale[1]);
  const px = g.getImageData(0, 0, W, H).data;
  const plane = W * H;
  const data = new Float32Array(3 * plane);
  const k = [1 / (255 * std[0]), 1 / (255 * std[1]), 1 / (255 * std[2])];
  const b = [mean[0] / std[0], mean[1] / std[1], mean[2] / std[2]];
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    data[i] = px[p] * k[0] - b[0];
    data[plane + i] = px[p + 1] * k[1] - b[1];
    data[2 * plane + i] = px[p + 2] * k[2] - b[2];
  }
  return { tensor: { data, shape: [1, 3, H, W] }, pixels: px, scale, offset };
}

export function canvasToBlob(canvas, type = 'image/png') {
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not encode the image.'))), type));
}

export function downloadBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}
