// Portrait matting: MODNet predicts a soft alpha matte for a [-1, 1]-normalized
// square input, which is stretched back over the full-resolution frame.
import { h } from '../lib/dom.js';
import { segmented } from './controls.js';
import { ctx2d, loadImage, makeCanvas, rasterize, sizeOf, toTensor } from './image.js';

const NORM = { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] };
const VIEWS = [
  { id: 'result', label: 'Result' },
  { id: 'split', label: 'Compare' },
  { id: 'matte', label: 'Matte' },
];
const BACKDROPS = [
  { id: 'original', label: 'None', swatch: 'original' },
  { id: 'none', label: 'Transparent', swatch: 'checker' },
  { id: 'blur', label: 'Blur', swatch: 'blur' },
  { id: 'sunset', label: 'Sunset', stops: ['#ffb36b', '#ff6a88', '#7b4bd6'] },
  { id: 'ocean', label: 'Ocean', stops: ['#5ee7df', '#2a78d6', '#1b2a6b'] },
  { id: 'mint', label: 'Mint', stops: ['#e6fff4', '#a8edea', '#7fd1b9'] },
  { id: 'paper', label: 'Paper', color: '#f3efe6' },
  { id: 'studio', label: 'Studio', color: '#1d2433' },
  { id: 'image', label: 'Your image' },
];

export function createMattingTask(env) {
  const [W, H] = env.spec.inputSize ?? [256, 256];
  const state = { view: 'result', backdrop: 'none', bgImage: null };
  const fileInput = h('input', { type: 'file', accept: 'image/*', hidden: true, onchange: (e) => e.target.files[0] && useBackground(e.target.files[0]) });
  const buffers = [];
  let flip = 0;
  let fg = null;
  const fit = (c, w, hh) => (c && c.width === w && c.height === hh ? c : makeCanvas(w, hh));
  let blurA = null;
  let blurB = null;

  async function useBackground(file) {
    const url = URL.createObjectURL(file);
    try {
      state.bgImage = rasterize(await loadImage(url), 2048);
      state.backdrop = 'image';
    } catch (err) {
      env.error(err.message);
    } finally {
      URL.revokeObjectURL(url);
      fileInput.value = '';
    }
    env.redraw({ controls: true });
  }

  function paintBackdrop(g, w, hh, photo) {
    const b = BACKDROPS.find((x) => x.id === state.backdrop);
    if (b.id === 'original') return g.drawImage(photo, 0, 0, w, hh);
    if (b.id === 'none') return g.clearRect(0, 0, w, hh);
    if (b.id === 'blur') {
      // Downscale-then-upscale blur works in every browser, unlike ctx.filter.
      const sw = Math.max(8, Math.round(w / 28));
      const sh = Math.max(8, Math.round(hh / 28));
      blurA = blurA ?? makeCanvas(1, 1);
      blurB = blurB ?? makeCanvas(1, 1);
      blurA.width = sw * 4;
      blurA.height = sh * 4;
      blurB.width = sw;
      blurB.height = sh;
      const ga = blurA.getContext('2d');
      const gb = blurB.getContext('2d');
      ga.imageSmoothingQuality = gb.imageSmoothingQuality = 'high';
      ga.drawImage(photo, 0, 0, blurA.width, blurA.height);
      gb.drawImage(blurA, 0, 0, sw, sh);
      g.imageSmoothingQuality = 'high';
      g.drawImage(blurB, 0, 0, w, hh);
      return;
    }
    if (b.id === 'image') {
      if (!state.bgImage) return g.clearRect(0, 0, w, hh);
      const [iw, ih] = sizeOf(state.bgImage);
      const s = Math.max(w / iw, hh / ih);
      return g.drawImage(state.bgImage, (w - iw * s) / 2, (hh - ih * s) / 2, iw * s, ih * s);
    }
    if (b.color) {
      g.fillStyle = b.color;
      return g.fillRect(0, 0, w, hh);
    }
    const grad = g.createLinearGradient(0, 0, w, hh);
    b.stops.forEach((c, i) => grad.addColorStop(i / (b.stops.length - 1), c));
    g.fillStyle = grad;
    g.fillRect(0, 0, w, hh);
  }

  function composite(target, photo, result) {
    const { width: w, height: hh } = photo;
    fg = fg && fg.width === w && fg.height === hh ? fg : makeCanvas(w, hh);
    const fgc = fg.getContext('2d');
    fgc.globalCompositeOperation = 'copy';
    fgc.drawImage(photo, 0, 0);
    fgc.globalCompositeOperation = 'destination-in';
    fgc.imageSmoothingQuality = 'high';
    fgc.drawImage(result.alpha, 0, 0, w, hh);
    fgc.globalCompositeOperation = 'source-over';
    const g = target.getContext('2d');
    paintBackdrop(g, w, hh, photo);
    g.drawImage(fg, 0, 0);
  }

  function matte(target, result) {
    const g = target.getContext('2d');
    g.fillStyle = '#000';
    g.fillRect(0, 0, target.width, target.height);
    g.imageSmoothingQuality = 'high';
    g.drawImage(result.gray, 0, 0, target.width, target.height);
  }

  return {
    live: true,
    accept: 'image/*,video/*',

    async infer(model, frame, { live }) {
      const { tensor } = toTensor(frame, W, H, NORM);
      // A live frame canvas is redrawn every animation frame, so keep a copy of the one
      // inferred on. Two buffer sets alternate: one on screen, one being filled.
      const buf = live ? (buffers[flip ^= 1] ??= {}) : {};
      let photo = frame;
      if (live) {
        photo = buf.photo = fit(buf.photo, frame.width, frame.height);
        photo.getContext('2d').drawImage(frame, 0, 0);
      }
      const [out] = await model.run([tensor]);
      const [mh, mw] = out.shape.slice(-2);
      const alpha = (buf.alpha = fit(buf.alpha, mw, mh));
      const gray = (buf.gray = fit(buf.gray, mw, mh));
      const ga = ctx2d(alpha);
      const gg = ctx2d(gray);
      const ia = ga.createImageData(mw, mh);
      const ig = gg.createImageData(mw, mh);
      for (let i = 0, p = 0; i < mw * mh; i++, p += 4) {
        const v = Math.round(Math.min(1, Math.max(0, out.data[i])) * 255);
        ia.data[p + 3] = v;
        ig.data[p] = ig.data[p + 1] = ig.data[p + 2] = v;
        ig.data[p + 3] = 255;
      }
      ga.putImageData(ia, 0, 0);
      gg.putImageData(ig, 0, 0);
      return { alpha, gray, photo };
    },

    split() {
      return state.view === 'split' ? ['Original', 'Cutout'] : null;
    },

    animating: () => false,

    transparent() {
      return state.view !== 'matte' && state.backdrop === 'none';
    },

    draw(canvas, frame, result, view) {
      const photo = result?.photo ?? frame;
      if (canvas.width !== photo.width || canvas.height !== photo.height) {
        canvas.width = photo.width;
        canvas.height = photo.height;
      }
      const g = canvas.getContext('2d');
      if (!result) {
        g.clearRect(0, 0, canvas.width, canvas.height);
        return g.drawImage(photo, 0, 0);
      }
      if (state.view === 'matte') return matte(canvas, result);
      composite(canvas, photo, result);
      if (state.view === 'split') {
        const x = Math.round(view.split * canvas.width);
        g.clearRect(0, 0, x, canvas.height);
        g.drawImage(photo, 0, 0, x, photo.height, 0, 0, x, canvas.height);
      }
    },

    controls() {
      return [
        segmented('View', VIEWS, state.view, (v) => {
          state.view = v;
          env.redraw({ controls: true });
        }),
        h(
          'div',
          { class: 'vision-control' },
          h('span', { class: 'vision-control__label' }, 'Background'),
          h(
            'div',
            { class: 'vision-swatches', role: 'group', 'aria-label': 'Background' },
            BACKDROPS.map((b) => {
              const active = b.id === state.backdrop;
              const style = b.color
                ? { background: b.color }
                : b.stops
                  ? { background: `linear-gradient(135deg, ${b.stops.join(', ')})` }
                  : b.id === 'image' && state.bgImage
                    ? { backgroundImage: `url(${state.bgImage.toDataURL('image/jpeg', 0.6)})` }
                    : null;
              const cls =
                b.swatch === 'original'
                  ? 'vision-swatch vision-swatch--original'
                  : b.swatch === 'checker'
                  ? 'vision-swatch vision-swatch--checker'
                  : b.swatch === 'blur'
                    ? 'vision-swatch vision-swatch--blur'
                    : b.id === 'image' && !state.bgImage
                      ? 'vision-swatch vision-swatch--add'
                      : 'vision-swatch';
              return h(
                'button',
                {
                  class: active ? `${cls} is-active` : cls,
                  type: 'button',
                  title: b.id === 'image' ? 'Use your own background image' : b.label,
                  'aria-label': b.label,
                  'aria-pressed': String(active),
                  style,
                  onclick: () => {
                    if (b.id === 'image' && (!state.bgImage || active)) return fileInput.click();
                    state.backdrop = b.id;
                    env.redraw({ controls: true });
                  },
                },
                b.id === 'image' && !state.bgImage ? '+' : null,
              );
            }),
          ),
          fileInput,
        ),
      ];
    },

    async exportImage(frame, result) {
      if (!result) return null;
      const photo = result.photo ?? frame;
      const c = makeCanvas(photo.width, photo.height);
      if (state.view === 'matte') matte(c, result);
      else composite(c, photo, result);
      return { canvas: c, name: state.view === 'matte' ? 'matte.png' : 'cutout.png' };
    },
  };
}
