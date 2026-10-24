import { getModel } from '../config/models.js';
import { gpuState } from '../gpu/context.js';
import { h, replace } from '../lib/dom.js';
import { formatBytes, formatMs } from '../lib/format.js';
import { ForwardModel } from '../runtime/forward.js';
import { resolveModelSource } from '../runtime/sources.js';
import { BACKDROPS, EFFECTS } from '../vision/effects.js';
import { EffectsRenderer } from '../vision/effects-gl.js';
import { canvasToBlob, downloadBlob, fitWithin, loadImage, rasterize, sizeOf, toTensor } from '../vision/image.js';
import { licenseBadgeEl } from './badges.js';
import { browseMore, detailHeader } from './detail-shell.js';
import { gpuFallbackPanel } from './gpu-fallback.js';
import { appIcon } from './app-icons.js';
import { hfBadge } from './hf-badge.js';
import { icon } from './icons.js';
import { onDeviceNote } from './on-device.js';
import { confirmModelMemory } from './memory-guard.js';

const NORM = { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] };
const LOAD_STAGES = [
  { id: 'runtime', label: 'Start the ExecuTorch runtime' },
  { id: 'weights', label: 'Download the model' },
  { id: 'gpu', label: 'Prepare it on your GPU' },
];
const glyph = (paths) => () =>
  h('span', {
    class: 'fx-icon',
    html: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`,
  });
const cameraIcon = glyph('<path d="M3 8a2 2 0 0 1 2-2h2l2-2h6l2 2h2a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8Z"/><circle cx="12" cy="13" r="3.5"/>');
const downloadIcon = glyph('<path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 20h16"/>');
const MAX_TILE = 720;
const MAX_FOCUS = 1600;
const MAX_EXPORT = 1920;

export function mountEffects(root, demo) {
  const model = getModel(demo.models[0]);
  const state = {
    phase: 'checking', // checking | unavailable | idle | loading | ready | error
    source: null,
    reasons: [],
    load: { stage: null, fractions: {}, detail: '' },
    loadError: null,
    loadAbort: null,
    net: null,
    src: null, // { kind: 'camera' | 'video' | 'image', el, mirror, stream?, sampleId?, starting? }
    srcUploaded: null,
    maskFor: null,
    busy: false,
    focus: null,
    split: 0.5,
    backdrop: BACKDROPS[0].id,
    bgImage: null,
    stats: { ms: null, fps: null, last: 0, renderFps: null, lastRender: 0 },
    timing: {},
    error: null,
    glError: null,
    cameraPaused: false,
    facing: 'user',
  };
  let disposed = false;
  let raf = 0;
  let dirty = true;
  let renderer = null;
  const t0 = performance.now();

  const tiles = EFFECTS.map((fx) => {
    const canvas = h('canvas', { class: 'fx-tile__canvas', 'aria-hidden': 'true' });
    const el = h(
      'figure',
      { class: 'fx-tile' },
      h('button', { class: 'fx-tile__btn', type: 'button', 'aria-label': `Open ${fx.label}`, onclick: () => setFocus(fx.id) }, canvas),
      h('figcaption', { class: 'fx-tile__label' }, fx.label),
    );
    return { fx, canvas, el };
  });
  const grid = h('div', { class: 'fx-grid' }, tiles.map((t) => t.el));

  const focusCanvas = h('canvas', { class: 'fx-focus__canvas' });
  const splitEl = h('div', { class: 'fx-split' }, h('span', { class: 'fx-split__knob' }, icon('chevronLeft', 14), icon('chevronRight', 14)));
  const tagL = h('span', { class: 'fx-tag fx-tag--l' }, 'Original');
  const tagR = h('span', { class: 'fx-tag fx-tag--r' });
  const focusFrame = h('div', { class: 'fx-focus__frame', tabindex: '0', 'aria-label': 'Drag to compare with the original' }, focusCanvas, splitEl, tagL, tagR);
  const focusBar = h('div', { class: 'fx-focus__bar' });
  const focusView = h('div', { class: 'fx-focus', hidden: true }, focusBar, focusFrame);

  const overlay = h('div', { class: 'fx-overlay', hidden: true });
  const notice = h('div', { class: 'fx-notice', hidden: true });
  const videoHolder = h('div', { class: 'fx-video-holder', 'aria-hidden': 'true' });
  const stage = h('div', { class: 'fx-stage' }, grid, focusView, overlay, videoHolder);
  const statEl = h('span', { class: 'fx-stat', hidden: true });
  const fileInput = h('input', { type: 'file', accept: 'image/*,video/*', hidden: true, onchange: (e) => e.target.files[0] && openFile(e.target.files[0]) });
  const bgInput = h('input', { type: 'file', accept: 'image/*', hidden: true, onchange: (e) => e.target.files[0] && useBackground(e.target.files[0]) });
  const inputsEl = h('div', { class: 'fx-bar__inputs' });
  const backdropsEl = h('div', { class: 'fx-backdrops' });
  const bar = h('div', { class: 'fx-bar' }, inputsEl, statEl);

  replace(root, h('div', { class: 'page detail fx-page' }, detailHeader(demo), bar, notice, stage, backdropsEl, fileInput, bgInput, browseMore(demo.id)));

  // ---------- model ----------
  resolveModelSource(model).then((source) => {
    if (disposed) return;
    if (source.kind !== 'unavailable') Object.assign(state, { source, phase: 'idle' });
    else Object.assign(state, { phase: 'unavailable', reasons: source.reasons ?? [] });
    renderAll();
    if (ForwardModel.isLoaded(source)) load();
  });
  const unsubscribeGpu = gpuState.subscribe(() => state.phase === 'idle' && renderOverlay());

  function modelBytes() {
    return state.source?.spec?.pteBytes ?? model.approxBytes;
  }

  async function load() {
    if (gpuState.get().status !== 'ready' || state.phase === 'loading') return;
    if (!(await confirmModelMemory(model))) return;
    const controller = new AbortController();
    Object.assign(state, { phase: 'loading', loadError: null, loadAbort: controller, load: { stage: 'runtime', fractions: {}, detail: '' } });
    renderAll();
    try {
      renderer ??= new EffectsRenderer();
    } catch (err) {
      Object.assign(state, { phase: 'idle', glError: err.message });
      renderAll();
      return;
    }
    try {
      const net = await ForwardModel.load(state.source, {
        signal: controller.signal,
        onProgress: (p) => {
          state.load.stage = p.stage;
          state.load.fractions[p.stage] = p.fraction;
          if (p.stage === 'weights') state.load.detail = p.fromCache ? 'Loaded from this device’s cache' : `${formatBytes(p.loaded)} / ${formatBytes(p.total || modelBytes())}`;
          if (p.stage === 'gpu') state.load.detail = 'Building GPU pipelines…';
          renderLoader();
        },
      });
      if (disposed || controller.signal.aborted) return net.dispose();
      Object.assign(state, { net, phase: 'ready' });
      renderAll();
      startCamera();
    } catch (err) {
      if (err.name === 'AbortError' || disposed) return;
      Object.assign(state, { phase: 'error', loadError: err.message });
      renderAll();
    } finally {
      state.loadAbort = null;
    }
  }

  // ---------- sources ----------
  function samples() {
    const shared = state.source?.shared?.visionSamples ?? {};
    return (shared.effects ?? shared.bgremove ?? []).filter((s) => !s.video);
  }

  function setSource(src) {
    stopSource();
    Object.assign(state, { src, srcUploaded: null, maskFor: null, error: null, cameraPaused: false });
    state.stats = { ...state.stats, ms: null, fps: null, last: 0 };
    renderer?.clearMask();
    dirty = true;
    renderAll();
  }

  function stopSource() {
    const s = state.src;
    if (!s) return;
    s.stream?.getTracks().forEach((t) => t.stop());
    if (s.kind !== 'image' && s.el) {
      s.el.pause();
      if (s.ownsUrl) URL.revokeObjectURL(s.el.src);
      s.el.removeAttribute('src');
      s.el.srcObject = null;
      s.el.remove();
    }
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) return cameraFailed('This browser does not give web pages access to a camera.');
    const facing = state.facing;
    setSource({ kind: 'camera', el: null, starting: true, mirror: facing === 'user' });
    const src = state.src;
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    } catch (err) {
      if (state.src === src) cameraFailed(err.name === 'NotAllowedError' ? 'Camera access was blocked. Allow it from the address bar, or try a photo.' : `The camera could not start: ${err.message}`);
      return;
    }
    if (state.src !== src || disposed) return stream.getTracks().forEach((t) => t.stop());
    const video = h('video', { class: 'fx-video', muted: true, playsinline: true, autoplay: true });
    video.muted = true;
    video.srcObject = stream;
    videoHolder.append(video);
    const facingMode = stream.getVideoTracks()[0]?.getSettings?.().facingMode;
    Object.assign(src, { el: video, stream, starting: false, mirror: (facingMode ?? facing) !== 'environment' });
    video.play().catch(() => {});
    renderAll();
  }

  function cameraFailed(message) {
    const first = samples()[0];
    if (first) usePhoto(state.source.url(first.path), first.path).then(() => showNotice(message));
    else {
      setSource(null);
      showNotice(message);
    }
  }

  async function usePhoto(url, sampleId = null) {
    try {
      const img = await loadImage(url);
      if (disposed) return;
      setSource({ kind: 'image', el: rasterize(img, MAX_EXPORT), mirror: false, sampleId });
    } catch (err) {
      showNotice(err.message);
    }
  }

  function openFile(file) {
    fileInput.value = '';
    const url = URL.createObjectURL(file);
    if (file.type.startsWith('video/')) {
      const video = h('video', { class: 'fx-video', muted: true, playsinline: true, loop: true });
      video.muted = true;
      video.addEventListener('error', () => state.src?.el === video && (setSource(null), showNotice('This browser cannot play that video. Try an MP4 or WebM file.')));
      video.src = url;
      videoHolder.append(video);
      setSource({ kind: 'video', el: video, mirror: false, ownsUrl: true });
      video.play().catch(() => {});
      return;
    }
    if (!file.type.startsWith('image/')) return showNotice('That file is not an image or a video.');
    usePhoto(url).finally(() => URL.revokeObjectURL(url));
  }

  async function useBackground(file) {
    bgInput.value = '';
    const url = URL.createObjectURL(file);
    try {
      const c = rasterize(await loadImage(url), 1920);
      state.bgImage = c;
      state.backdrop = 'image';
      renderer?.setBackground(c, c.width, c.height);
      dirty = true;
    } catch (err) {
      showNotice(err.message);
    } finally {
      URL.revokeObjectURL(url);
    }
    renderBackdrops();
  }

  function pickBackdrop(id) {
    state.backdrop = id;
    renderer?.setBackground(id === 'image' ? state.bgImage : null, state.bgImage?.width, state.bgImage?.height);
    dirty = true;
    renderBackdrops();
  }

  function onVisibility() {
    if (document.hidden && state.src?.kind === 'camera') {
      stopSource();
      Object.assign(state, { src: null, cameraPaused: true });
      renderAll();
    } else if (document.hidden) state.src?.el?.pause?.();
    else if (state.src?.kind === 'video') state.src.el.play().catch(() => {});
  }
  document.addEventListener('visibilitychange', onVisibility);

  function onKey(e) {
    if (e.key === 'Escape' && state.focus) setFocus(null);
  }
  document.addEventListener('keydown', onKey);

  // ---------- frame loop ----------
  function sourceReady() {
    const el = state.src?.el;
    if (!el) return false;
    return state.src.kind === 'image' || (el.readyState >= 2 && el.videoWidth > 0);
  }

  function infer(src) {
    state.busy = true;
    const start = performance.now();
    const { tensor } = toTensor(src.el, 256, 256, NORM);
    const prepared = performance.now();
    state.net
      .run([tensor])
      .then(([out]) => {
        if (state.src !== src || disposed) return;
        const ran = performance.now();
        const [mh, mw] = out.shape.slice(-2);
        const alpha = new Uint8Array(mw * mh);
        for (let i = 0; i < alpha.length; i++) alpha[i] = Math.max(0, Math.min(255, out.data[i] * 255 + 0.5));
        renderer.setMask(alpha, mw, mh);
        const t = state.net.lastTiming ?? {};
        const blend = (k, v) => (state.timing[k] = state.timing[k] == null ? v : state.timing[k] * 0.8 + v * 0.2);
        blend('prepare', prepared - start);
        blend('upload', t.upload ?? 0);
        blend('forward', t.forward ?? ran - prepared);
        blend('readback', t.readback ?? 0);
        blend('mask', performance.now() - ran);
        state.maskFor = src;
        const now = performance.now();
        const s = state.stats;
        const ms = now - start;
        const dt = s.last ? now - s.last : null;
        Object.assign(s, { ms: s.ms == null ? ms : s.ms * 0.8 + ms * 0.2, fps: dt ? (s.fps == null ? 1000 / dt : s.fps * 0.8 + 200 / dt) : s.fps, last: now });
        dirty = true;
        renderStat();
      })
      .catch((err) => state.src === src && showNotice(`The model failed on a frame: ${err.message}`))
      .finally(() => {
        state.busy = false;
      });
  }

  function tileSize() {
    const [w, h0] = sizeOf(state.src.el);
    const width = Math.min(MAX_TILE, Math.round((tiles[0].canvas.clientWidth || 360) * devicePixelRatio));
    return [width, Math.round((width * h0) / w)];
  }

  function tick(now) {
    raf = requestAnimationFrame(tick);
    const src = state.src;
    if (!renderer || state.phase !== 'ready' || !sourceReady()) return;
    const live = src.kind !== 'image';
    if (live || state.srcUploaded !== src) {
      renderer.setSource(src.el, ...sizeOf(src.el));
      if (state.srcUploaded !== src) {
        state.srcUploaded = src;
        renderOverlay();
      }
    }
    if (!state.busy && !document.hidden && (live || state.maskFor !== src)) infer(src);
    const time = (now - t0) / 1000;
    const gradient = (BACKDROPS.find((b) => b.id === state.backdrop) ?? BACKDROPS[0]).rgb;
    const common = { mirror: src.mirror, time, gradient };
    if (state.focus) {
      const [w, h0] = sizeOf(src.el);
      const [fw, fh] = fitWithin(w, h0, Math.min(MAX_FOCUS, Math.round(Math.max(320, stage.clientWidth - 52) * devicePixelRatio)));
      renderer.render(state.focus, focusCanvas, { ...common, width: fw, height: fh, split: state.split });
    } else {
      const [w, h0] = tileSize();
      // A still photo only needs repainting when something changed, except for the animated effect.
      for (const t of tiles) if (live || dirty || t.fx.id === 'neon') renderer.render(t.fx.id, t.canvas, { ...common, width: w, height: h0 });
    }
    dirty = false;
    const s = state.stats;
    if (s.lastRender) s.renderFps = s.renderFps == null ? 1000 / (now - s.lastRender) : s.renderFps * 0.95 + 50 / (now - s.lastRender);
    s.lastRender = now;
  }
  raf = requestAnimationFrame(tick);
  const statTimer = setInterval(renderStat, 500);

  // ---------- focus mode ----------
  function setFocus(id) {
    state.focus = id;
    state.split = 0.5;
    dirty = true;
    renderFocus();
    renderBackdrops();
    if (id) focusFrame.focus({ preventScroll: true });
  }

  let dragging = false;
  const moveSplit = (e) => {
    const r = focusCanvas.getBoundingClientRect();
    state.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    splitEl.style.left = `${state.split * 100}%`;
  };
  focusFrame.addEventListener('pointerdown', (e) => {
    dragging = true;
    focusFrame.setPointerCapture(e.pointerId);
    moveSplit(e);
  });
  focusFrame.addEventListener('pointermove', (e) => dragging && moveSplit(e));
  focusFrame.addEventListener('pointerup', () => (dragging = false));
  focusFrame.addEventListener('pointercancel', () => (dragging = false));
  focusFrame.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    state.split = Math.min(1, Math.max(0, state.split + (e.key === 'ArrowLeft' ? -0.05 : 0.05)));
    splitEl.style.left = `${state.split * 100}%`;
  });

  async function downloadFrame() {
    if (!renderer || !sourceReady()) return;
    const [w, h0] = fitWithin(...sizeOf(state.src.el), MAX_EXPORT);
    const out = document.createElement('canvas');
    const gradient = (BACKDROPS.find((b) => b.id === state.backdrop) ?? BACKDROPS[0]).rgb;
    renderer.render(state.focus, out, { width: w, height: h0, split: 0, mirror: state.src.mirror, time: (performance.now() - t0) / 1000, gradient });
    dirty = true;
    downloadBlob(await canvasToBlob(out), `${state.focus}.png`);
  }

  // ---------- rendering ----------
  function renderAll() {
    if (disposed) return;
    renderInputs();
    renderOverlay();
    renderFocus();
    renderBackdrops();
    renderStat();
  }

  function showNotice(message) {
    state.error = message;
    notice.hidden = !message;
    if (message) replace(notice, h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, message), h('button', { class: 'fx-notice__close', type: 'button', 'aria-label': 'Dismiss', onclick: () => showNotice(null) }, icon('close', 14))));
  }

  function renderInputs() {
    const ready = state.phase === 'ready';
    bar.hidden = !ready;
    if (!ready) return;
    const camOn = state.src?.kind === 'camera';
    replace(
      inputsEl,
      h('button', { class: camOn ? 'btn btn--primary btn--sm' : 'btn btn--ghost btn--sm', type: 'button', 'aria-pressed': String(camOn), onclick: () => (camOn ? setSource(null) : startCamera()) }, camOn ? icon('stop', 14) : cameraIcon(), camOn ? 'Stop camera' : 'Camera'),
      camOn ? h('button', { class: 'btn btn--ghost btn--sm', type: 'button', title: 'Switch between front and back cameras', onclick: () => ((state.facing = state.facing === 'user' ? 'environment' : 'user'), startCamera()) }, icon('refresh', 14), 'Flip') : null,
      samples().map((s) =>
        h(
          'button',
          { class: state.src?.sampleId === s.path ? 'fx-thumb is-active' : 'fx-thumb', type: 'button', title: s.label, 'aria-label': `Use photo: ${s.label}`, onclick: () => usePhoto(state.source.url(s.path), s.path) },
          h('img', { class: 'fx-thumb__img', src: state.source.url(s.path), alt: '', loading: 'lazy' }),
        ),
      ),
      h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => fileInput.click() }, icon('plus', 14), 'Photo or video'),
    );
  }

  function renderStat() {
    const { ms, fps, renderFps } = state.stats;
    statEl.hidden = ms == null || state.phase !== 'ready' || !state.src;
    if (statEl.hidden) return;
    const live = state.src.kind !== 'image';
    statEl.textContent = [
      `model ${formatMs(ms)}`,
      live && fps ? `${fps < 1 ? '<1' : Math.round(fps)} masks/s` : null,
      renderFps ? `display ${Math.round(Math.min(renderFps, 240))} fps` : null,
    ]
      .filter(Boolean)
      .join(' · ');
    const t = state.timing;
    statEl.title = [
      'Model: how long one ExecuTorch pass takes; masks/s: how often the person mask updates.',
      'Display: how often the effects redraw (they reuse the latest mask).',
      '',
      'One pass, averaged:',
      `  prepare frame  ${formatMs(t.prepare)}`,
      `  copy to runtime  ${formatMs(t.upload)}`,
      `  ExecuTorch forward (GPU)  ${formatMs(t.forward)}`,
      `  read back  ${formatMs(t.readback)}`,
      `  build mask  ${formatMs(t.mask)}`,
    ].join('\n');
  }

  function renderFocus() {
    const fx = EFFECTS.find((e) => e.id === state.focus);
    focusView.hidden = !fx;
    grid.hidden = Boolean(fx);
    if (!fx) return;
    tagR.textContent = fx.label;
    splitEl.style.left = `${state.split * 100}%`;
    replace(
      focusBar,
      h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => setFocus(null) }, icon('grid', 14), 'All effects'),
      h(
        'div',
        { class: 'segmented segmented--sm fx-focus__pick', role: 'group', 'aria-label': 'Effect' },
        EFFECTS.filter((e) => e.id !== 'original').map((e) =>
          h('button', { class: e.id === fx.id ? 'segmented__btn is-active' : 'segmented__btn', type: 'button', 'aria-pressed': String(e.id === fx.id), onclick: () => setFocus(e.id) }, e.label),
        ),
      ),
      h('button', { class: 'btn btn--primary btn--sm', type: 'button', onclick: downloadFrame }, downloadIcon(), 'Download frame'),
    );
  }

  function renderBackdrops() {
    const show = state.phase === 'ready' && (!state.focus || state.focus === 'virtual');
    backdropsEl.hidden = !show;
    if (!show) return;
    replace(
      backdropsEl,
      h('span', { class: 'fx-backdrops__label' }, 'Virtual background'),
      BACKDROPS.map((b) =>
        h('button', {
          class: state.backdrop === b.id ? 'fx-swatch is-active' : 'fx-swatch',
          type: 'button',
          title: b.label,
          'aria-label': b.label,
          'aria-pressed': String(state.backdrop === b.id),
          style: { background: `linear-gradient(135deg, ${b.stops.join(', ')})` },
          onclick: () => pickBackdrop(b.id),
        }),
      ),
      h(
        'button',
        {
          class: state.backdrop === 'image' ? 'fx-swatch fx-swatch--image is-active' : 'fx-swatch fx-swatch--image',
          type: 'button',
          title: 'Use your own image',
          'aria-label': 'Use your own image',
          style: state.bgImage ? { backgroundImage: `url(${state.bgImage.toDataURL('image/jpeg', 0.6)})` } : null,
          onclick: () => (state.bgImage && state.backdrop !== 'image' ? pickBackdrop('image') : bgInput.click()),
        },
        state.bgImage ? null : icon('plus', 14),
      ),
    );
  }

  function renderOverlay() {
    let content = null;
    if (state.phase === 'unavailable') content = unavailableCard();
    else if (state.phase === 'idle' || state.phase === 'error') content = preloadCard();
    else if (state.phase === 'loading') content = loaderCard();
    else if (state.phase === 'ready' && state.cameraPaused && !state.src) content = smallCard('The camera turned off while this tab was in the background.', h('button', { class: 'btn btn--primary', type: 'button', onclick: startCamera }, cameraIcon(), 'Turn it back on'));
    else if (state.phase === 'ready' && state.src?.starting) content = smallCard('Waiting for your camera…', null, true);
    else if (state.phase === 'ready' && !state.src) content = smallCard('Choose a source to see the effects.', h('div', { class: 'fx-card__btns' }, h('button', { class: 'btn btn--primary', type: 'button', onclick: startCamera }, cameraIcon(), 'Use camera'), h('button', { class: 'btn btn--ghost', type: 'button', onclick: () => fileInput.click() }, icon('plus', 15), 'Photo or video')));
    else if (state.phase === 'ready' && state.srcUploaded !== state.src) content = smallCard('Opening…', null, true);
    overlay.hidden = !content;
    stage.classList.toggle('fx-stage--covered', Boolean(content));
    replace(overlay, content);
  }

  function smallCard(text, action, spinner = false) {
    return h('div', { class: 'fx-card fx-card--small' }, spinner ? h('span', { class: 'fx-spinner' }) : null, h('p', { class: 'preload__sub' }, text), action);
  }

  function unavailableCard() {
    return h(
      'div',
      { class: 'fx-card' },
      h('div', { class: 'preload__glyph', 'aria-hidden': 'true' }, icon('lock', 24)),
      h('h2', { class: 'preload__title' }, demo.title),
      h('p', { class: 'preload__sub' }, demo.blurb),
      h('p', { class: 'preload__note preload__note--strong' }, `Not available in this build: ${model.name} can’t be downloaded from this site yet.`),
      h('div', { class: 'preload__facts' }, h('p', { class: 'preload__note' }, `Checkpoint ${formatBytes(model.approxBytes)} `, licenseBadgeEl(model)), state.reasons.map((r) => h('p', { class: 'preload__note' }, r.text ?? String(r)))),
    );
  }

  function preloadCard() {
    const gpu = gpuState.get();
    const size = formatBytes(modelBytes());
    return h(
      'div',
      { class: 'fx-card' },
      h('div', { class: 'app-glyph', 'aria-hidden': 'true' }, appIcon(demo.id, 64)),
      h('h2', { class: 'preload__title' }, demo.title),
      h('p', { class: 'preload__sub' }, `Six live effects from one ${model.shortName} person mask per frame, drawn on your GPU.`),
      state.glError ? h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, state.glError)) : null,
      state.phase === 'error' ? h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, `Load failed: ${state.loadError}`)) : null,
      gpu.status === 'ready'
        ? h('button', { class: 'btn btn--primary btn--xl', type: 'button', disabled: Boolean(state.glError), onclick: load }, icon('bolt', 18), `Load model and start camera · ${size}`)
        : gpu.status === 'probing'
          ? h('button', { class: 'btn btn--primary btn--xl', type: 'button', disabled: true }, 'Detecting GPU…')
          : gpuFallbackPanel(gpu, { compact: true }),
      h('p', { class: 'preload__note' }, 'Downloads once, then stays cached on this device. Camera frames never leave the tab.'), hfBadge(state.source),
      onDeviceNote(model),
    );
  }

  let loaderEls = null;
  function loaderCard() {
    const bar0 = h('div', { class: 'progress__bar' });
    const detail = h('p', { class: 'loader__detail' });
    const steps = LOAD_STAGES.map((s) => h('li', { class: 'steps__item' }, h('span', { class: 'steps__dot' }), s.label));
    loaderEls = { bar: bar0, detail, steps };
    queueMicrotask(renderLoader);
    return h(
      'div',
      { class: 'fx-card loader' },
      h('h2', { class: 'loader__title' }, `Loading ${model.shortName}`),
      h('div', { class: 'progress', role: 'progressbar', 'aria-label': 'Load progress', 'aria-valuemin': '0', 'aria-valuemax': '100' }, bar0),
      detail,
      h('ol', { class: 'steps' }, steps),
      hfBadge(state.source),
      h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => (state.loadAbort?.abort(), (state.phase = 'idle'), renderAll()) }, 'Cancel'),
    );
  }

  function renderLoader() {
    if (!loaderEls || state.phase !== 'loading') return;
    const { fractions, stage: current, detail } = state.load;
    const weights = { runtime: 0.06, weights: 0.84, gpu: 0.1 };
    let total = 0;
    LOAD_STAGES.forEach((s, i) => {
      const f = Math.min(1, fractions[s.id] ?? 0);
      total += weights[s.id] * f;
      loaderEls.steps[i].classList.toggle('is-done', f >= 1);
      loaderEls.steps[i].classList.toggle('is-active', s.id === current && f < 1);
    });
    loaderEls.bar.style.width = `${(total * 100).toFixed(1)}%`;
    loaderEls.bar.parentElement.setAttribute('aria-valuenow', String(Math.round(total * 100)));
    loaderEls.detail.textContent = detail;
  }

  return () => {
    disposed = true;
    cancelAnimationFrame(raf);
    clearInterval(statTimer);
    unsubscribeGpu();
    stopSource();
    state.loadAbort?.abort();
    renderer?.dispose();
    document.removeEventListener('visibilitychange', onVisibility);
    document.removeEventListener('keydown', onKey);
  };
}
