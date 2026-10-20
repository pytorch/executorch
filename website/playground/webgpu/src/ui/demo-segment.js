import { getModel } from '../config/models.js';
import { gpuState } from '../gpu/context.js';
import { h, replace, svg } from '../lib/dom.js';
import { formatBytes, formatMs } from '../lib/format.js';
import { LABEL, maskContours, SegmentAnything } from '../runtime/sam.js';
import { resolveModelSource } from '../runtime/sources.js';
import { licenseBadgeEl } from './badges.js';
import { browseMore, detailHeader } from './detail-shell.js';
import { gpuFallbackPanel } from './gpu-fallback.js';
import { appIcon } from './app-icons.js';
import { hfBadge } from './hf-badge.js';
import { icon } from './icons.js';
import { onDeviceNote } from './on-device.js';

const COLORS = ['#3b82f6', '#f97316', '#10b981', '#e11d48', '#8b5cf6', '#eab308', '#06b6d4', '#ec4899'];
const MAX_WORKING_SIDE = 2048;
const DRAG_THRESHOLD = 6;
const HOVER_DELAY = 45;
// Hover previews only when one decode is comfortably interactive on this GPU.
const HOVER_BUDGET_MS = 160;
const ANT_FRAME_MS = 50;
const ANT_LINGER_MS = 4000;
const DETAIL = [
  { id: 'auto', label: 'Auto', title: 'Pick the most confident mask' },
  { id: 0, label: 'Fine', title: 'Smallest of the decoder’s candidate masks' },
  { id: 1, label: 'Mid', title: 'Middle candidate mask' },
  { id: 2, label: 'Whole', title: 'Largest candidate mask' },
];
const LOAD_STAGES = [
  { id: 'runtime', label: 'Start the ExecuTorch runtime' },
  { id: 'weights', label: 'Download the encoder and decoder' },
  { id: 'prompts', label: 'Load the prompt encoder tables' },
  { id: 'gpu', label: 'Prepare both programs for your GPU' },
];

// The loaded model outlives the view, so leaving the page and coming back does
// not download or rebuild anything.
let kept = null; // { key, engine }

export function mountSegment(root, demo) {
  const model = getModel(demo.models[0]);
  const state = {
    phase: 'checking', // checking | unavailable | idle | loading | ready | error
    reasons: [],
    load: { stage: null, done: new Set(), loaded: 0, total: 0, fromCache: false },
    loadError: null,
    samples: [],
    image: null, // { canvas, width, height, key }
    encoding: false,
    encodeStarted: 0,
    encodeMs: null,
    encodeError: null,
    decodeMs: null,
    objects: [newObject(0)],
    active: 0,
    detail: 'auto',
    mode: LABEL.POSITIVE,
    history: [],
    hover: null,
    preview: null,
    drag: null,
    camera: null,
    dropping: false,
    notice: null,
  };
  let source = null;
  let sam = null;
  let loadAbort = null;
  let disposed = false;
  let raf = 0;
  let hoverTimer = 0;
  let encodeTimer = 0;
  let previewBusy = false;
  let objectSeq = 1;
  let imageSeq = 0;

  // ---------- static structure ----------
  const imageCanvas = h('canvas', { class: 'seg-canvas', 'aria-hidden': 'true' });
  const overlay = h('canvas', {
    class: 'seg-overlay',
    role: 'img',
    'aria-label': 'Image. Click an object to segment it; Shift-click to exclude an area; drag to draw a box.',
  });
  const panel = h('div', { class: 'seg-panel' });
  const hint = h('div', { class: 'seg-chip seg-chip--hint' });
  const stat = h('div', { class: 'seg-chip seg-chip--stat mono' });
  const status = h('div', { class: 'seg-chip seg-chip--status', 'aria-live': 'polite' });
  const sweep = h('div', { class: 'seg-sweep' });
  const stage = h('div', { class: 'seg-stage' }, imageCanvas, overlay, sweep, hint, status, stat, panel);
  const stageWrap = h('div', { class: 'seg-stage-wrap' }, stage);
  const toolbar = h('div', { class: 'seg-toolbar' });
  const notice = h('div', { class: 'seg-notice' });
  const tray = h('div', { class: 'seg-tray' });
  const side = h('aside', { class: 'metrics seg-side', 'aria-label': 'Objects and export' });
  const gpuSlot = h('div', null);
  const fileInput = h('input', { type: 'file', accept: 'image/*', class: 'seg-file', onchange: (e) => e.target.files[0] && loadFile(e.target.files[0]) });

  replace(root,
    h(
      'div',
      { class: 'page detail segment-page' },
      detailHeader(demo),
      gpuSlot,
      h('div', { class: 'detail-grid' }, h('div', { class: 'seg-main' }, toolbar, notice, stageWrap, tray, fileInput), side),
      browseMore(demo.id),
    ),
  );

  // ---------- model ----------
  async function resolve() {
    source = await resolveModelSource(model);
    if (disposed) return;
    if (source.kind === 'unavailable') {
      state.phase = 'unavailable';
      state.reasons = source.reasons ?? [];
      renderAll();
      return;
    }
    if (kept?.key === source.url(source.spec.encoder)) sam = kept.engine;
    state.phase = sam ? 'ready' : 'idle';
    state.samples = (source.shared.segmentSamples ?? []).map((s) => ({ url: source.url(s.path), label: s.label }));
    renderAll();
    if (state.samples.length && !state.image) loadUrl(state.samples[0].url, state.samples[0].label);
  }

  async function load() {
    if (state.phase === 'loading' || gpuState.get().status !== 'ready') return;
    const controller = new AbortController();
    loadAbort = controller;
    Object.assign(state, { phase: 'loading', loadError: null, load: { stage: 'runtime', done: new Set(), loaded: 0, total: 0, fromCache: false } });
    renderAll();
    try {
      const engine = await SegmentAnything.load(source, {
        signal: controller.signal,
        onProgress: (p) => {
          const L = state.load;
          if (p.stage !== L.stage && p.stage !== 'prompts') {
            LOAD_STAGES.forEach((s) => {
              if (LOAD_STAGES.findIndex((x) => x.id === s.id) < LOAD_STAGES.findIndex((x) => x.id === p.stage)) L.done.add(s.id);
            });
            L.stage = p.stage;
          }
          if (p.stage === 'prompts') L.done.add('prompts');
          if (p.stage === 'weights') Object.assign(L, { loaded: p.loaded, total: p.total, fromCache: p.fromCache });
          if (p.stage === 'gpu' && p.fraction === 1) L.done.add('gpu');
          schedulePanel();
        },
      });
      if (disposed || controller.signal.aborted) {
        engine.dispose();
        return;
      }
      sam = engine;
      kept = { key: source.url(source.spec.encoder), engine };
      state.phase = 'ready';
      renderAll();
      if (state.image) encode();
    } catch (err) {
      if (disposed || controller.signal.aborted) return;
      Object.assign(state, { phase: 'error', loadError: err.message });
      renderAll();
    } finally {
      if (loadAbort === controller) loadAbort = null;
    }
  }

  // Byte progress arrives per network chunk; repaint the loader at most once a frame.
  let panelFrame = 0;
  function schedulePanel() {
    panelFrame ||= requestAnimationFrame(() => {
      panelFrame = 0;
      if (!disposed && state.phase === 'loading') renderPanel();
    });
  }

  function cancelLoad() {
    loadAbort?.abort();
    state.phase = 'idle';
    renderAll();
  }

  // ---------- images ----------
  function loadUrl(url, label) {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => setImage(img, img.naturalWidth, img.naturalHeight, url);
    img.onerror = () => showNotice(`Could not open ${label ?? 'that image'}.`);
    img.src = url;
  }

  function loadFile(file) {
    if (!file.type.startsWith('image/')) {
      showNotice('That file is not an image.');
      return;
    }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setImage(img, img.naturalWidth, img.naturalHeight, `file:${file.name}:${file.size}`);
      URL.revokeObjectURL(url);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      showNotice('Your browser could not decode that image.');
    };
    img.src = url;
    fileInput.value = '';
  }

  // Everything downstream works on one bounded canvas copy, so huge photos and
  // camera frames cost the same and exports come out at a sane size.
  function setImage(src, w, hgt, key) {
    const scale = Math.min(1, MAX_WORKING_SIDE / Math.max(w, hgt));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(w * scale));
    canvas.height = Math.max(1, Math.round(hgt * scale));
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(src, 0, 0, canvas.width, canvas.height);
    imageSeq++;
    Object.assign(state, {
      image: { canvas, width: canvas.width, height: canvas.height, key },
      objects: [newObject(0)],
      active: 0,
      history: [],
      preview: null,
      hover: null,
      encodeMs: null,
      encodeError: null,
    });
    objectSeq = 1;
    stopCamera();
    layout();
    renderAll();
    if (sam) encode();
  }

  async function encode() {
    const seq = imageSeq;
    const { canvas, width, height } = state.image;
    Object.assign(state, { encoding: true, encodeStarted: performance.now(), encodeError: null });
    clearInterval(encodeTimer);
    encodeTimer = setInterval(renderStatus, 100);
    renderAll();
    try {
      const { ms } = await sam.setImage(canvas, { width, height });
      if (seq !== imageSeq || disposed) return;
      state.encodeMs = ms;
      // The first decode compiles the decoder's shaders; pay for it now, not on the
      // first click. A second one measures the steady-state latency that decides
      // whether hover previews are affordable.
      const warm = (x, y) => sam.segment({ points: [{ x: width * x, y: height * y, label: LABEL.POSITIVE }] });
      warm(0.5, 0.5)
        .then(() => warm(0.25, 0.25))
        .then(() => {
          if (seq === imageSeq && !disposed) {
            state.decodeMs ??= sam.decodeMs;
            renderStatus();
          }
        })
        .catch(() => {});
    } catch (err) {
      if (seq === imageSeq) state.encodeError = err.message;
    } finally {
      if (seq === imageSeq) {
        state.encoding = false;
        clearInterval(encodeTimer);
        if (!disposed) renderAll();
      }
    }
    // Clicks made while the encoder ran were queued behind it; decode them now.
    if (seq === imageSeq && !state.encodeError) state.objects.forEach((o) => hasPrompt(o) && !o.result && segmentObject(o));
  }

  // ---------- camera ----------
  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      showNotice('This browser does not offer camera access here.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 } }, audio: false });
      if (disposed) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      const video = h('video', { class: 'seg-video', playsinline: true, muted: true, autoplay: true });
      video.srcObject = stream;
      state.camera = { stream, video };
      renderPanel();
      renderStatus();
    } catch (err) {
      showNotice(err.name === 'NotAllowedError' ? 'Camera permission was declined.' : `Camera unavailable: ${err.message}`);
    }
  }

  function snapshot() {
    const { video } = state.camera ?? {};
    if (!video?.videoWidth) return;
    setImage(video, video.videoWidth, video.videoHeight, `camera:${Date.now()}`);
  }

  function stopCamera() {
    state.camera?.stream.getTracks().forEach((t) => t.stop());
    state.camera = null;
  }

  // ---------- prompts ----------
  function newObject(i) {
    return { id: i, color: COLORS[i % COLORS.length], points: [], box: null, result: null, shape: null, version: 0, pending: false, error: null };
  }

  function hasPrompt(o) {
    return o.points.some((p) => p.label === LABEL.POSITIVE) || o.box;
  }

  function activeObject() {
    return state.objects[state.active];
  }

  function remember() {
    state.history.push({
      active: state.active,
      objects: state.objects.map((o) => ({ ...o, points: [...o.points] })),
    });
    if (state.history.length > 60) state.history.shift();
  }

  function undo() {
    const prev = state.history.pop();
    if (!prev) return;
    state.objects = prev.objects.map((o) => ({ ...o, version: o.version + 1000, pending: false }));
    state.active = prev.active;
    state.preview = null;
    renderAll();
  }

  function clearAll() {
    if (!state.objects.some((o) => o.points.length || o.box)) return;
    remember();
    state.objects = [newObject(0)];
    state.active = 0;
    objectSeq = 1;
    state.preview = null;
    renderAll();
  }

  function addObject() {
    const cur = activeObject();
    if (cur && !cur.points.length && !cur.box) return;
    remember();
    state.objects.push(newObject(objectSeq++));
    state.active = state.objects.length - 1;
    state.preview = null;
    renderAll();
  }

  function removeObject(i) {
    remember();
    state.objects.splice(i, 1);
    if (!state.objects.length) {
      state.objects.push(newObject(0));
      objectSeq = 1;
    }
    state.active = Math.min(state.active, state.objects.length - 1);
    renderAll();
  }

  function selectObject(i) {
    state.active = i;
    state.preview = null;
    renderAll();
  }

  function addPoint(x, y, label) {
    const o = activeObject();
    remember();
    // Clicking an existing point of this object takes it back out.
    const r = hitRadius();
    const near = o.points.findIndex((p) => Math.hypot(p.x - x, p.y - y) < r);
    if (near >= 0) o.points.splice(near, 1);
    else o.points.push({ x, y, label });
    state.preview = null;
    segmentObject(o);
  }

  function setBox(box) {
    const o = activeObject();
    remember();
    o.box = box;
    state.preview = null;
    segmentObject(o);
  }

  async function segmentObject(o) {
    const version = ++o.version;
    o.error = null;
    if (!hasPrompt(o)) {
      Object.assign(o, { result: null, shape: null, pending: false });
      renderAll();
      return;
    }
    if (!sam || state.encoding || !state.image || state.encodeError) {
      o.pending = Boolean(sam && state.encoding);
      renderAll();
      return;
    }
    o.pending = true;
    renderAll();
    try {
      const seq = imageSeq;
      const result = await sam.segment({ points: o.points, box: o.box, detail: state.detail });
      if (disposed || seq !== imageSeq || version !== o.version) return;
      state.decodeMs = sam.decodeMs ?? state.decodeMs;
      Object.assign(o, { result, shape: buildShape(result), pending: false });
    } catch (err) {
      if (version !== o.version) return;
      Object.assign(o, { pending: false, error: err.message });
    }
    if (!disposed) renderAll();
  }

  function setDetail(d) {
    state.detail = d;
    state.objects.forEach((o) => hasPrompt(o) && segmentObject(o));
    renderToolbar();
  }

  function setMode(m) {
    state.mode = m;
    renderToolbar();
    renderStatus();
  }

  // ---------- hover preview ----------
  function schedulePreview() {
    clearTimeout(hoverTimer);
    if (!canPreview()) return;
    hoverTimer = setTimeout(runPreview, HOVER_DELAY);
  }

  function canPreview() {
    return sam && state.image && !state.encoding && !state.encodeError && state.decodeMs != null && state.decodeMs < HOVER_BUDGET_MS && state.hover && !state.drag;
  }

  async function runPreview() {
    if (previewBusy || !canPreview()) return;
    previewBusy = true;
    const at = state.hover;
    const o = activeObject();
    const seq = imageSeq;
    try {
      const result = await sam.segment({ points: [...o.points, { x: at.x, y: at.y, label: state.mode }], box: o.box, detail: state.detail });
      if (!disposed && seq === imageSeq && state.hover === at && o === activeObject()) state.preview = { at, color: o.color, shape: buildShape(result) };
    } catch {
      state.preview = null;
    } finally {
      previewBusy = false;
    }
    if (state.hover && state.hover !== at) schedulePreview();
    requestDraw();
  }

  // ---------- pointer input ----------
  function toImage(e) {
    const r = overlay.getBoundingClientRect();
    const { width, height } = state.image;
    return {
      x: Math.max(0, Math.min(width - 1, ((e.clientX - r.left) / r.width) * width)),
      y: Math.max(0, Math.min(height - 1, ((e.clientY - r.top) / r.height) * height)),
    };
  }

  function hitRadius() {
    const r = overlay.getBoundingClientRect();
    return (12 * state.image.width) / Math.max(1, r.width);
  }

  function interactive() {
    return state.phase === 'ready' && state.image && !state.camera;
  }

  overlay.addEventListener('pointerdown', (e) => {
    if (!interactive() || (e.button !== 0 && e.button !== 2)) return;
    overlay.setPointerCapture(e.pointerId);
    const p = toImage(e);
    const negative = e.button === 2 || e.shiftKey || e.altKey || state.mode === LABEL.NEGATIVE;
    state.drag = { start: p, now: p, sx: e.clientX, sy: e.clientY, box: false, negative };
  });
  overlay.addEventListener('pointermove', (e) => {
    if (!interactive()) return;
    const p = toImage(e);
    if (state.drag) {
      state.drag.now = p;
      if (!state.drag.box && Math.hypot(e.clientX - state.drag.sx, e.clientY - state.drag.sy) > DRAG_THRESHOLD) state.drag.box = true;
      state.preview = null;
      requestDraw();
      return;
    }
    if (e.pointerType !== 'mouse') return;
    state.hover = p;
    if (state.preview && Math.hypot(state.preview.at.x - p.x, state.preview.at.y - p.y) > hitRadius() * 3) state.preview = null;
    schedulePreview();
    requestDraw();
  });
  overlay.addEventListener('pointerup', (e) => {
    const d = state.drag;
    state.drag = null;
    if (!d || !interactive()) return;
    if (d.box) {
      const box = { x0: Math.min(d.start.x, d.now.x), y0: Math.min(d.start.y, d.now.y), x1: Math.max(d.start.x, d.now.x), y1: Math.max(d.start.y, d.now.y) };
      if (box.x1 - box.x0 > 4 && box.y1 - box.y0 > 4) setBox(box);
      else requestDraw();
      return;
    }
    addPoint(d.start.x, d.start.y, d.negative ? LABEL.NEGATIVE : LABEL.POSITIVE);
  });
  overlay.addEventListener('pointercancel', () => {
    state.drag = null;
    requestDraw();
  });
  overlay.addEventListener('pointerleave', () => {
    state.hover = null;
    state.preview = null;
    clearTimeout(hoverTimer);
    requestDraw();
  });
  overlay.addEventListener('contextmenu', (e) => interactive() && e.preventDefault());

  // ---------- drop / paste / keys ----------
  stage.addEventListener('dragover', (e) => {
    if (state.phase === 'unavailable' || !e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault();
    if (!state.dropping) {
      state.dropping = true;
      stage.classList.add('is-dropping');
    }
  });
  stage.addEventListener('dragleave', (e) => {
    if (stage.contains(e.relatedTarget)) return;
    state.dropping = false;
    stage.classList.remove('is-dropping');
  });
  stage.addEventListener('drop', (e) => {
    e.preventDefault();
    state.dropping = false;
    stage.classList.remove('is-dropping');
    const file = [...(e.dataTransfer?.files ?? [])].find((f) => f.type.startsWith('image/'));
    if (file) loadFile(file);
  });
  const onPaste = (e) => {
    if (state.phase === 'unavailable' || isTyping(e.target)) return;
    const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith('image/'));
    const file = item?.getAsFile();
    if (file) {
      e.preventDefault();
      loadFile(file);
    }
  };
  const onKey = (e) => {
    if (isTyping(e.target)) return;
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      undo();
    } else if (e.key === 'Escape' && state.drag) {
      state.drag = null;
      requestDraw();
    }
  };
  document.addEventListener('paste', onPaste);
  document.addEventListener('keydown', onKey);

  // ---------- geometry & drawing ----------
  // The stage takes the image's aspect once the model is ready; before that it is a
  // poster sized for the panel on top, with the image cover-cropped behind it.
  let layoutKey = '';
  function poster() {
    return state.phase !== 'ready' || Boolean(state.camera) || Boolean(state.encodeError) || !state.image;
  }

  function layout(force = false) {
    const avail = stageWrap.clientWidth || 800;
    const maxH = Math.max(320, Math.min(window.innerHeight * 0.72, 780));
    const covered = poster();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const key = `${avail}|${maxH}|${covered}|${state.image?.key}|${dpr}`;
    if (!force && key === layoutKey) return;
    layoutKey = key;
    let w = avail;
    let hh = Math.round(Math.max(Math.min(380, maxH), Math.min(maxH, avail * 0.62)));
    if (!covered) {
      const scale = Math.min(avail / state.image.width, maxH / state.image.height);
      w = Math.round(state.image.width * scale);
      hh = Math.round(state.image.height * scale);
    }
    stage.style.width = `${w}px`;
    stage.style.height = `${hh}px`;
    for (const c of [imageCanvas, overlay]) {
      const cw = Math.round(w * dpr);
      const ch = Math.round(hh * dpr);
      if (c.width !== cw || c.height !== ch) {
        c.width = cw;
        c.height = ch;
      }
    }
    const ctx = imageCanvas.getContext('2d');
    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    if (state.image) {
      const { canvas, width, height } = state.image;
      ctx.imageSmoothingQuality = 'high';
      if (covered) {
        const s = Math.max(imageCanvas.width / width, imageCanvas.height / height);
        ctx.drawImage(canvas, (imageCanvas.width - width * s) / 2, (imageCanvas.height - height * s) / 2, width * s, height * s);
      } else {
        ctx.drawImage(canvas, 0, 0, imageCanvas.width, imageCanvas.height);
      }
    }
    requestDraw();
  }

  const resizer = new ResizeObserver(() => layout());
  const onResize = () => layout();
  window.addEventListener('resize', onResize);
  resizer.observe(stageWrap);

  let dirty = false;
  let lastDraw = 0;
  let onScreen = true;
  const visibility = new IntersectionObserver(([entry]) => {
    onScreen = entry.isIntersecting;
    if (onScreen) requestDraw();
  });
  visibility.observe(stage);

  let animateUntil = 0;
  function requestDraw() {
    dirty = true;
    animateUntil = performance.now() + ANT_LINGER_MS;
    if (!raf && !disposed) raf = requestAnimationFrame(drawFrame);
  }

  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Redraws on change. Between changes only the marching outline moves, at a
  // modest frame rate, for a few seconds after the last interaction, and only
  // while the stage is on screen, so an idle page costs nothing.
  function drawFrame(now) {
    raf = 0;
    const animated = !reduceMotion && onScreen && !document.hidden && now < animateUntil && state.objects.some((o) => o.shape);
    if (!dirty && now - lastDraw < ANT_FRAME_MS) {
      if (animated) raf = requestAnimationFrame(drawFrame);
      return;
    }
    dirty = false;
    lastDraw = now;
    const ctx = overlay.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (poster()) return;
    const W = overlay.width;
    const H = overlay.height;
    const dpr = W / Math.max(1, overlay.clientWidth);
    const dash = reduceMotion ? 0 : -(now / 60) % 16;

    state.objects.forEach((o, i) => {
      if (!o.shape) return;
      const path = scaledPath(o.shape, W, H);
      const active = i === state.active;
      ctx.save();
      ctx.globalAlpha = active ? 0.45 : 0.32;
      ctx.fillStyle = o.color;
      ctx.fill(path, 'evenodd');
      ctx.globalAlpha = 1;
      ctx.lineJoin = 'round';
      ctx.strokeStyle = o.color;
      ctx.lineWidth = (active ? 2.5 : 1.75) * dpr;
      ctx.stroke(path);
      if (active) {
        ctx.strokeStyle = 'rgba(255,255,255,0.95)';
        ctx.lineWidth = 1.25 * dpr;
        ctx.setLineDash([6 * dpr, 5 * dpr]);
        ctx.lineDashOffset = dash * dpr;
        ctx.stroke(path);
      }
      ctx.restore();
    });

    if (state.preview?.shape) {
      const path = scaledPath(state.preview.shape, W, H);
      ctx.save();
      ctx.globalAlpha = 0.28;
      ctx.fillStyle = state.preview.color;
      ctx.fill(path, 'evenodd');
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1.25 * dpr;
      ctx.setLineDash([3 * dpr, 4 * dpr]);
      ctx.stroke(path);
      ctx.restore();
    }

    const sx = W / state.image.width;
    const sy = H / state.image.height;
    state.objects.forEach((o, i) => {
      const alpha = i === state.active ? 1 : 0.55;
      if (o.box) drawBox(ctx, o.box, sx, sy, dpr, o.color, alpha, false);
      o.points.forEach((p) => drawPoint(ctx, p.x * sx, p.y * sy, p.label, o.color, dpr, alpha));
    });
    const d = state.drag;
    if (d?.box) {
      const box = { x0: d.start.x, y0: d.start.y, x1: d.now.x, y1: d.now.y };
      drawBox(ctx, box, sx, sy, dpr, activeObject().color, 1, true);
    } else if (state.hover && !d) {
      drawCursor(ctx, state.hover.x * sx, state.hover.y * sy, dpr, state.mode);
    }
    if (animated) raf = requestAnimationFrame(drawFrame);
  }

  function scaledPath(shape, W, H) {
    const key = `${W}x${H}`;
    if (shape.key !== key) {
      const p = new Path2D();
      p.addPath(shape.path, new DOMMatrix().scale(W / shape.lw, H / shape.lh).translate(0.5, 0.5));
      Object.assign(shape, { key, scaled: p });
    }
    return shape.scaled;
  }

  function drawPoint(ctx, x, y, label, color, dpr, alpha) {
    const r = 9.5 * dpr;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.shadowColor = 'rgba(0,0,0,0.45)';
    ctx.shadowBlur = 8 * dpr;
    ctx.beginPath();
    ctx.arc(x, y, r + 1.25 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.arc(x, y, r - 1 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = label === LABEL.NEGATIVE ? '#e11d48' : color;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.beginPath();
    ctx.moveTo(x - r * 0.45, y);
    ctx.lineTo(x + r * 0.45, y);
    if (label !== LABEL.NEGATIVE) {
      ctx.moveTo(x, y - r * 0.45);
      ctx.lineTo(x, y + r * 0.45);
    }
    ctx.lineWidth = 2 * dpr;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.restore();
  }

  function drawBox(ctx, b, sx, sy, dpr, color, alpha, live) {
    const x = Math.min(b.x0, b.x1) * sx;
    const y = Math.min(b.y0, b.y1) * sy;
    const w = Math.abs(b.x1 - b.x0) * sx;
    const hh = Math.abs(b.y1 - b.y0) * sy;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineWidth = 3.5 * dpr;
    ctx.strokeStyle = 'rgba(0,0,0,0.35)';
    ctx.strokeRect(x, y, w, hh);
    ctx.lineWidth = 2 * dpr;
    ctx.strokeStyle = live ? '#fff' : color;
    if (live) ctx.setLineDash([7 * dpr, 5 * dpr]);
    ctx.strokeRect(x, y, w, hh);
    if (live) {
      ctx.globalAlpha = 0.12;
      ctx.fillStyle = color;
      ctx.fillRect(x, y, w, hh);
    }
    ctx.restore();
  }

  function drawCursor(ctx, x, y, dpr, mode) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, 5 * dpr, 0, Math.PI * 2);
    ctx.lineWidth = 2 * dpr;
    ctx.strokeStyle = 'rgba(0,0,0,0.45)';
    ctx.stroke();
    ctx.lineWidth = 1.25 * dpr;
    ctx.strokeStyle = mode === LABEL.NEGATIVE ? '#fda4af' : '#fff';
    ctx.stroke();
    ctx.restore();
  }

  // ---------- export ----------
  function cutout(objects) {
    const shapes = objects.filter((o) => o.shape);
    if (!shapes.length || !state.image) return null;
    const { canvas: src, width, height } = state.image;
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const o of shapes) {
      const b = o.shape.bounds;
      const fx = width / o.shape.lw;
      const fy = height / o.shape.lh;
      x0 = Math.min(x0, (b.x0 + 0.5) * fx);
      y0 = Math.min(y0, (b.y0 + 0.5) * fy);
      x1 = Math.max(x1, (b.x1 + 0.5) * fx);
      y1 = Math.max(y1, (b.y1 + 0.5) * fy);
    }
    const pad = 2;
    x0 = Math.max(0, Math.floor(x0) - pad);
    y0 = Math.max(0, Math.floor(y0) - pad);
    x1 = Math.min(width, Math.ceil(x1) + pad);
    y1 = Math.min(height, Math.ceil(y1) + pad);
    const out = document.createElement('canvas');
    out.width = Math.max(1, x1 - x0);
    out.height = Math.max(1, y1 - y0);
    const ctx = out.getContext('2d');
    ctx.translate(-x0, -y0);
    for (const o of shapes) {
      const p = new Path2D();
      p.addPath(o.shape.path, new DOMMatrix().scale(width / o.shape.lw, height / o.shape.lh).translate(0.5, 0.5));
      ctx.fill(p, 'evenodd');
    }
    ctx.globalCompositeOperation = 'source-in';
    ctx.drawImage(src, 0, 0);
    return out;
  }

  function toBlob(canvas) {
    return new Promise((res) => canvas.toBlob(res, 'image/png'));
  }

  async function download(objects, name) {
    const c = cutout(objects);
    if (!c) return;
    const blob = await toBlob(c);
    const url = URL.createObjectURL(blob);
    h('a', { href: url, download: `${name}.png` }).click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  async function copy(btn) {
    const c = cutout(state.objects);
    if (!c) return;
    const label = btn.querySelector('.seg-btn-label');
    try {
      if (!window.ClipboardItem || !navigator.clipboard?.write) throw new Error('unsupported');
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': toBlob(c) })]);
      label.textContent = 'Copied';
    } catch {
      label.textContent = 'Copy blocked';
    }
    setTimeout(() => {
      if (label.isConnected) label.textContent = 'Copy';
    }, 1600);
  }

  // ---------- rendering ----------
  function renderAll() {
    if (disposed) return;
    layout();
    renderToolbar();
    renderPanel();
    renderStatus();
    renderTray();
    renderSide();
    requestDraw();
  }

  function renderToolbar() {
    toolbar.hidden = state.phase === 'unavailable' || state.phase === 'checking';
    const enabled = state.phase === 'ready' && state.image;
    const canUndo = state.history.length > 0;
    const hasAny = state.objects.some((o) => o.points.length || o.box);
    replace(toolbar,
      h(
        'div',
        { class: 'segmented', role: 'group', 'aria-label': 'Click adds' },
        modeBtn(LABEL.POSITIVE, 'Include', icon('plus', 14)),
        modeBtn(LABEL.NEGATIVE, 'Exclude', h('span', { class: 'seg-minus', 'aria-hidden': 'true' })),
      ),
      h(
        'div',
        { class: 'segmented segmented--sm', role: 'group', 'aria-label': 'Mask detail' },
        h('span', { class: 'segmented__label' }, 'Detail'),
        DETAIL.map((d) =>
          h('button', { class: state.detail === d.id ? 'segmented__btn is-active' : 'segmented__btn', type: 'button', title: d.title, 'aria-pressed': String(state.detail === d.id), onclick: () => setDetail(d.id) }, d.label),
        ),
      ),
      h(
        'div',
        { class: 'seg-toolbar__right' },
        h('button', { class: 'btn btn--ghost btn--sm', type: 'button', disabled: !enabled || !canUndo, onclick: undo, title: 'Undo (Ctrl/⌘ Z)', 'aria-label': 'Undo' }, undoIcon(), h('span', { class: 'seg-tool-label' }, 'Undo')),
        h('button', { class: 'btn btn--ghost btn--sm', type: 'button', disabled: !enabled || !hasAny, onclick: clearAll, 'aria-label': 'Clear all' }, icon('close', 14), h('span', { class: 'seg-tool-label' }, 'Clear')),
      ),
    );
  }

  function modeBtn(mode, label, glyph) {
    const on = state.mode === mode;
    return h('button', { class: on ? 'segmented__btn seg-mode is-active' : 'segmented__btn seg-mode', type: 'button', 'aria-pressed': String(on), onclick: () => setMode(mode) }, glyph, label);
  }

  function renderPanel() {
    stage.classList.toggle('has-image', Boolean(state.image));
    stage.classList.toggle('is-encoding', state.encoding);
    const content = panelContent();
    panel.hidden = !content;
    stage.classList.toggle('is-covered', Boolean(content));
    replace(panel, content);
  }

  function panelContent() {
    if (state.camera) {
      return h(
        'div',
        { class: 'seg-camera' },
        state.camera.video,
        h(
          'div',
          { class: 'seg-camera__bar' },
          h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => (stopCamera(), renderAll()) }, 'Cancel'),
          h('button', { class: 'seg-shutter', type: 'button', 'aria-label': 'Take photo', onclick: snapshot }),
          h('span', { class: 'seg-camera__spacer' }),
        ),
      );
    }
    const gpu = gpuState.get().status;
    switch (state.phase) {
      case 'checking':
        return h('div', { class: 'seg-card' }, h('p', { class: 'preload__sub' }, 'Looking for the model…'));
      case 'unavailable':
        return h(
          'div',
          { class: 'seg-card' },
          h('div', { class: 'preload__glyph' }, icon('lock', 26)),
          h('h2', { class: 'preload__title' }, 'Not available in this build'),
          h('p', { class: 'preload__sub' }, `${model.name} is not hosted here yet, so this demo cannot run.`),
          licenseBadgeEl(model),
          state.reasons.length ? h('ul', { class: 'seg-reasons' }, state.reasons.map((r) => h('li', null, r.text ?? String(r)))) : null,
        );
      case 'idle':
        return h(
          'div',
          { class: 'seg-card seg-card--glass' },
          h('div', { class: 'app-glyph', 'aria-hidden': 'true' }, appIcon(demo.id, 64)),
          h('h2', { class: 'preload__title' }, model.name),
          h('p', { class: 'preload__sub' }, 'Image encoder + mask decoder, compiled for WebGPU with ExecuTorch'),
          h(
            'button',
            { class: 'btn btn--primary btn--xl', type: 'button', disabled: gpu !== 'ready', onclick: load },
            icon('bolt', 18),
            gpu === 'ready' ? `Load model · ${formatBytes(model.approxBytes)}` : gpu === 'probing' ? 'Detecting GPU…' : 'Needs WebGPU',
          ),
          h('p', { class: 'preload__note' }, 'Downloads once, then loads from your browser cache. Your images never leave this tab.'), hfBadge(source), onDeviceNote(model),
        );
      case 'loading': {
        const L = state.load;
        const frac = L.total ? L.loaded / L.total : 0;
        return h(
          'div',
          { class: 'seg-card seg-card--glass seg-card--loader' },
          h('h2', { class: 'loader__title' }, `Loading ${model.shortName}`),
          h('div', { class: 'progress' }, h('div', { class: 'progress__bar', style: { width: `${(L.done.has('weights') ? 1 : frac) * 100}%` } })),
          h('p', { class: 'loader__detail' }, L.total ? `${formatBytes(L.loaded)} / ${formatBytes(L.total)}${L.fromCache ? ' · from cache' : ''}` : ' '),
          h(
            'ol',
            { class: 'steps' },
            LOAD_STAGES.map((s) =>
              h('li', { class: L.done.has(s.id) ? 'steps__item is-done' : L.stage === s.id ? 'steps__item is-active' : 'steps__item' }, h('span', { class: 'steps__dot' }), s.label),
            ),
          ),
          hfBadge(source),
          h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: cancelLoad }, 'Cancel'),
        );
      }
      case 'error':
        return h(
          'div',
          { class: 'seg-card seg-card--glass' },
          h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, `Could not load the model: ${state.loadError}`)),
          h('button', { class: 'btn btn--primary', type: 'button', onclick: load }, icon('refresh', 16), 'Try again'),
        );
      default:
        break;
    }
    if (!state.image) {
      return h(
        'div',
        { class: 'seg-drop' },
        h('div', { class: 'seg-drop__icon' }, imageIcon(28)),
        h('p', { class: 'seg-drop__title' }, 'Drop an image here'),
        h('p', { class: 'preload__sub' }, 'or paste one, upload a file, or take a photo'),
        h('div', { class: 'seg-drop__actions' }, uploadBtn('btn btn--primary btn--sm'), cameraBtn('btn btn--ghost btn--sm')),
      );
    }
    if (state.encodeError) {
      return h(
        'div',
        { class: 'seg-card seg-card--glass' },
        h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, `The image encoder failed: ${state.encodeError}`)),
        h('button', { class: 'btn btn--primary btn--sm', type: 'button', onclick: encode }, icon('refresh', 15), 'Try again'),
      );
    }
    return null;
  }

  function renderStatus() {
    if (disposed) return;
    const ready = state.phase === 'ready' && state.image && !state.camera;
    hint.hidden = !ready || state.encoding || Boolean(state.encodeError);
    replace(hint,
      'ontouchstart' in window && !matchMedia('(pointer: fine)').matches
        ? 'Tap an object · drag for a box'
        : state.mode === LABEL.NEGATIVE
          ? 'Click to exclude · drag for a box'
          : ['Click an object · ', h('kbd', { class: 'seg-kbd' }, 'Shift'), ' click to exclude · drag for a box'],
    );
    const pending = state.objects.some((o) => o.pending);
    status.hidden = !(ready && (state.encoding || pending));
    if (state.encoding) {
      const s = (performance.now() - state.encodeStarted) / 1000;
      replace(status, h('span', { class: 'seg-spinner' }), `Reading the image · ${s.toFixed(1)} s`);
    } else if (pending) {
      replace(status, h('span', { class: 'seg-spinner' }), 'Decoding mask…');
    }
    const parts = [];
    if (state.encodeMs != null) parts.push(`encoder ${formatMs(state.encodeMs)}`);
    if (state.decodeMs != null) parts.push(`decoder ${formatMs(state.decodeMs)}`);
    stat.hidden = !ready || !parts.length;
    stat.textContent = parts.join(' · ');
  }

  function renderTray() {
    const show = state.phase !== 'unavailable' && state.phase !== 'checking';
    tray.hidden = !show;
    if (!show) return;
    replace(tray,
      state.samples.map((s) =>
        h(
          'button',
          { class: state.image?.key === s.url ? 'seg-thumb is-active' : 'seg-thumb', type: 'button', title: s.label, onclick: () => loadUrl(s.url, s.label) },
          h('img', { class: 'seg-thumb__img', src: s.url, alt: s.label, loading: 'lazy' }),
          h('span', { class: 'seg-thumb__label' }, s.label),
        ),
      ),
      h('div', { class: 'seg-tray__actions' }, uploadBtn('seg-action'), cameraBtn('seg-action'), h('a', { class: 'sample-credits', href: '#/models' }, 'Photo credits')),
    );
  }

  function uploadBtn(cls) {
    return h('button', { class: cls, type: 'button', title: 'Upload an image (you can also drop or paste one)', onclick: () => fileInput.click() }, uploadIcon(), 'Upload');
  }

  function cameraBtn(cls) {
    return h('button', { class: cls, type: 'button', title: 'Take a photo with your camera', onclick: startCamera }, cameraIcon(), 'Camera');
  }

  function renderSide() {
    const ready = state.phase === 'ready' && state.image;
    const any = state.objects.some((o) => o.shape);
    const cur = activeObject();
    replace(side,
      h('h2', { class: 'metrics__title' }, 'Objects'),
      h('div', { class: 'metrics__sub' }, h('span', null, ready ? 'Each object gets its own color' : 'Load the model, then click the image')),
      h(
        'ul',
        { class: 'seg-objects' },
        state.objects.map((o, i) =>
          h(
            'li',
            { class: i === state.active ? 'seg-object is-active' : 'seg-object' },
            h(
              'button',
              { class: 'seg-object__main', type: 'button', onclick: () => selectObject(i), 'aria-pressed': String(i === state.active) },
              h('span', { class: 'seg-swatch', style: { background: o.color } }),
              h('span', { class: 'seg-object__text' }, h('span', { class: 'seg-object__name' }, `Object ${o.id + 1}`), h('span', { class: 'seg-object__meta' }, objectMeta(o))),
            ),
            o.shape ? h('button', { class: 'icon-btn seg-object__act', type: 'button', title: 'Download this cut-out', 'aria-label': `Download object ${o.id + 1} as PNG`, onclick: () => download([o], `object-${o.id + 1}`) }, downloadIcon()) : null,
            o.points.length || o.box ? h('button', { class: 'icon-btn seg-object__act', type: 'button', title: 'Remove object', 'aria-label': `Remove object ${o.id + 1}`, onclick: () => removeObject(i) }, icon('close', 15)) : null,
          ),
        ),
      ),
      h('button', { class: 'btn btn--ghost btn--sm btn--block', type: 'button', disabled: !ready || (!cur.points.length && !cur.box), onclick: addObject }, icon('plus', 15), 'Add object'),
      h('h3', { class: 'metrics__h' }, 'Export'),
      h(
        'div',
        { class: 'seg-export' },
        h('button', { class: 'btn btn--primary btn--sm', type: 'button', disabled: !any, onclick: () => download(state.objects, 'cutout') }, scissorsIcon(), 'Cut out PNG'),
        h('button', { class: 'btn btn--ghost btn--sm', type: 'button', disabled: !any, onclick: (e) => copy(e.currentTarget) }, icon('copy', 15), h('span', { class: 'seg-btn-label' }, 'Copy')),
      ),
      h('p', { class: 'metrics__note' }, 'A transparent PNG of every object, cropped to fit, at the image’s resolution.'),
      h('h3', { class: 'metrics__h' }, 'How it works'),
      h(
        'ol',
        { class: 'pipeline' },
        h('li', null, `The image encoder reads the picture once${state.encodeMs != null ? ` (${formatMs(state.encodeMs)} here)` : ''} and keeps its features on the GPU.`),
        h('li', null, `Each click or box runs only the small mask decoder${state.decodeMs != null ? ` (${formatMs(state.decodeMs)})` : ''}, so masks follow you as you work.`),
        h('li', null, 'Both run in this tab with ExecuTorch on WebGPU. Nothing is uploaded.'),
      ),
      h('p', { class: 'metrics__note' }, 'This build’s decoder takes two prompt points at a time, so objects with more points are combined from several decodes.'),
    );
  }

  function objectMeta(o) {
    if (o.error) return `Failed: ${o.error}`;
    if (o.pending) return state.encoding ? 'Waiting for the encoder…' : 'Decoding…';
    const pos = o.points.filter((p) => p.label === LABEL.POSITIVE).length;
    const neg = o.points.length - pos;
    const bits = [];
    if (pos) bits.push(`${pos} in`);
    if (neg) bits.push(`${neg} out`);
    if (o.box) bits.push('box');
    if (!bits.length) return 'Empty · click the image';
    if (!o.result) return bits.join(' · ');
    if (o.result.area === 0) return `${bits.join(' · ')} · no mask`;
    return `${bits.join(' · ')} · ${pct(o.result.area)} of image · ${Math.round(o.result.score * 100)}% stable`;
  }

  function showNotice(text) {
    state.notice = text;
    replace(notice, h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, text)));
    setTimeout(() => {
      if (state.notice === text) {
        state.notice = null;
        replace(notice);
      }
    }, 5000);
  }

  const unsubscribe = gpuState.subscribe((s) => {
    replace(gpuSlot, s.status === 'unavailable' || s.status === 'lost' ? gpuFallbackPanel(s, { compact: true }) : '');
    if (!disposed) renderPanel();
  });

  layout();
  renderAll();
  resolve();

  return () => {
    disposed = true;
    loadAbort?.abort();
    cancelAnimationFrame(raf);
    cancelAnimationFrame(panelFrame);
    clearTimeout(hoverTimer);
    clearInterval(encodeTimer);
    resizer.disconnect();
    window.removeEventListener('resize', onResize);
    visibility.disconnect();
    unsubscribe();
    stopCamera();
    document.removeEventListener('paste', onPaste);
    document.removeEventListener('keydown', onKey);
  };
}

// Outline + fill geometry for one mask, in decoder-grid units. Specks smaller than
// a couple of grid cells are dropped; corner-cutting smooths the stair-steps.
function buildShape(result) {
  const { logits, logitsWidth: lw, logitsHeight: lh } = result;
  const loops = maskContours(logits, lw, lh).filter((l) => Math.abs(polygonArea(l)) >= 2.5).map((l) => chaikin(chaikin(l)));
  if (!loops.length) return null;
  const path = new Path2D();
  const bounds = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
  for (const loop of loops) {
    path.moveTo(loop[0][0], loop[0][1]);
    for (let i = 1; i < loop.length; i++) path.lineTo(loop[i][0], loop[i][1]);
    path.closePath();
    for (const [x, y] of loop) {
      if (x < bounds.x0) bounds.x0 = x;
      if (y < bounds.y0) bounds.y0 = y;
      if (x > bounds.x1) bounds.x1 = x;
      if (y > bounds.y1) bounds.y1 = y;
    }
  }
  return { path, lw, lh, bounds, key: null, scaled: null };
}

function polygonArea(loop) {
  let a = 0;
  for (let i = 0, j = loop.length - 1; i < loop.length; j = i++) a += (loop[j][0] + loop[i][0]) * (loop[j][1] - loop[i][1]);
  return a / 2;
}

function chaikin(loop) {
  const out = [];
  for (let i = 0; i < loop.length; i++) {
    const [ax, ay] = loop[i];
    const [bx, by] = loop[(i + 1) % loop.length];
    out.push([0.75 * ax + 0.25 * bx, 0.75 * ay + 0.25 * by], [0.25 * ax + 0.75 * bx, 0.25 * ay + 0.75 * by]);
  }
  return out;
}

function pct(f) {
  return f < 0.01 ? '<1%' : `${Math.round(f * 100)}%`;
}

function isTyping(el) {
  return el instanceof HTMLElement && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
}

const stroke = (d, size = 15) =>
  svg(`<svg class="icon" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`);
const undoIcon = () => stroke('<path d="M9 14 4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/>');
const uploadIcon = () => stroke('<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>');
const cameraIcon = () => stroke('<path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1Z"/><circle cx="12" cy="13.5" r="3.5"/>');
const downloadIcon = () => stroke('<path d="M12 4v12M7 11l5 5 5-5"/><path d="M5 20h14"/>');
const scissorsIcon = () => stroke('<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M8.6 7.6 20 18M8.6 16.4 20 6"/>');
const imageIcon = (size) => stroke('<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/>', size);
