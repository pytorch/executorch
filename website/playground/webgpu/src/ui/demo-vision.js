import { getModel } from '../config/models.js';
import { gpuState } from '../gpu/context.js';
import { h, replace } from '../lib/dom.js';
import { formatBytes, formatMs } from '../lib/format.js';
import { ForwardModel } from '../runtime/forward.js';
import { resolveModelSource } from '../runtime/sources.js';
import { createDepthTask } from '../vision/depth.js';
import { createDetectTask } from '../vision/detect.js';
import { canvasToBlob, downloadBlob, loadImage, rasterize } from '../vision/image.js';
import { createMattingTask } from '../vision/matting.js';
import { createSuperresTask } from '../vision/superres.js';
import { licenseBadgeEl } from './badges.js';
import { browseMore, detailHeader } from './detail-shell.js';
import { gpuFallbackPanel } from './gpu-fallback.js';
import { appIcon } from './app-icons.js';
import { hfBadge } from './hf-badge.js';
import { icon } from './icons.js';
import { onDeviceNote } from './on-device.js';
import { confirmModelMemory } from './memory-guard.js';

const TASKS = { depth: createDepthTask, detect: createDetectTask, bgremove: createMattingTask, superres: createSuperresTask };
const COPY = {
  depth: { verb: 'Estimating depth', ready: 'Pick a photo or turn on your camera to see depth.' },
  detect: { verb: 'Detecting', ready: 'Pick a photo or turn on your camera to find objects.' },
  bgremove: { verb: 'Cutting out', ready: 'Pick a portrait or turn on your camera for a virtual background.' },
  superres: { verb: 'Upscaling', ready: 'Pick a small photo to upscale it 4×.' },
};
const LOAD_STAGES = [
  { id: 'runtime', label: 'Start the ExecuTorch runtime' },
  { id: 'weights', label: 'Download the model' },
  { id: 'gpu', label: 'Prepare it on your GPU' },
];
const MAX_STILL = 1600;
const MAX_LIVE = 1280;
const glyph = (paths) => () =>
  h('span', {
    class: 'vision-btn-icon',
    html: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`,
  });
const cameraIcon = glyph('<path d="M3 8a2 2 0 0 1 2-2h2l2-2h6l2 2h2a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8Z"/><circle cx="12" cy="13" r="3.5"/>');
const uploadIcon = glyph('<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>');
const downloadIcon = glyph('<path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 20h16"/>');

export function mountVision(root, demo) {
  const model = getModel(demo.models[0]);
  const copy = COPY[demo.task];
  const state = {
    phase: 'checking', // checking | unavailable | idle | loading | ready | error
    source: null,
    reasons: [],
    load: { stage: null, fractions: {}, detail: '' },
    loadError: null,
    loadAbort: null,
    net: null,
    frame: null, // { canvas, label, kind: 'image' | 'camera' | 'video', video?, stream?, mirror?, sampleId? }
    result: null,
    resultFor: null,
    busy: false,
    liveBusy: false,
    stillToken: 0,
    progress: null,
    error: null,
    stats: { ms: null, fps: null, lastDone: 0, detail: '' },
    split: 0.5,
    cameraPaused: false,
    lastStill: null,
    facing: 'user',
  };
  let disposed = false;
  let raf = 0;
  let dirty = true;
  let task = null;

  const canvas = h('canvas', { class: 'vision-canvas', 'aria-label': `${demo.title} output` });
  const splitEl = h('div', { class: 'vision-split', hidden: true }, h('span', { class: 'vision-split__knob' }, icon('chevronLeft', 14), icon('chevronRight', 14)));
  const tagL = h('span', { class: 'vision-tag vision-tag--l', hidden: true });
  const tagR = h('span', { class: 'vision-tag vision-tag--r', hidden: true });
  const frameWrap = h('div', { class: 'vision-frame' }, canvas, splitEl, tagL, tagR);
  const statEl = h('span', { class: 'vision-stat', hidden: true });
  const legendEl = h('div', { class: 'vision-legend', hidden: true });
  const busyEl = h('div', { class: 'vision-busy', hidden: true });
  const overlay = h('div', { class: 'vision-overlay' });
  const errorEl = h('div', { class: 'vision-error', hidden: true });
  const videoHolder = h('div', { class: 'vision-video-holder', 'aria-hidden': 'true' });
  const stage = h('div', { class: 'vision-stage' }, frameWrap, statEl, legendEl, busyEl, overlay, errorEl, videoHolder);

  const fileInput = h('input', { type: 'file', hidden: true, onchange: (e) => e.target.files[0] && openFile(e.target.files[0]) });
  const inputSection = h('section', { class: 'vision-panel__section' });
  const controlsSection = h('section', { class: 'vision-panel__section' });
  const summarySection = h('section', { class: 'vision-panel__section', hidden: true });
  const actionsSection = h('section', { class: 'vision-panel__section vision-panel__actions' });
  const aboutSection = h('section', { class: 'vision-panel__section vision-panel__about' });
  const panel = h('aside', { class: 'vision-panel', 'aria-label': 'Input and view settings' }, inputSection, controlsSection, summarySection, actionsSection, aboutSection, fileInput);
  const layout = h('div', { class: 'vision' }, stage, panel);

  replace(root, h('div', { class: 'page detail vision-page' }, detailHeader(demo), layout, browseMore(demo.id)));

  // ---------- model ----------
  resolveModelSource(model).then((source) => {
    if (disposed) return;
    if (source.kind !== 'unavailable') {
      state.source = source;
      state.phase = 'idle';
      task = TASKS[demo.task]({
        spec: source.spec,
        redraw: ({ controls = false, summary = false } = {}) => {
          dirty = true;
          if (controls) renderControls();
          if (controls || summary) renderSummary();
          renderLegend();
        },
        rerun: () => {
          renderControls();
          if (state.frame && !isLive()) runStill();
        },
        error: (msg) => showError(msg),
      });
      if (task.overlay) frameWrap.append(task.overlay);
      fileInput.accept = task.accept;
    } else {
      state.phase = 'unavailable';
      state.reasons = source.reasons ?? [];
    }
    renderAll();
  }).then(() => state.source && ForwardModel.isLoaded(state.source) && load());

  const unsubscribeGpu = gpuState.subscribe(() => {
    if (state.phase === 'idle' || state.phase === 'error') renderOverlay();
  });

  function displayName() {
    return state.source?.spec?.name ?? model.name;
  }

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
      if (disposed || controller.signal.aborted) {
        net.dispose();
        return;
      }
      Object.assign(state, { net, phase: 'ready' });
      renderAll();
      if (state.frame) runStill();
      else if (samples().length) pickSample(samples()[0]);
    } catch (err) {
      if (err.name === 'AbortError' || disposed) return;
      Object.assign(state, { phase: 'error', loadError: err.message });
      renderAll();
    } finally {
      state.loadAbort = null;
    }
  }

  function cancelLoad() {
    state.loadAbort?.abort();
    state.phase = 'idle';
    renderAll();
  }

  // ---------- inputs ----------
  function samples() {
    if (!state.source || state.source.kind === 'unavailable') return [];
    const list = state.source.shared?.visionSamples?.[demo.task] ?? [];
    return list.filter((s) => task.live || !s.video);
  }

  function isLive() {
    return state.frame?.kind === 'camera' || state.frame?.kind === 'video';
  }

  function setFrame(frame) {
    stopLive();
    // Remember the photo on screen so stopping the camera can bring it back.
    if (state.frame?.kind === 'image' && frame && frame.kind !== 'image') {
      state.lastStill = { frame: state.frame, result: state.resultFor === state.frame ? state.result : null };
    } else if (frame?.kind === 'image') {
      state.lastStill = null;
    }
    state.cameraPaused = false;
    state.frame = frame;
    state.result = null;
    state.resultFor = null;
    state.error = null;
    state.stats = { ms: null, fps: null, lastDone: 0, detail: '' };
    task?.resetLive?.();
    dirty = true;
    renderInputs();
    renderSummary();
    renderActions();
    renderOverlay();
    renderStat();
    renderLegend();
    renderError();
  }

  async function useImage(url, label, sampleId = null) {
    try {
      const img = await loadImage(url);
      if (disposed) return;
      setFrame({ canvas: rasterize(img, MAX_STILL), label, kind: 'image', sampleId });
      if (state.phase === 'ready') runStill();
    } catch (err) {
      showError(err.message);
    }
  }

  function pickSample(s) {
    if (s.video) return useVideo(state.source.url(s.path), s.label, s.path);
    useImage(state.source.url(s.path), s.label, s.path);
  }

  function openFile(file) {
    fileInput.value = '';
    if (file.type.startsWith('video/')) {
      if (!task.live) return showError('This demo works on photos. Pick an image instead.');
      return useVideo(URL.createObjectURL(file), file.name, null, true);
    }
    if (!file.type.startsWith('image/')) return showError('That file is not an image.');
    const url = URL.createObjectURL(file);
    useImage(url, file.name).finally(() => URL.revokeObjectURL(url));
  }

  async function useVideo(url, label, sampleId, ownsUrl = false) {
    const video = h('video', { class: 'vision-video', muted: true, playsinline: true, loop: true, preload: 'auto' });
    video.muted = true;
    video.addEventListener('error', () => {
      if (state.frame?.video !== video) return;
      setFrame(null);
      renderAll();
      showError('This browser cannot play that video. Try an MP4 (H.264) or WebM file.');
    });
    video.src = url;
    videoHolder.append(video);
    setFrame({ canvas: null, label, kind: 'video', video, sampleId, ownsUrl });
    try {
      await video.play();
    } catch (err) {
      if (state.frame?.video === video) showError(`The video could not play: ${err.message}`);
    }
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) return showError('This browser does not expose a camera to web pages.');
    const facing = state.facing;
    setFrame({ canvas: null, label: 'Camera', kind: 'camera', starting: true, mirror: facing === 'user' });
    const frame = state.frame;
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    } catch (err) {
      if (state.frame === frame) {
        setFrame(null);
        showError(err.name === 'NotAllowedError' ? 'Camera permission was denied. Allow it in the address bar to try again.' : `The camera could not start: ${err.message}`);
      }
      return;
    }
    if (state.frame !== frame || disposed) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    const settings = stream.getVideoTracks()[0]?.getSettings?.() ?? {};
    const video = h('video', { class: 'vision-video', muted: true, playsinline: true, autoplay: true });
    video.muted = true;
    video.srcObject = stream;
    videoHolder.append(video);
    Object.assign(frame, { video, stream, starting: false, mirror: (settings.facingMode ?? facing) !== 'environment' });
    state.cameraPaused = false;
    try {
      await video.play();
    } catch {
      // Autoplay of a muted camera stream is allowed; play() can still reject on teardown.
    }
    renderInputs();
  }

  function stopLive() {
    const f = state.frame;
    if (!f) return;
    f.stream?.getTracks().forEach((t) => t.stop());
    if (f.video) {
      f.video.pause();
      if (f.ownsUrl) URL.revokeObjectURL(f.video.src);
      f.video.removeAttribute('src');
      f.video.srcObject = null;
      f.video.remove();
    }
  }

  function toggleCamera() {
    if (state.frame?.kind === 'camera') {
      const still = state.lastStill;
      setFrame(still?.frame ?? null);
      if (still?.result) Object.assign(state, { result: still.result, resultFor: still.frame });
      renderAll();
      return;
    }
    startCamera();
  }

  function flipCamera() {
    state.facing = state.facing === 'user' ? 'environment' : 'user';
    startCamera();
  }

  function onVisibility() {
    if (document.hidden) {
      if (state.frame?.kind === 'camera') {
        stopLive();
        state.frame = null;
        state.cameraPaused = true;
        renderAll();
      }
      state.frame?.video?.pause();
    } else if (state.frame?.kind === 'video') {
      state.frame.video.play().catch(() => {});
    }
  }

  function onPaste(e) {
    if (state.phase !== 'ready') return;
    const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith('image/'));
    if (item) {
      e.preventDefault();
      openFile(item.getAsFile());
    }
  }

  function onDragOver(e) {
    if (state.phase !== 'ready' || ![...(e.dataTransfer?.types ?? [])].includes('Files')) return;
    e.preventDefault();
    layout.classList.add('is-dropping');
  }

  function onDrop(e) {
    layout.classList.remove('is-dropping');
    if (state.phase !== 'ready') return;
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    e.preventDefault();
    openFile(file);
  }

  document.addEventListener('visibilitychange', onVisibility);
  document.addEventListener('paste', onPaste);
  layout.addEventListener('dragover', onDragOver);
  layout.addEventListener('dragleave', (e) => {
    if (!layout.contains(e.relatedTarget)) layout.classList.remove('is-dropping');
  });
  layout.addEventListener('drop', onDrop);

  // ---------- inference ----------
  async function runStill() {
    if (!state.net || !state.frame?.canvas || isLive()) return;
    const token = ++state.stillToken;
    const frame = state.frame;
    state.busy = true;
    state.error = null;
    state.progress = task.live ? null : { fraction: 0, detail: '' };
    renderBusy();
    renderError();
    const t0 = performance.now();
    try {
      const result = await task.infer(state.net, frame.canvas, {
        live: false,
        isStale: () => token !== state.stillToken || disposed,
        onProgress: (fraction, detail) => {
          if (token !== state.stillToken) return;
          state.progress = { fraction, detail };
          renderBusy();
        },
      });
      if (token !== state.stillToken || disposed || !result) return;
      state.result = result;
      state.resultFor = frame;
      const ms = performance.now() - t0;
      state.stats = { ms, fps: null, lastDone: performance.now(), detail: result.tiles ? `${result.tiles} tiles` : '' };
      dirty = true;
    } catch (err) {
      if (token === state.stillToken) showError(`The model failed on this image: ${err.message}`);
    } finally {
      if (token === state.stillToken) {
        state.busy = false;
        state.progress = null;
        renderBusy();
        renderStat();
        renderSummary();
        renderActions();
        renderLegend();
      }
    }
  }

  // At most one live inference is in flight; frames that arrive meanwhile are only displayed.
  function runLive(frame) {
    state.liveBusy = true;
    const t0 = performance.now();
    task
      .infer(state.net, frame.canvas, { live: true, isStale: () => state.frame !== frame || disposed })
      .then((result) => {
        if (state.frame !== frame || disposed || !result) return;
        const now = performance.now();
        const ms = now - t0;
        const s = state.stats;
        const dt = s.lastDone ? now - s.lastDone : null;
        state.stats = {
          ms: s.ms == null ? ms : s.ms * 0.8 + ms * 0.2,
          fps: dt ? (s.fps == null ? 1000 / dt : s.fps * 0.8 + (1000 / dt) * 0.2) : null,
          lastDone: now,
          detail: '',
        };
        state.result = result;
        state.resultFor = frame;
        dirty = true;
        renderStat();
        if (task.summary) renderSummary();
        if (legendEl.hidden) renderLegend();
        if (actionsSection.querySelector('button')?.disabled) renderActions();
        if (!state.error) return;
        state.error = null;
        renderError();
      })
      .catch((err) => {
        if (state.frame === frame) showError(`The model failed on a frame: ${err.message}`);
      })
      .finally(() => {
        state.liveBusy = false;
      });
  }

  function grab(frame) {
    const v = frame.video;
    if (!v || v.readyState < 2 || !v.videoWidth) return false;
    if (!frame.canvas) {
      frame.canvas = rasterize(v, MAX_LIVE, { mirror: frame.mirror });
      renderOverlay();
      renderActions();
      return true;
    }
    const g = frame.canvas.getContext('2d');
    g.save();
    if (frame.mirror) {
      g.translate(frame.canvas.width, 0);
      g.scale(-1, 1);
    }
    g.drawImage(v, 0, 0, frame.canvas.width, frame.canvas.height);
    g.restore();
    return true;
  }

  function tick() {
    raf = requestAnimationFrame(tick);
    const frame = state.frame;
    if (!frame || !task) return;
    if (isLive()) {
      if (!grab(frame)) return;
      if (state.net && !state.liveBusy && !document.hidden) runLive(frame);
      dirty = true;
    }
    if (!frame.canvas) return;
    if (dirty || (state.result && task.animating())) {
      dirty = false;
      const result = state.resultFor === frame ? state.result : null;
      task.draw(canvas, frame.canvas, result, { split: state.split, live: isLive() });
      // A resize changes how big overlay strokes must be, so draw once more at the new size.
      if (fitCanvas()) dirty = true;
      paintSplit(result);
    }
  }
  raf = requestAnimationFrame(tick);
  // Overlays scale their strokes and labels to the displayed size.
  const resizeObserver = new ResizeObserver(() => (dirty = true));
  resizeObserver.observe(stage);

  // ---------- stage interaction ----------
  let drag = null;
  frameWrap.addEventListener('pointerdown', (e) => {
    if (!state.result) return;
    const rect = canvas.getBoundingClientRect();
    const labels = task.split();
    if (task.drag?.(0, 0, 'start')) {
      drag = { kind: 'orbit', x: e.clientX, y: e.clientY, rect };
    } else if (labels) {
      drag = { kind: 'split', rect };
      state.split = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      dirty = true;
    } else return;
    frameWrap.setPointerCapture(e.pointerId);
    frameWrap.classList.add('is-dragging');
  });
  frameWrap.addEventListener('pointermove', (e) => {
    const rect = drag?.rect ?? canvas.getBoundingClientRect();
    if (task?.hover && state.result && e.pointerType === 'mouse') {
      task.hover({ x: (e.clientX - rect.left) / rect.width, y: (e.clientY - rect.top) / rect.height }, canvas, { split: state.split });
    }
    if (!drag) return;
    if (drag.kind === 'orbit') {
      task.drag((e.clientX - drag.x) / rect.width, (e.clientY - drag.y) / rect.height, 'move');
      drag.x = e.clientX;
      drag.y = e.clientY;
    } else {
      state.split = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    }
    dirty = true;
  });
  const endDrag = () => {
    if (drag?.kind === 'orbit') task.drag(0, 0, 'end');
    drag = null;
    frameWrap.classList.remove('is-dragging');
  };
  frameWrap.addEventListener('pointerup', endDrag);
  frameWrap.addEventListener('pointercancel', endDrag);
  frameWrap.addEventListener('pointerleave', () => task?.hover?.(null, canvas, { split: state.split }));
  frameWrap.addEventListener('keydown', (e) => {
    if (!task?.split() || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return;
    state.split = Math.min(1, Math.max(0, state.split + (e.key === 'ArrowLeft' ? -0.05 : 0.05)));
    dirty = true;
    e.preventDefault();
  });

  // Scales the canvas to fill the stage (up or down) at its own aspect ratio.
  let fitKey = '';
  function fitCanvas() {
    const key = `${canvas.width}x${canvas.height}@${stage.clientWidth}x${window.innerHeight}`;
    if (key === fitKey) return false;
    fitKey = key;
    const cs = getComputedStyle(stage);
    const availW = stage.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    const availH = Math.min(window.innerHeight * (availW < 600 ? 0.64 : 0.72), 760);
    const k = Math.min(availW / canvas.width, availH / canvas.height);
    const w = `${Math.floor(canvas.width * k)}px`;
    const hh = `${Math.floor(canvas.height * k)}px`;
    if (canvas.style.width === w && canvas.style.height === hh) return false;
    canvas.style.width = w;
    canvas.style.height = hh;
    return true;
  }

  function paintSplit(result) {
    const labels = result ? task.split() : null;
    splitEl.hidden = tagL.hidden = tagR.hidden = !labels;
    frameWrap.tabIndex = labels ? 0 : -1;
    frameWrap.classList.toggle('vision-frame--checker', Boolean(task.transparent?.()));
    frameWrap.classList.toggle('vision-frame--split', Boolean(labels));
    frameWrap.classList.toggle('vision-frame--orbit', Boolean(result && task.animating()));
    if (!labels) return;
    splitEl.style.left = `${state.split * 100}%`;
    tagL.textContent = labels[0];
    tagR.textContent = labels[1];
  }

  async function exportPng() {
    const out = await task.exportImage?.(state.frame.canvas, state.result);
    if (!out) return;
    downloadBlob(await canvasToBlob(out.canvas), out.name);
  }

  function showError(message) {
    state.error = message;
    renderError();
  }

  // ---------- rendering ----------
  function renderAll() {
    if (disposed) return;
    renderInputs();
    renderControls();
    renderSummary();
    renderActions();
    renderAbout();
    renderOverlay();
    renderStat();
    renderLegend();
    renderBusy();
    renderError();
    dirty = true;
  }

  function renderInputs() {
    if (state.phase !== 'ready') {
      replace(inputSection, h('h2', { class: 'vision-panel__title' }, 'Input'), h('p', { class: 'vision-panel__muted' }, state.phase === 'unavailable' ? 'No model in this build.' : 'Load the model to pick a photo or start your camera.'));
      return;
    }
    const f = state.frame;
    const cameraOn = f?.kind === 'camera';
    const list = samples();
    replace(
      inputSection,
      h('h2', { class: 'vision-panel__title' }, 'Input'),
      list.length
        ? h(
            'div',
            { class: 'vision-samples', role: 'group', 'aria-label': 'Sample images' },
            list.map((s) =>
              h(
                'button',
                { class: f?.sampleId === s.path ? 'vision-sample is-active' : 'vision-sample', type: 'button', title: s.label, onclick: () => pickSample(s) },
                h('img', { class: 'vision-sample__img', src: state.source.url(s.poster ?? s.path), alt: '', loading: 'lazy' }),
                s.video ? h('span', { class: 'vision-sample__play' }, icon('play', 11)) : null,
                h('span', { class: 'vision-sample__label' }, s.label),
              ),
            ),
          )
        : null,
      list.length ? h('a', { class: 'sample-credits', href: '#/models' }, 'Photo credits') : null,
      h(
        'div',
        { class: 'vision-input-btns' },
        h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => fileInput.click() }, uploadIcon(), task.live ? 'Upload' : 'Upload photo'),
        task.live
          ? h('button', { class: cameraOn ? 'btn btn--primary btn--sm' : 'btn btn--ghost btn--sm', type: 'button', 'aria-pressed': String(cameraOn), onclick: toggleCamera }, cameraOn ? icon('stop', 15) : cameraIcon(), cameraOn ? 'Stop camera' : 'Camera')
          : null,
        cameraOn ? h('button', { class: 'btn btn--ghost btn--sm', type: 'button', title: 'Switch between front and back cameras', onclick: flipCamera }, icon('refresh', 15), 'Flip') : null,
      ),
      h('p', { class: 'vision-panel__hint' }, task.live ? 'Or drop / paste an image or a video.' : 'Or drop / paste an image. Small pictures show the biggest gains.'),
    );
  }

  function renderControls() {
    if (!task || state.phase !== 'ready') {
      controlsSection.hidden = true;
      return;
    }
    controlsSection.hidden = false;
    replace(controlsSection, h('h2', { class: 'vision-panel__title' }, 'Output'), task.controls());
  }

  function renderSummary() {
    const node = state.phase === 'ready' && task?.summary ? task.summary(state.resultFor === state.frame ? state.result : null) : null;
    summarySection.hidden = !node;
    if (node) replace(summarySection, h('h2', { class: 'vision-panel__title' }, 'Found'), node);
  }

  function renderActions() {
    const can = state.phase === 'ready' && task?.exportImage && state.result && state.resultFor === state.frame;
    actionsSection.hidden = !(state.phase === 'ready' && task?.exportImage);
    replace(actionsSection, h('button', { class: 'btn btn--ghost btn--sm btn--block', type: 'button', disabled: !can, onclick: exportPng }, downloadIcon(), 'Download PNG'));
  }

  function renderAbout() {
    replace(
      aboutSection,
      h('p', { class: 'vision-about__model' }, displayName()),
      h('div', { class: 'vision-about__row' }, h('span', { class: 'vision-panel__muted' }, formatBytes(modelBytes(), { approx: !state.source?.spec?.pteBytes && model.approxBytesIsEstimate }))),
      h('p', { class: 'vision-panel__muted' }, 'Runs on your GPU with ExecuTorch. Your photos and camera never leave this tab.'),
    );
  }

  function renderOverlay() {
    const gpu = gpuState.get();
    let content = null;
    if (state.phase === 'checking') content = null;
    else if (state.phase === 'unavailable') content = unavailablePanel();
    else if (state.phase === 'idle' || state.phase === 'error') content = preloadPanel(gpu);
    else if (state.phase === 'loading') content = loaderPanel();
    else if (state.cameraPaused && !state.frame) content = pausedPanel();
    else if (isLive() && !state.frame.canvas) content = h('div', { class: 'vision-card vision-card--compact' }, h('span', { class: 'vision-spinner' }), h('p', { class: 'preload__sub' }, state.frame.kind === 'camera' ? 'Starting your camera…' : 'Opening the video…'));
    else if (!state.frame) content = emptyPanel();
    overlay.hidden = !content;
    stage.classList.toggle('vision-stage--empty', !state.frame?.canvas);
    replace(overlay, content);
  }

  function unavailablePanel() {
    return h(
      'div',
      { class: 'vision-card' },
      h('div', { class: 'preload__glyph', 'aria-hidden': 'true' }, icon('lock', 24)),
      h('h2', { class: 'preload__title' }, displayName()),
      h('p', { class: 'preload__sub' }, demo.blurb),
      h('p', { class: 'preload__note preload__note--strong' }, 'Not available in this build: the model can’t be downloaded from this site yet.'),
      h(
        'div',
        { class: 'preload__facts' },
        h('p', { class: 'preload__note' }, `Checkpoint ${formatBytes(model.approxBytes, { approx: model.approxBytesIsEstimate })} `, licenseBadgeEl(model)),
        state.reasons.map((r) => h('p', { class: 'preload__note' }, r.text ?? String(r))),
      ),
    );
  }

  function preloadPanel(gpu) {
    const size = formatBytes(modelBytes());
    return h(
      'div',
      { class: 'vision-card' },
      h('div', { class: 'app-glyph', 'aria-hidden': 'true' }, appIcon(demo.id, 64)),
      h('h2', { class: 'preload__title' }, displayName()),
      h('p', { class: 'preload__sub' }, demo.blurb),
      state.phase === 'error' ? h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, `Load failed: ${state.loadError}`)) : null,
      gpu.status === 'ready'
        ? h('button', { class: 'btn btn--primary btn--xl', type: 'button', onclick: load }, icon('bolt', 18), state.phase === 'error' ? `Try again · ${size}` : `Load model · ${size}`)
        : gpu.status === 'probing'
          ? h('button', { class: 'btn btn--primary btn--xl', type: 'button', disabled: true }, 'Detecting GPU…')
          : gpuFallbackPanel(gpu, { compact: true }),
      h('p', { class: 'preload__note' }, 'Downloads once, then stays cached on this device. Images and camera frames never leave the tab.'), hfBadge(state.source), onDeviceNote(model),
    );
  }

  let loaderEls = null;
  function loaderPanel() {
    const bar = h('div', { class: 'progress__bar' });
    const detail = h('p', { class: 'loader__detail' });
    const steps = LOAD_STAGES.map((s) => h('li', { class: 'steps__item' }, h('span', { class: 'steps__dot' }), s.label));
    loaderEls = { bar, detail, steps };
    queueMicrotask(renderLoader);
    return h(
      'div',
      { class: 'vision-card loader' },
      h('h2', { class: 'loader__title' }, `Loading ${state.source.spec.shortName ?? model.shortName}`),
      h('div', { class: 'progress', role: 'progressbar', 'aria-label': 'Load progress', 'aria-valuemin': '0', 'aria-valuemax': '100' }, bar),
      detail,
      h('ol', { class: 'steps' }, steps),
      hfBadge(state.source),
      h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: cancelLoad }, 'Cancel'),
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
      const el = loaderEls.steps[i];
      el.classList.toggle('is-done', f >= 1);
      el.classList.toggle('is-active', s.id === current && f < 1);
    });
    loaderEls.bar.style.width = `${(total * 100).toFixed(1)}%`;
    loaderEls.bar.parentElement.setAttribute('aria-valuenow', String(Math.round(total * 100)));
    loaderEls.detail.textContent = detail;
  }

  function emptyPanel() {
    return h(
      'div',
      { class: 'vision-card vision-card--compact' },
      h('div', { class: 'preload__glyph preload__glyph--ok', 'aria-hidden': 'true' }, icon('check', 22)),
      h('h2', { class: 'preload__title' }, 'Model ready'),
      h('p', { class: 'preload__sub' }, copy.ready),
      h(
        'div',
        { class: 'vision-input-btns' },
        h('button', { class: 'btn btn--primary', type: 'button', onclick: () => fileInput.click() }, uploadIcon(), 'Choose a photo'),
        task.live ? h('button', { class: 'btn btn--ghost', type: 'button', onclick: startCamera }, cameraIcon(), 'Use camera') : null,
      ),
      h('p', { class: 'preload__note' }, 'You can also drop or paste an image anywhere here.'),
    );
  }

  function pausedPanel() {
    return h(
      'div',
      { class: 'vision-card vision-card--compact' },
      h('p', { class: 'preload__sub' }, 'The camera turned off while this tab was in the background.'),
      h('button', { class: 'btn btn--primary', type: 'button', onclick: startCamera }, cameraIcon(), 'Turn it back on'),
    );
  }

  function renderStat() {
    const { ms, fps, detail } = state.stats;
    statEl.hidden = ms == null || state.phase !== 'ready';
    if (statEl.hidden) return;
    statEl.textContent = [formatMs(ms), fps ? (fps < 1 ? '<1 fps' : `${fps.toFixed(fps >= 10 ? 0 : 1)} fps`) : null, detail || null].filter(Boolean).join(' · ');
  }

  function renderLegend() {
    const node = state.phase === 'ready' && state.result && task.legend ? task.legend() : null;
    legendEl.hidden = !node;
    if (node) replace(legendEl, node);
  }

  function renderBusy() {
    const show = state.busy && !isLive() && state.phase === 'ready';
    busyEl.hidden = !show;
    stage.classList.toggle('is-busy', show);
    if (!show) return;
    const p = state.progress;
    replace(
      busyEl,
      h('span', { class: 'vision-spinner' }),
      h('span', null, p?.detail ? `${copy.verb} · ${p.detail}` : `${copy.verb}…`),
      p ? h('span', { class: 'vision-busy__bar' }, h('span', { class: 'vision-busy__fill', style: { width: `${(p.fraction * 100).toFixed(1)}%` } })) : null,
    );
  }

  function renderError() {
    errorEl.hidden = !state.error;
    if (state.error) replace(errorEl, h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, state.error)));
  }

  return () => {
    disposed = true;
    cancelAnimationFrame(raf);
    resizeObserver.disconnect();
    unsubscribeGpu();
    stopLive();
    state.loadAbort?.abort();
    state.stillToken++;
    document.removeEventListener('visibilitychange', onVisibility);
    document.removeEventListener('paste', onPaste);
  };
}
