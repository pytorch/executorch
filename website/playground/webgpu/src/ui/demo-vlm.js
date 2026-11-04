import { getModel } from '../config/models.js';
import { gpuState } from '../gpu/context.js';
import { h, replace } from '../lib/dom.js';
import { formatBytes } from '../lib/format.js';
import { resolveModelSource } from '../runtime/sources.js';
import { VisionLanguageModel } from '../runtime/vlm.js';
import { licenseBadgeEl } from './badges.js';
import { browseMore, detailHeader } from './detail-shell.js';
import { gpuFallbackPanel } from './gpu-fallback.js';
import { appIcon } from './app-icons.js';
import { hfBadge } from './hf-badge.js';
import { icon } from './icons.js';
import { onDeviceNote } from './on-device.js';
import { confirmModelMemory } from './memory-guard.js';

const QUICK = ['Describe this image', 'What text can you read?', 'How many people are there?', 'What’s unusual here?'];
const MAX_NEW_TOKENS = 192;
const LOAD_STAGES = [
  { id: 'runtime', label: 'Start the ExecuTorch runtime', weight: 0.04 },
  { id: 'download', label: 'Download the model', weight: 0.8 },
  { id: 'gpu', label: 'Load the vision tower and decoder onto the GPU', weight: 0.16 },
];

const CAMERA_ICON = '<svg class="icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1Z"/><circle cx="12" cy="13" r="3.5"/></svg>';
const UPLOAD_ICON = '<svg class="icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></svg>';
const PHOTO_ICON = '<svg class="icon" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="9" cy="10" r="1.8"/><path d="m21 16-5-5-9 9"/></svg>';
const svgIcon = (markup) => h('span', { class: 'vlm-ico', html: markup });

// The runtime owns a WebGPU device and ~1 GB of weights it cannot release, so
// one instance outlives the view and is reused when the visitor comes back.
let loaded = null;

export function mountVlm(root, demo) {
  const model = getModel(demo.models[0]);
  const state = {
    source: null, // resolved model source
    phase: loaded ? 'ready' : 'idle', // idle | loading | ready | error | unavailable
    load: { stage: null, fractions: {}, detail: '' },
    loadError: null,
    loadAbort: null,
    image: null, // { url, label, revoke }
    encoding: false,
    encodeError: null,
    turns: [],
    generating: false,
    genAbort: null,
    camera: null, // { stream, error }
    samples: [],
  };
  let disposed = false;
  let imageEl = null;

  // ---------- media column ----------
  const fileInput = h('input', { type: 'file', accept: 'image/*', class: 'vlm-file', tabindex: '-1', 'aria-hidden': 'true', onchange: (e) => {
    const f = e.target.files?.[0];
    if (f) openFile(f);
    e.target.value = '';
  } });
  const stage = h('div', { class: 'vlm-stage', 'aria-label': 'Photo' });
  const tray = h('div', { class: 'vlm-tray', role: 'group', 'aria-label': 'Choose a photo' });
  const media = h(
    'section',
    { class: 'vlm-media' },
    stage,
    tray,
    h('p', { class: 'vlm-privacy' }, icon('lock', 13), 'Your photo stays in this tab. Nothing is uploaded.'),
    fileInput,
  );

  // ---------- chat column ----------
  const statusEl = h('span', { class: 'vlm-status' });
  const newChatBtn = h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: newChat }, icon('refresh', 14), 'New chat');
  const body = h('div', { class: 'vlm-chat__body', 'aria-live': 'polite' });
  const chips = h('div', { class: 'vlm-chips', role: 'group', 'aria-label': 'Quick questions' });
  const input = h('input', { class: 'vlm-input', type: 'text', autocomplete: 'off', 'aria-label': 'Ask about the photo', maxlength: '300' });
  const sendBtn = h('button', { class: 'send-btn', type: 'submit', 'aria-label': 'Send' }, icon('arrowUp', 18));
  const composer = h('form', { class: 'vlm-composer', onsubmit: (e) => (e.preventDefault(), onSend()) }, input, sendBtn);
  const chat = h(
    'section',
    { class: 'vlm-chat', 'aria-label': 'Conversation about the photo' },
    h(
      'header',
      { class: 'vlm-chat__head' },
      h('span', { class: 'vlm-avatar', 'aria-hidden': 'true' }, icon('sparkle', 16)),
      h('div', { class: 'vlm-chat__who' }, h('strong', null, model.shortName), statusEl),
      newChatBtn,
    ),
    body,
    h('div', { class: 'vlm-chat__foot' }, chips, composer),
  );

  replace(root, h('div', { class: 'page detail vlm-page' }, detailHeader(demo), h('div', { class: 'vlm' }, media, chat), browseMore(demo.id)));

  // ---------- image input ----------
  ['dragenter', 'dragover'].forEach((ev) =>
    media.addEventListener(ev, (e) => {
      if (![...(e.dataTransfer?.types ?? [])].includes('Files')) return;
      e.preventDefault();
      stage.classList.add('is-hot');
    }),
  );
  media.addEventListener('dragleave', (e) => {
    if (!media.contains(e.relatedTarget)) stage.classList.remove('is-hot');
  });
  media.addEventListener('drop', (e) => {
    e.preventDefault();
    stage.classList.remove('is-hot');
    const f = [...(e.dataTransfer?.files ?? [])].find((x) => x.type.startsWith('image/'));
    if (f) openFile(f);
  });
  const onPaste = (e) => {
    const item = [...(e.clipboardData?.items ?? [])].find((i) => i.kind === 'file' && i.type.startsWith('image/'));
    if (!item) return;
    e.preventDefault();
    openFile(item.getAsFile(), 'Pasted image');
  };
  document.addEventListener('paste', onPaste);

  function openFile(file, label = file.name || 'Your photo') {
    const url = URL.createObjectURL(file);
    showImage({ url, label, revoke: true });
  }

  function showImage(img) {
    stopCamera();
    state.genAbort?.abort();
    if (state.image?.revoke) URL.revokeObjectURL(state.image.url);
    state.image = img;
    state.turns = [];
    state.encodeError = null;
    imageEl = new Image();
    imageEl.decoding = 'async';
    imageEl.src = img.url;
    imageEl
      .decode()
      .then(() => {
        if (disposed || state.image !== img) return;
        renderAll();
        if (loaded) encode();
      })
      .catch(() => {
        if (state.image !== img) return;
        state.image = null;
        state.encodeError = 'That file could not be read as an image.';
        renderAll();
      });
    renderAll();
  }

  async function encode() {
    const img = state.image;
    if (!img || !imageEl?.complete || !loaded) return;
    state.encoding = true;
    state.encodeError = null;
    renderAll();
    try {
      const r = await loaded.setImage(imageEl);
      if (r.superseded || state.image !== img) return;
    } catch (err) {
      if (state.image === img) state.encodeError = `The vision encoder failed: ${err.message}`;
    } finally {
      if (state.image === img) state.encoding = false;
      if (!disposed) renderAll();
    }
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      state.camera = { error: 'This browser does not expose a camera to web pages.' };
      renderAll();
      return;
    }
    state.camera = { stream: null, error: null };
    renderAll();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false });
      if (disposed || !state.camera) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      state.camera.stream = stream;
    } catch (err) {
      if (!state.camera) return;
      state.camera.error = err.name === 'NotAllowedError' ? 'Camera permission was denied. You can allow it from the address bar.' : `No camera available (${err.message}).`;
    }
    renderAll();
  }

  function stopCamera() {
    state.camera?.stream?.getTracks().forEach((t) => t.stop());
    state.camera = null;
  }

  function snapshot(video) {
    if (!video.videoWidth) return;
    const c = document.createElement('canvas');
    c.width = video.videoWidth;
    c.height = video.videoHeight;
    c.getContext('2d').drawImage(video, 0, 0);
    c.toBlob((blob) => blob && showImage({ url: URL.createObjectURL(blob), label: 'Camera snapshot', revoke: true }), 'image/jpeg', 0.92);
  }

  // ---------- model ----------
  async function resolveSource() {
    state.source = await resolveModelSource(model);
    if (disposed) return;
    if (state.source.kind !== 'unavailable') {
      state.samples = (state.source.shared.vlmSamples ?? []).map((s) => ({ url: state.source.url(s.path), label: s.label }));
    } else if (!loaded) {
      state.phase = 'unavailable';
    }
    renderAll();
  }

  async function load() {
    if (state.phase === 'loading' || !state.source || state.source.kind === 'unavailable') return;
    if (!(await confirmModelMemory(model))) return;
    const controller = new AbortController();
    Object.assign(state, { phase: 'loading', loadError: null, loadAbort: controller, load: { stage: 'runtime', fractions: {}, detail: '' } });
    loaderEls = null;
    renderAll();
    try {
      const vlm = await VisionLanguageModel.load(state.source, {
        signal: controller.signal,
        onProgress: (p) => {
          state.load.stage = p.stage;
          state.load.fractions[p.stage] = p.fraction;
          if (p.stage === 'download') {
            state.load.detail = p.fromCache && p.fraction >= 1 ? 'Loaded from this device’s cache' : `${formatBytes(p.loaded)} of ${formatBytes(p.total)}`;
          } else if (p.stage === 'gpu') state.load.detail = 'Building GPU programs…';
          renderLoader();
        },
      });
      loaded = vlm;
      if (disposed) return;
      state.phase = 'ready';
      renderAll();
      if (state.image) encode();
      else input.focus();
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

  // ---------- conversation ----------
  function newChat() {
    state.genAbort?.abort();
    state.turns = [];
    loaded?.resetConversation();
    renderAll();
  }

  function onSend(text = input.value) {
    if (state.generating) {
      state.genAbort?.abort();
      return;
    }
    const q = text.trim();
    if (!q || !canAsk()) return;
    input.value = '';
    ask(q);
  }

  function canAsk() {
    return state.phase === 'ready' && !!state.image && !state.camera && !state.encodeError && imageEl?.complete;
  }

  async function ask(question) {
    const turn = { question, answer: '', stage: state.encoding ? 'vision' : 'prefill', streaming: true };
    state.turns.push(turn);
    const controller = new AbortController();
    Object.assign(state, { generating: true, genAbort: controller });
    renderAll();
    try {
      const r = await loaded.ask(question, {
        signal: controller.signal,
        maxNewTokens: MAX_NEW_TOKENS,
        onStage: (s) => {
          turn.stage = s;
          paintTurn(turn);
        },
        onText: (t) => {
          turn.answer = t;
          turn.stage = 'decode';
          paintTurn(turn);
        },
      });
      turn.answer = r.text;
      turn.result = r;
      const notes = [];
      if (r.stop === 'aborted') notes.push('Stopped.');
      else if (r.stop === 'length') notes.push(`Cut off at ${r.newTokens} tokens.`);
      if (!r.text && r.stop === 'eos') notes.push('The model ended its answer without saying anything. Try rephrasing.');
      if (r.restarted) notes.push('Earlier turns no longer fit in the model’s context, so this answer only sees the photo and this question.');
      turn.note = notes.join(' ') || null;
    } catch (err) {
      turn.error = err.name === 'AbortError' ? null : err.message;
      turn.note = err.name === 'AbortError' ? 'Stopped before the answer started.' : null;
    } finally {
      turn.streaming = false;
      if (state.genAbort === controller) Object.assign(state, { generating: false, genAbort: null });
      paintTurn(turn);
      if (!disposed) renderAll();
    }
  }

  // ---------- rendering ----------
  function renderAll() {
    if (disposed) return;
    renderMedia();
    renderStatus();
    renderBody();
    renderChips();
    renderComposer();
  }

  // The stage is rebuilt only when what it shows changes, so the photo does not
  // flash or replay its entrance on every status update.
  let stageKey = null;
  let stageEls = null;
  function renderMedia() {
    stage.classList.toggle('is-busy', state.encoding);
    const cam = state.camera;
    const key = cam ? `cam:${!!cam.stream}:${cam.error}` : state.image && imageEl ? `img:${state.image.url}` : `empty:${state.encodeError}`;
    if (key !== stageKey) {
      stageKey = key;
      stageEls = null;
      if (cam) replace(stage, cameraView());
      else if (state.image && imageEl) {
        stageEls = {
          scan: h('div', { class: 'vlm-scan', 'aria-hidden': 'true' }),
          caption: h('div', { class: 'vlm-stage__caption' }),
        };
        replace(
          stage,
          h('img', { class: 'vlm-stage__backdrop', src: state.image.url, alt: '', 'aria-hidden': 'true' }),
          h('img', { class: 'vlm-photo', src: state.image.url, alt: state.image.label }),
          stageEls.scan,
          stageEls.caption,
        );
      } else replace(stage, dropZone());
    }
    if (stageEls) {
      stageEls.scan.hidden = !state.encoding;
      replace(stageEls.caption, state.encoding ? h('span', { class: 'vlm-dot' }) : null, state.encoding ? 'Looking at the photo…' : state.image.label);
    }
    renderTray();
  }

  function dropZone() {
    return h(
      'div',
      { class: 'vlm-drop' },
      h('div', { class: 'vlm-drop__glyph' }, svgIcon(PHOTO_ICON)),
      h('p', { class: 'vlm-drop__title' }, 'Drop a photo here'),
      h('p', { class: 'vlm-drop__sub' }, 'or paste one, pick a file, or take a snapshot'),
      h(
        'div',
        { class: 'vlm-drop__actions' },
        h('button', { class: 'btn btn--primary btn--sm', type: 'button', onclick: () => fileInput.click() }, svgIcon(UPLOAD_ICON), 'Choose a photo'),
        h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: startCamera }, svgIcon(CAMERA_ICON), 'Camera'),
      ),
      state.encodeError ? h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, state.encodeError)) : null,
    );
  }

  function cameraView() {
    const cam = state.camera;
    if (cam.error) {
      return h(
        'div',
        { class: 'vlm-drop' },
        h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, cam.error)),
        h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => (stopCamera(), renderAll()) }, 'Back'),
      );
    }
    const video = h('video', { class: 'vlm-video', autoplay: true, playsinline: true, muted: true });
    if (cam.stream) {
      video.srcObject = cam.stream;
      video.play().catch(() => {});
    }
    return h(
      'div',
      { class: 'vlm-camera' },
      video,
      cam.stream ? null : h('div', { class: 'vlm-camera__wait' }, 'Waiting for the camera…'),
      h(
        'div',
        { class: 'vlm-camera__bar' },
        h('button', { class: 'vlm-camera__cancel', type: 'button', 'aria-label': 'Close camera', onclick: () => (stopCamera(), renderAll()) }, icon('close', 18)),
        h('button', { class: 'vlm-shutter', type: 'button', 'aria-label': 'Take snapshot', disabled: !cam.stream, onclick: () => snapshot(video) }),
        h('span', { class: 'vlm-camera__spacer' }),
      ),
    );
  }

  function renderTray() {
    const active = state.camera ? null : state.image?.url;
    replace(
      tray,
      state.samples.map((s) =>
        h(
          'button',
          {
            class: s.url === active ? 'vlm-thumb is-active' : 'vlm-thumb',
            type: 'button',
            title: s.label,
            'aria-label': `Use the ${s.label} sample`,
            'aria-pressed': String(s.url === active),
            onclick: () => s.url !== active && showImage({ url: s.url, label: s.label }),
          },
          h('img', { src: s.url, alt: '', loading: 'lazy' }),
        ),
      ),
      h('button', { class: 'vlm-tray__btn', type: 'button', onclick: () => fileInput.click() }, svgIcon(UPLOAD_ICON), h('span', null, 'Upload')),
      h('button', { class: state.camera ? 'vlm-tray__btn is-active' : 'vlm-tray__btn', type: 'button', onclick: () => (state.camera ? (stopCamera(), renderAll()) : startCamera()) }, svgIcon(CAMERA_ICON), h('span', null, 'Camera')),
    );
  }

  function renderStatus() {
    let tone = 'idle';
    let text = 'Not loaded';
    if (state.phase === 'loading') (tone = 'busy'), (text = 'Loading…');
    else if (state.phase === 'error') (tone = 'bad'), (text = 'Load failed');
    else if (state.phase === 'unavailable') text = 'Not in this build';
    else if (state.phase === 'ready') {
      if (state.encoding) (tone = 'busy'), (text = 'Looking at the photo…');
      else if (state.generating) (tone = 'busy'), (text = 'Answering…');
      else (tone = 'ready'), (text = 'On your GPU');
    }
    statusEl.className = `vlm-status vlm-status--${tone}`;
    statusEl.textContent = text;
    newChatBtn.hidden = !(state.phase === 'ready' && state.turns.length);
  }

  // Turns keep their elements across renders so finished answers never re-animate.
  let thread = null;
  function renderBody() {
    if (state.phase === 'ready' && state.turns.length) {
      if (thread?.turns !== state.turns) thread = { turns: state.turns, el: h('div', { class: 'vlm-thread' }, contextMarker()) };
      for (const t of state.turns) if (!t.el) thread.el.append(turnEl(t));
      if (body.firstChild !== thread.el) replace(body, thread.el);
    } else {
      let content;
      if (state.phase === 'unavailable') content = unavailablePanel();
      else if (state.phase === 'loading') content = loaderEls?.panel ?? loaderPanel();
      else if (state.phase !== 'ready') content = preloadPanel();
      else content = readyEmpty();
      replace(body, content);
    }
    body.scrollTop = body.scrollHeight;
  }

  function preloadPanel() {
    const gpu = gpuState.get();
    return h(
      'div',
      { class: 'preload vlm-hero' },
      h('div', { class: 'app-glyph', 'aria-hidden': 'true' }, appIcon(demo.id, 64)),
      h('h2', { class: 'preload__title' }, 'Ask questions about any photo'),
      h('p', { class: 'preload__sub' }, `${model.name}: a vision encoder and a small language model, running on your GPU with ExecuTorch.`),
      state.phase === 'error' ? h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, `Load failed: ${state.loadError}`)) : null,
      gpu.status === 'unavailable' || gpu.status === 'lost'
        ? gpuFallbackPanel(gpu, { compact: true })
        : h(
            'button',
            { class: 'btn btn--primary btn--xl', type: 'button', disabled: !state.source || gpu.status === 'probing', onclick: load },
            icon('bolt', 18),
            state.phase === 'error' ? 'Try again' : `Load model · ${formatBytes(model.approxBytes, { approx: model.approxBytesIsEstimate })}`,
          ),
      h('p', { class: 'preload__note' }, 'One-time download, then cached on this device. Your photos and questions never leave the tab.'), hfBadge(state.source), onDeviceNote(model),
    );
  }

  function unavailablePanel() {
    const reasons = state.source?.reasons ?? [];
    return h(
      'div',
      { class: 'preload vlm-hero' },
      h('div', { class: 'preload__glyph', 'aria-hidden': 'true' }, icon('lock', 24)),
      h('h2', { class: 'preload__title' }, 'Not available in this build'),
      h('p', { class: 'preload__sub' }, `${model.name} can’t be downloaded from this site yet, so no model runs here and no answers are generated.`),
      licenseBadgeEl(model),
      reasons.length ? h('ul', { class: 'vlm-reasons' }, reasons.map((r) => h('li', null, r.text ?? String(r)))) : null,
    );
  }

  let loaderEls = null;
  function loaderPanel() {
    const bar = h('div', { class: 'progress__bar' });
    const detail = h('p', { class: 'loader__detail' });
    const steps = LOAD_STAGES.map((s) => h('li', { class: 'steps__item' }, h('span', { class: 'steps__dot' }), s.label));
    const note = h('p', { class: 'preload__note' });
    const panel = h(
      'div',
      { class: 'loader' },
      h('h2', { class: 'loader__title' }, `Loading ${model.shortName}`),
      h('div', { class: 'progress', role: 'progressbar', 'aria-label': 'Load progress', 'aria-valuemin': '0', 'aria-valuemax': '100' }, bar),
      detail,
      h('ol', { class: 'steps' }, steps),
      note,
      h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: cancelLoad }, 'Cancel'),
    );
    loaderEls = { panel, bar, detail, steps, note };
    queueMicrotask(renderLoader);
    return panel;
  }

  function renderLoader() {
    if (!loaderEls || state.phase !== 'loading') return;
    const { fractions, stage: current, detail } = state.load;
    const idx = LOAD_STAGES.findIndex((s) => s.id === current);
    let total = 0;
    LOAD_STAGES.forEach((s, i) => {
      const f = i < idx ? 1 : i === idx ? (fractions[s.id] ?? 0) : 0;
      total += s.weight * f;
      loaderEls.steps[i].classList.toggle('is-done', i < idx || (i === idx && f >= 1));
      loaderEls.steps[i].classList.toggle('is-active', i === idx && f < 1);
    });
    loaderEls.bar.style.width = `${(total * 100).toFixed(1)}%`;
    loaderEls.bar.parentElement.setAttribute('aria-valuenow', String(Math.round(total * 100)));
    loaderEls.detail.textContent = detail;
    loaderEls.note.textContent = state.image ? 'Your photo is ready. The conversation opens as soon as the model is.' : 'Pick a photo while you wait.';
  }

  function readyEmpty() {
    return h(
      'div',
      { class: 'ready-empty' },
      h('div', { class: 'preload__glyph preload__glyph--ok', 'aria-hidden': 'true' }, icon('check', 24)),
      h('h2', { class: 'preload__title' }, state.camera ? 'Frame your shot' : state.image ? 'What would you like to know?' : 'Model ready'),
      h(
        'p',
        { class: 'preload__sub' },
        state.camera
          ? 'Press the shutter, then ask about the snapshot.'
          : state.image
            ? 'Ask anything about the photo, then keep the conversation going with follow-ups.'
            : 'Pick a sample, upload a photo, paste one, or take a camera snapshot.',
      ),
      state.encodeError ? h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, state.encodeError)) : null,
    );
  }

  function contextMarker() {
    return h(
      'div',
      { class: 'vlm-context' },
      state.image ? h('img', { class: 'vlm-context__thumb', src: state.image.url, alt: '' }) : null,
      h('span', null, `${state.image?.label ?? 'Photo'} · new conversation`),
    );
  }

  function turnEl(turn) {
    if (turn.el) return turn.el;
    const answer = h('div', { class: 'vlm-answer' });
    const el = h(
      'div',
      { class: 'vlm-turn' },
      h('div', { class: 'vlm-q' }, turn.question),
      h('div', { class: 'vlm-a' }, h('span', { class: 'vlm-avatar vlm-avatar--sm', 'aria-hidden': 'true' }, icon('sparkle', 13)), answer),
    );
    turn.answerEl = answer;
    turn.el = el;
    paintTurn(turn);
    return el;
  }

  function paintTurn(turn) {
    const el = turn.answerEl;
    if (!el || disposed) return;
    const waiting = turn.streaming && !turn.answer;
    replace(
      el,
      waiting
        ? h('div', { class: 'vlm-typing' }, h('span', { class: 'vlm-typing__dots' }, h('i'), h('i'), h('i')), turn.stage === 'vision' ? 'Looking at the photo…' : 'Thinking…')
        : turn.answer
          ? h('p', { class: 'vlm-answer__text' }, turn.answer, turn.streaming ? h('span', { class: 'caret' }) : null)
          : null,
      turn.error ? h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, turn.error)) : null,
      !turn.streaming && (turn.result || turn.note) ? h('div', { class: 'vlm-stat' }, statLine(turn)) : null,
    );
    if (turn.streaming) body.scrollTop = body.scrollHeight;
  }

  function statLine(turn) {
    const r = turn.result;
    const parts = [];
    if (r?.newTokens) {
      parts.push(`first word ${seconds(r.ttftMs)}`);
      if (r.tps) parts.push(`${r.tps >= 10 ? Math.round(r.tps) : r.tps.toFixed(1)} tok/s`);
    }
    if (turn.note) parts.push(turn.note);
    return parts.join(' · ');
  }

  function renderChips() {
    const show = state.phase === 'ready' && state.image && !state.camera && !state.encodeError;
    chips.hidden = !show;
    if (!show) return;
    const disabled = state.generating || !canAsk();
    replace(chips, QUICK.map((q) => h('button', { class: 'suggest vlm-chip', type: 'button', disabled, onclick: () => onSend(q) }, q)));
  }

  function renderComposer() {
    const ready = state.phase === 'ready';
    input.disabled = !ready || !state.image || !!state.camera;
    input.placeholder = state.phase === 'unavailable' ? 'Not available in this build' : !ready ? 'Load the model to start' : state.camera ? 'Take a snapshot to ask about it' : !state.image ? 'Pick a photo first' : state.turns.length ? 'Ask a follow-up…' : 'Ask about this photo…';
    sendBtn.disabled = !state.generating && !canAsk();
    replace(sendBtn, icon(state.generating ? 'stop' : 'arrowUp', 18));
    sendBtn.setAttribute('aria-label', state.generating ? 'Stop' : 'Send');
    sendBtn.classList.toggle('send-btn--stop', state.generating);
  }

  const unsubscribeGpu = gpuState.subscribe(() => {
    if (state.phase === 'idle' || state.phase === 'error') renderBody();
  });

  renderAll();
  resolveSource();

  return () => {
    disposed = true;
    unsubscribeGpu();
    document.removeEventListener('paste', onPaste);
    state.loadAbort?.abort();
    state.genAbort?.abort();
    stopCamera();
    if (state.image?.revoke) URL.revokeObjectURL(state.image.url);
  };
}

function seconds(ms) {
  return ms >= 10000 ? `${Math.round(ms / 1000)} s` : `${(ms / 1000).toFixed(1)} s`;
}
