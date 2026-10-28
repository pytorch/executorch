import { decodeAudioFile, MAX_FILE_SECONDS } from '../audio/decode.js';
import { MicError, micSupport, recordMic } from '../audio/mic.js';
import { encodeWav, formatClock, SPEECH_RATE } from '../audio/pcm.js';
import { createWaveform } from '../audio/waveform.js';
import { getModel } from '../config/models.js';
import { gpuState } from '../gpu/context.js';
import { h, replace, svg } from '../lib/dom.js';
import { formatBytes, formatMs } from '../lib/format.js';
import { isWhisperLoaded, loadWhisper, WHISPER_LANGUAGES, WHISPER_MODEL_ID } from '../runtime/whisper.js';
import { resolveModelSource } from '../runtime/sources.js';
import { licenseBadgeEl } from './badges.js';
import { browseMore, detailHeader } from './detail-shell.js';
import { gpuFallbackPanel } from './gpu-fallback.js';
import { appIcon } from './app-icons.js';
import { hfBadge } from './hf-badge.js';
import { icon } from './icons.js';
import { onDeviceNote } from './on-device.js';
import { confirmModelMemory } from './memory-guard.js';

const MAX_REC_SECONDS = 300;
const MIN_CLIP_SECONDS = 0.4;
const MODE_KEY = 'etwg-stt-mode';
const LOAD_STAGES = [
  { id: 'runtime', label: 'Start the ExecuTorch runtime', weight: 0.04 },
  { id: 'download', label: 'Download the model', weight: 0.84 },
  { id: 'gpu', label: 'Upload graphs to the GPU', weight: 0.12 },
];

const glyph = (paths, size = 18) =>
  svg(`<svg class="icon" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`);
const micIcon = (s) => glyph('<rect x="9" y="2.5" width="6" height="12" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7"/>', s);
const uploadIcon = (s) => glyph('<path d="M12 15V4M7.5 8.5 12 4l4.5 4.5"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>', s);
const fileIcon = (s) => glyph('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z"/><path d="M14 3v5h5M10 17v-4l4-1v4"/><circle cx="9" cy="17" r="1.2"/><circle cx="13" cy="16" r="1.2"/>', s);

const loadMode = () => {
  try {
    return localStorage.getItem(MODE_KEY) === 'hold' ? 'hold' : 'toggle';
  } catch {
    return 'toggle';
  }
};

export function mountStt(root, demo) {
  const model = getModel(WHISPER_MODEL_ID);
  const state = {
    phase: isWhisperLoaded() ? 'ready' : 'checking', // checking | unavailable | idle | loading | ready | error
    reasons: [],
    load: { stage: null, fractions: {}, loaded: 0, total: 0, detail: '' },
    loadError: null,
    gpu: gpuState.get(),
    mode: loadMode(),
    language: 'en',
    rec: 'idle', // idle | starting | recording | transcribing
    notice: null, // { tone: 'error' | 'info', text }
    last: null,
    cards: [],
  };
  let asr = null;
  let disposed = false;
  let loadAbort = null;
  let txAbort = null;
  let mic = null;
  let stopWanted = false;
  let startToken = 0;
  let level = 0;
  let raf = 0;
  let dragDepth = 0;
  const player = new Audio();
  let playing = null;

  // ---------- persistent elements ----------
  const langSelect = h(
    'select',
    { class: 'select stt-lang', 'aria-label': 'Spoken language', onchange: (e) => (state.language = e.target.value) },
    WHISPER_LANGUAGES.map((l) => h('option', { value: l.code }, l.label)),
  );
  const stage = h('section', { class: 'stt-stage', 'aria-label': 'Recorder' });
  const list = h('div', { class: 'stt-list', 'aria-live': 'polite' });
  const countEl = h('span', { class: 'stt-side__count' });
  const clearBtn = h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => clearCards() }, 'Clear');
  const fileInput = h('input', { class: 'stt-file', type: 'file', accept: 'audio/*,.wav,.mp3,.m4a,.ogg,.oga,.webm,.flac', onchange: (e) => onFiles(e.target.files) });

  const recBtn = h('button', { class: 'stt-rec', type: 'button' });
  const orb = h('div', { class: 'stt-orb' }, h('span', { class: 'stt-orb__halo' }), h('span', { class: 'stt-orb__ring' }), recBtn);
  const clock = h('div', { class: 'stt-clock', 'aria-hidden': 'true' }, '00:00.0');
  const hint = h('p', { class: 'stt-hint' });
  const waveCanvas = h('canvas', { class: 'stt-wave', 'aria-hidden': 'true' });
  const wave = createWaveform(waveCanvas);
  // The in-progress highlight is a CSS animation, not canvas redraws, so it
  // costs the GPU nothing while the model is using it.
  const sweep = h('span', { class: 'stt-wave__sweep', hidden: true }, h('span', { class: 'stt-wave__glint' }));
  const waveWrap = h('div', { class: 'stt-wave-wrap' }, waveCanvas, sweep);
  const setSweep = (lo, hi) => {
    sweep.hidden = lo == null;
    if (lo != null) Object.assign(sweep.style, { left: `${lo * 100}%`, width: `${Math.max(0.04, hi - lo) * 100}%` });
  };
  const noticeSlot = h('div', { class: 'stt-notice-slot' });
  const statEl = h('p', { class: 'stt-stat' });
  const modeBtns = {
    toggle: h('button', { class: 'segmented__btn', type: 'button', onclick: () => setMode('toggle') }, 'Tap to record'),
    hold: h('button', { class: 'segmented__btn', type: 'button', onclick: () => setMode('hold') }, 'Hold to talk'),
  };

  replace(
    root,
    h(
      'div',
      { class: 'page detail stt-page' },
      detailHeader(demo, { controls: h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Language'), langSelect) }),
      h(
        'div',
        { class: 'truth' },
        h(
          'div',
          { class: 'truth__item truth__item--ok' },
          h('span', { class: 'truth__tag' }, 'Runs on your GPU'),
          h('span', null, 'Whisper base runs through ExecuTorch on WebGPU, log-mel front end included. Audio never leaves this tab.'),
        ),
      ),
      h(
        'div',
        { class: 'stt-layout' },
        stage,
        h(
          'section',
          { class: 'stt-side', 'aria-label': 'Transcripts' },
          h('div', { class: 'stt-side__head' }, h('h2', { class: 'stt-side__title' }, 'Transcripts', countEl), clearBtn),
          list,
        ),
      ),
      browseMore(demo.id),
    ),
  );

  // ---------- events ----------
  recBtn.addEventListener('click', () => {
    if (state.mode === 'toggle') toggleRecording();
  });
  recBtn.addEventListener('pointerdown', (e) => {
    if (state.mode !== 'hold' || e.button !== 0) return;
    e.preventDefault();
    recBtn.setPointerCapture?.(e.pointerId);
    startRecording();
  });
  const release = () => {
    if (state.mode === 'hold') stopRecording();
  };
  recBtn.addEventListener('pointerup', release);
  recBtn.addEventListener('pointercancel', release);
  recBtn.addEventListener('contextmenu', (e) => state.mode === 'hold' && e.preventDefault());

  const typing = (el) => el && (/^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName) || el.isContentEditable || (el.tagName === 'BUTTON' && el !== recBtn) || el.tagName === 'A');
  const onKeyDown = (e) => {
    if (state.phase !== 'ready') return;
    if (e.key === 'Escape' && (state.rec === 'recording' || state.rec === 'starting')) {
      cancelRecording();
      return;
    }
    if (e.code !== 'Space' || typing(e.target)) return;
    e.preventDefault();
    if (e.repeat) return;
    if (state.mode === 'hold') startRecording();
    else toggleRecording();
  };
  const onKeyUp = (e) => {
    if (e.code !== 'Space' || state.phase !== 'ready' || typing(e.target)) return;
    e.preventDefault();
    if (state.mode === 'hold') stopRecording();
  };
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);

  stage.addEventListener('dragenter', (e) => {
    if (!canAccept(e)) return;
    e.preventDefault();
    dragDepth++;
    stage.classList.add('stt-stage--drag');
  });
  stage.addEventListener('dragover', (e) => {
    if (!canAccept(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  });
  stage.addEventListener('dragleave', () => {
    if (--dragDepth <= 0) {
      dragDepth = 0;
      stage.classList.remove('stt-stage--drag');
    }
  });
  stage.addEventListener('drop', (e) => {
    if (!canAccept(e)) return;
    e.preventDefault();
    dragDepth = 0;
    stage.classList.remove('stt-stage--drag');
    onFiles(e.dataTransfer.files);
  });
  const canAccept = (e) => state.phase === 'ready' && state.rec === 'idle' && [...(e.dataTransfer?.types ?? [])].includes('Files');

  player.addEventListener('ended', () => setPlaying(null));
  player.addEventListener('pause', () => setPlaying(null));

  const unsubscribeGpu = gpuState.subscribe((s) => {
    const changed = s.status !== state.gpu.status;
    state.gpu = s;
    if (changed && state.phase !== 'ready') renderStage();
  });

  // ---------- model ----------
  (async () => {
    if (state.phase === 'ready') {
      asr = await loadWhisper();
      return renderAll();
    }
    renderAll();
    const source = await resolveModelSource(model);
    if (disposed) return;
    state.source = source;
    if (source.kind === 'unavailable') Object.assign(state, { phase: 'unavailable', reasons: source.reasons });
    else state.phase = 'idle';
    renderAll();
  })();

  async function load() {
    if (state.phase === 'loading') return;
    if (!(await confirmModelMemory(model))) return;
    loadAbort = new AbortController();
    Object.assign(state, { phase: 'loading', loadError: null, load: { stage: 'runtime', fractions: {}, loaded: 0, total: 0, detail: '' } });
    renderStage();
    try {
      asr = await loadWhisper({
        signal: loadAbort.signal,
        onProgress: (p) => {
          state.load.stage = p.stage;
          state.load.fractions[p.stage] = p.fraction;
          if (p.stage === 'download') Object.assign(state.load, { loaded: p.loaded, total: p.total });
          state.load.detail = p.detail ?? '';
          updateLoader();
        },
      });
      if (disposed) return;
      state.phase = 'ready';
      renderAll();
    } catch (err) {
      if (disposed || err.name === 'AbortError') return;
      Object.assign(state, { phase: 'error', loadError: err.message });
      renderStage();
    } finally {
      loadAbort = null;
    }
  }

  function cancelLoad() {
    loadAbort?.abort();
    state.phase = 'idle';
    renderStage();
  }

  // ---------- recording ----------
  function setMode(mode) {
    state.mode = mode;
    try {
      localStorage.setItem(MODE_KEY, mode);
    } catch {
      // Non-persistent is fine.
    }
    updateControls();
  }

  function toggleRecording() {
    if (state.rec === 'recording') stopRecording();
    else if (state.rec === 'idle') startRecording();
  }

  async function startRecording() {
    if (state.phase !== 'ready' || state.rec !== 'idle') return;
    stopWanted = false;
    const token = ++startToken;
    state.rec = 'starting';
    state.notice = null;
    level = 0;
    updateControls();
    renderNotice();
    try {
      const handle = await recordMic({
        maxSeconds: MAX_REC_SECONDS,
        onLevel: ({ rms }) => {
          level = Math.max(Math.min(1, Math.sqrt(rms) * 2.4), level * 0.8);
          wave.push(rms);
        },
        onAutoStop: () => {
          state.notice = { tone: 'info', text: `Recording stops at ${MAX_REC_SECONDS / 60} minutes.` };
          stopRecording();
        },
      });
      if (disposed || token !== startToken) return handle.cancel();
      mic = handle;
      state.rec = 'recording';
      wave.clear();
      updateControls();
      tick();
      if (stopWanted) stopRecording();
    } catch (err) {
      if (disposed || token !== startToken) return;
      state.rec = 'idle';
      state.notice = { tone: 'error', text: err instanceof MicError ? err.message : `Could not start the microphone: ${err.message}` };
      updateControls();
      renderNotice();
    }
  }

  function tick() {
    cancelAnimationFrame(raf);
    const step = () => {
      if (!mic || state.rec !== 'recording') return;
      clock.textContent = formatClock(mic.seconds, { tenths: true });
      level *= 0.94;
      orb.style.setProperty('--lvl', level.toFixed(3));
      raf = requestAnimationFrame(step);
    };
    step();
  }

  async function stopRecording() {
    if (state.rec === 'starting') {
      stopWanted = true;
      return;
    }
    if (state.rec !== 'recording' || !mic) return;
    const handle = mic;
    mic = null;
    cancelAnimationFrame(raf);
    orb.style.setProperty('--lvl', '0');
    state.rec = 'transcribing';
    updateControls();
    const samples = await handle.stop();
    if (disposed) return;
    if (samples.length < MIN_CLIP_SECONDS * SPEECH_RATE) {
      state.rec = 'idle';
      state.notice = { tone: 'info', text: state.mode === 'hold' ? 'That was very short. Keep holding while you speak.' : 'That was very short. Tap, speak, then tap again.' };
      wave.clear();
      updateControls();
      renderNotice();
      return;
    }
    await transcribe(samples, { source: 'mic', label: 'Microphone' });
  }

  function cancelRecording() {
    const handle = mic;
    mic = null;
    stopWanted = false;
    startToken++;
    cancelAnimationFrame(raf);
    orb.style.setProperty('--lvl', '0');
    state.rec = 'idle';
    handle?.cancel();
    wave.clear();
    clock.textContent = '00:00.0';
    updateControls();
  }

  // ---------- files ----------
  async function onFiles(files) {
    const file = files?.[0];
    fileInput.value = '';
    if (!file || state.phase !== 'ready' || state.rec !== 'idle') return;
    state.rec = 'transcribing';
    state.notice = { tone: 'info', text: `Decoding ${file.name}…` };
    updateControls();
    renderNotice();
    let samples;
    try {
      samples = await decodeAudioFile(file);
    } catch (err) {
      state.rec = 'idle';
      state.notice = { tone: 'error', text: err.message };
      updateControls();
      renderNotice();
      return;
    }
    if (disposed) return;
    state.notice = null;
    renderNotice();
    if (samples.length < MIN_CLIP_SECONDS * SPEECH_RATE) {
      state.rec = 'idle';
      state.notice = { tone: 'info', text: 'That file has almost no audio in it.' };
      updateControls();
      renderNotice();
      return;
    }
    await transcribe(samples, { source: 'file', label: file.name });
  }

  // ---------- transcription ----------
  async function transcribe(samples, { source, label }) {
    const card = {
      source,
      label,
      at: new Date(),
      audioSec: samples.length / SPEECH_RATE,
      text: '',
      status: 'running',
      progress: 0,
      segments: [],
      url: URL.createObjectURL(new Blob([encodeWav(samples)], { type: 'audio/wav' })),
      el: null,
    };
    state.cards.unshift(card);
    state.rec = 'transcribing';
    clock.textContent = formatClock(card.audioSec, { tenths: true });
    wave.showClip(samples);
    wave.setProgress(0);
    card.windows = Math.max(1, Math.ceil(card.audioSec / asr.info.windowSec));
    setSweep(0, 1 / card.windows);
    renderList();
    updateControls();
    txAbort = new AbortController();
    const signal = txAbort.signal;
    try {
      const r = await asr.transcribe(samples, {
        language: state.language,
        signal,
        onPartial: (text, { window, windows }) => {
          card.text = text;
          card.progress = window / windows;
          card.windows = windows;
          wave.setProgress(card.progress);
          setSweep(card.progress, Math.min(1, (window + 1) / windows));
          updateCard(card);
          updateHint();
        },
      });
      Object.assign(card, { text: r.text, segments: r.segments, ms: r.ms, status: signal.aborted ? 'stopped' : 'done' });
      state.last = { audioSec: r.audioSec, ms: r.ms };
    } catch (err) {
      if (err.name === 'AbortError') card.status = 'stopped';
      else Object.assign(card, { status: 'error', error: err.message });
    } finally {
      txAbort = null;
      if (!disposed) {
        state.rec = 'idle';
        setSweep(null);
        wave.setProgress(1);
        updateCard(card);
        updateControls();
      }
    }
  }

  function stopTranscription() {
    txAbort?.abort();
    updateHint();
  }

  function clearCards() {
    if (state.rec === 'transcribing') return;
    player.pause();
    for (const c of state.cards) URL.revokeObjectURL(c.url);
    state.cards = [];
    renderList();
  }

  function setPlaying(card) {
    const prev = playing;
    playing = card;
    if (prev?.el) updateCard(prev);
    if (card?.el) updateCard(card);
  }

  function togglePlay(card) {
    if (playing === card) {
      player.pause();
      return;
    }
    player.src = card.url;
    player.currentTime = 0;
    player.play().then(() => setPlaying(card), () => setPlaying(null));
  }

  async function copy(card, btn) {
    const text = cardText(card);
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = h('textarea', { class: 'stt-file' }, text);
      document.body.append(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
    replace(btn, icon('check', 15), 'Copied');
    btn.classList.add('is-done');
    setTimeout(() => {
      if (!btn.isConnected) return;
      replace(btn, icon('copy', 15), 'Copy');
      btn.classList.remove('is-done');
    }, 1600);
  }

  const speed = (audioSec, ms) => {
    const x = audioSec / (ms / 1000);
    return `${x.toFixed(x >= 10 ? 0 : x >= 1 ? 1 : 2)}× real time`;
  };

  const cardText = (card) => (card.segments.length > 1 ? card.segments.map((s) => `[${formatClock(s.start)}] ${s.text}`).join('\n') : card.text);

  // ---------- rendering ----------
  function renderAll() {
    if (disposed) return;
    renderStage();
    renderList();
  }

  function renderStage() {
    if (disposed) return;
    stage.classList.toggle('stt-stage--ready', state.phase === 'ready');
    if (state.phase === 'ready') {
      replace(
        stage,
        h(
          'div',
          { class: 'stt-stage__top' },
          h('div', { class: 'segmented segmented--sm', role: 'group', 'aria-label': 'Recording mode' }, modeBtns.toggle, modeBtns.hold),
          h('span', { class: 'stt-ready' }, h('span', { class: 'stt-ready__dot' }), 'Whisper base ready'),
        ),
        h('div', { class: 'stt-recorder' }, orb, clock, hint),
        waveWrap,
        noticeSlot,
        h(
          'div',
          { class: 'stt-drop' },
          h('span', { class: 'stt-drop__icon' }, uploadIcon(20)),
          h(
            'div',
            { class: 'stt-drop__text' },
            h('strong', null, 'Drop an audio file'),
            h('span', null, `WAV, MP3, M4A, OGG or WebM, up to ${MAX_FILE_SECONDS / 60} minutes`),
          ),
          h('button', { class: 'btn btn--ghost btn--sm stt-drop__btn', type: 'button', onclick: () => fileInput.click() }, 'Choose file'),
          fileInput,
        ),
        statEl,
      );
      wave.redraw();
      updateControls();
      renderNotice();
      return;
    }
    replace(stage, h('div', { class: 'stt-stage__center' }, stageBody()));
  }

  function stageBody() {
    const size = formatBytes(model.approxBytes, { approx: model.approxBytesIsEstimate });
    if (state.phase === 'checking') return h('div', { class: 'preload' }, h('div', { class: 'preload__glyph' }, micIcon(26)), h('p', { class: 'preload__sub' }, 'Checking for the model…'));
    if (state.phase === 'unavailable') {
      return h(
        'div',
        { class: 'preload' },
        h('div', { class: 'preload__glyph' }, icon('lock', 26)),
        h('h2', { class: 'preload__title' }, 'Not available in this build'),
        h('p', { class: 'preload__sub' }, `Whisper base (${size}) is not hosted here yet, so nothing can be downloaded or transcribed.`),
        licenseBadgeEl(model),
        state.reasons.length ? h('ul', { class: 'stt-reasons' }, state.reasons.map((r) => h('li', null, r.text))) : null,
        h('p', { class: 'preload__note' }, 'The microphone is never accessed on this page until the model can run.'),
      );
    }
    const gpu = state.gpu;
    if (gpu.status === 'unavailable' || gpu.status === 'lost') return gpuFallbackPanel(gpu, { compact: true });
    if (state.phase === 'loading') return loaderEl();
    return h(
      'div',
      { class: 'preload' },
      h('div', { class: 'app-glyph', 'aria-hidden': 'true' }, appIcon(demo.id, 64)),
      h('h2', { class: 'preload__title' }, 'Whisper base'),
      h('p', { class: 'preload__sub' }, `Speech recognition · ${size} · runs on your GPU`),
      state.phase === 'error' ? h('div', { class: 'inline-error' }, icon('alert', 16), h('span', null, state.loadError)) : null,
      h(
        'button',
        { class: 'btn btn--primary btn--xl', type: 'button', disabled: gpu.status !== 'ready', onclick: () => load() },
        state.phase === 'error' ? icon('refresh', 18) : icon('bolt', 18),
        state.phase === 'error' ? 'Try again' : 'Load model',
      ),
      h(
        'div',
        { class: 'preload__facts' },
        h('p', { class: 'preload__note' }, 'Downloads once, then your browser keeps a copy.'), hfBadge(state.source), onDeviceNote(model),
        h('p', { class: 'preload__note' }, 'Nothing is recorded until you press the microphone, and no audio is uploaded anywhere.'),
      ),
    );
  }

  let loaderParts = null;
  function loaderEl() {
    const bar = h('div', { class: 'progress__bar' });
    const detail = h('p', { class: 'loader__detail' });
    const steps = LOAD_STAGES.map((s) => h('li', { class: 'steps__item' }, h('span', { class: 'steps__dot' }), s.label));
    loaderParts = { bar, detail, steps };
    const el = h(
      'div',
      { class: 'loader' },
      h('div', { class: 'app-glyph', 'aria-hidden': 'true' }, appIcon(demo.id, 64)),
      h('h2', { class: 'loader__title' }, 'Loading Whisper base'),
      h('div', { class: 'progress', role: 'progressbar', 'aria-label': 'Loading progress' }, bar),
      detail,
      h('ul', { class: 'steps' }, steps),
      hfBadge(state.source),
      h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => cancelLoad() }, 'Cancel'),
    );
    updateLoader();
    return el;
  }

  function updateLoader() {
    if (!loaderParts || state.phase !== 'loading') return;
    const { stage: cur, fractions, loaded, total, detail } = state.load;
    const idx = LOAD_STAGES.findIndex((s) => s.id === cur);
    let overall = 0;
    LOAD_STAGES.forEach((s, i) => {
      const f = i < idx ? 1 : i === idx ? (fractions[s.id] ?? 0) : 0;
      overall += s.weight * f;
      loaderParts.steps[i].className = i < idx ? 'steps__item is-done' : i === idx ? 'steps__item is-active' : 'steps__item';
    });
    loaderParts.bar.style.width = `${(overall * 100).toFixed(1)}%`;
    loaderParts.detail.textContent =
      cur === 'download' && total ? `${formatBytes(loaded)} of ${formatBytes(total)}` : cur === 'gpu' && detail ? `Loading the ${detail}` : cur === 'runtime' ? 'Compiling the runtime' : '';
  }

  function updateControls() {
    if (state.phase !== 'ready') return;
    for (const [k, b] of Object.entries(modeBtns)) {
      b.className = state.mode === k ? 'segmented__btn is-active' : 'segmented__btn';
      b.disabled = state.rec !== 'idle';
    }
    const rec = state.rec;
    orb.className = `stt-orb stt-orb--${rec}`;
    replace(recBtn, rec === 'recording' ? h('span', { class: 'stt-rec__stop' }) : rec === 'transcribing' ? h('span', { class: 'stt-rec__spin' }) : micIcon(40));
    recBtn.disabled = rec === 'transcribing';
    recBtn.setAttribute('aria-label', rec === 'recording' ? 'Stop recording' : state.mode === 'hold' ? 'Hold to record' : 'Start recording');
    recBtn.setAttribute('aria-pressed', rec === 'recording' ? 'true' : 'false');
    if (rec === 'idle' && !state.cards.some((c) => c.status === 'running')) clock.textContent = '00:00.0';
    clock.classList.toggle('is-live', rec === 'recording');
    const dropBtn = stage.querySelector('.stt-drop__btn');
    if (dropBtn) dropBtn.disabled = rec !== 'idle';
    clearBtn.disabled = rec === 'transcribing' || !state.cards.length;
    langSelect.disabled = rec === 'transcribing';
    statEl.textContent = state.last ? `Last clip: ${formatClock(state.last.audioSec)} of audio in ${formatMs(state.last.ms)} · ${speed(state.last.audioSec, state.last.ms)}` : '';
    updateHint();
  }

  function updateHint() {
    if (state.phase !== 'ready') return;
    const support = micSupport();
    let content;
    if (state.rec === 'starting') content = 'Waiting for the microphone…';
    else if (state.rec === 'recording') content = state.mode === 'hold' ? 'Listening. Release to transcribe.' : 'Listening. Tap again to transcribe. Esc cancels.';
    else if (state.rec === 'transcribing') {
      const card = state.cards.find((c) => c.status === 'running');
      const part = card?.windows > 1 ? ` part ${Math.min(card.windows, Math.floor(card.progress * card.windows) + 1)} of ${card.windows}` : '';
      content = [
        txAbort?.signal.aborted ? 'Stopping after this step…' : `Transcribing${part}…`,
        card && !txAbort?.signal.aborted ? h('button', { class: 'stt-hint__stop', type: 'button', onclick: () => stopTranscription() }, icon('stop', 12), 'Stop') : null,
      ];
    } else if (support !== 'ok') content = support === 'insecure' ? 'The microphone needs https or localhost. You can still upload a file.' : 'No microphone API here. You can still upload a file.';
    else content = state.mode === 'hold' ? 'Hold the button (or Space) while you speak' : 'Tap the button (or press Space) to start';
    replace(hint, content);
  }

  function renderNotice() {
    if (state.phase !== 'ready') return;
    const n = state.notice;
    replace(
      noticeSlot,
      n ? h('div', { class: n.tone === 'error' ? 'inline-error stt-notice' : 'stt-notice stt-notice--info' }, icon(n.tone === 'error' ? 'alert' : 'info', 16), h('span', null, n.text)) : null,
    );
  }

  function renderList() {
    if (disposed) return;
    countEl.textContent = state.cards.length ? String(state.cards.length) : '';
    clearBtn.disabled = state.rec === 'transcribing' || !state.cards.length;
    if (!state.cards.length) {
      replace(
        list,
        h(
          'div',
          { class: 'stt-empty' },
          h('div', { class: 'stt-empty__art', 'aria-hidden': 'true' }, Array.from({ length: 9 }, (_, i) => h('span', { class: 'stt-empty__bar', style: { animationDelay: `${i * 0.11}s` } }))),
          h('p', { class: 'stt-empty__title' }, 'Your transcripts appear here'),
          h('p', { class: 'stt-empty__text' }, state.phase === 'ready'
                ? 'Record a sentence or drop in a clip. Each one becomes a card you can replay and copy.'
                : state.phase === 'unavailable'
                  ? 'Nothing to show until the model can run here.'
                  : 'Load the model to start. Recordings stay in this tab.'),
        ),
      );
      return;
    }
    replace(list, state.cards.map((c) => cardEl(c)));
  }

  function cardEl(card) {
    card.el = h('article', { class: 'stt-card' });
    updateCard(card);
    return card.el;
  }

  function updateCard(card) {
    if (!card.el || disposed) return;
    const running = card.status === 'running';
    card.el.className = `stt-card${running ? ' stt-card--live' : ''}${card.status === 'error' ? ' stt-card--error' : ''}`;
    const time = card.at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    let body;
    if (card.status === 'error') body = h('p', { class: 'stt-card__text stt-card__text--muted' }, `Transcription failed: ${card.error}`);
    else if (!running && !card.text) body = h('p', { class: 'stt-card__text stt-card__text--muted' }, card.status === 'stopped' ? 'Stopped before any words were decoded.' : 'No speech detected.');
    else if (!running && card.segments.length > 1) {
      body = h(
        'ol',
        { class: 'stt-segs' },
        card.segments.map((s) => h('li', { class: 'stt-seg' }, h('span', { class: 'stt-seg__time' }, formatClock(s.start)), h('span', { class: 'stt-seg__text' }, s.text))),
      );
    } else body = h('p', { class: 'stt-card__text' }, card.text || h('span', { class: 'stt-card__wait' }, 'Listening to the clip…'), running ? h('span', { class: 'caret' }) : null);

    const stat = running
      ? h('span', { class: 'stt-card__stat' }, h(
            'span',
            { class: 'mini-progress' },
            card.windows > 1
              ? h('span', { class: 'mini-progress__bar', style: { width: `${Math.round(card.progress * 100)}%` } })
              : h('span', { class: 'mini-progress__bar stt-busy-bar' }),
          ), 'Transcribing')
      : card.ms
        ? h('span', { class: 'stt-card__stat' }, `${formatMs(card.ms)} · ${speed(card.audioSec, card.ms)}${card.status === 'stopped' ? ' · stopped' : ''}`)
        : h('span', { class: 'stt-card__stat' }, card.status === 'stopped' ? 'Stopped' : '');

    const isPlaying = playing === card;
    replace(
      card.el,
      h(
        'header',
        { class: 'stt-card__head' },
        h('span', { class: card.source === 'mic' ? 'stt-card__src stt-card__src--mic' : 'stt-card__src' }, card.source === 'mic' ? micIcon(14) : fileIcon(14)),
        h('span', { class: 'stt-card__label', title: card.label }, card.label),
        h('span', { class: 'stt-card__meta' }, `${time} · ${formatClock(card.audioSec)}`),
      ),
      body,
      h(
        'footer',
        { class: 'stt-card__foot' },
        stat,
        h(
          'div',
          { class: 'stt-card__actions' },
          h(
            'button',
            { class: 'stt-act', type: 'button', 'aria-label': isPlaying ? 'Stop playback' : 'Play the clip', onclick: () => togglePlay(card) },
            icon(isPlaying ? 'stop' : 'play', 14),
            isPlaying ? 'Stop' : 'Play',
          ),
          running || !card.text
            ? null
            : h('button', { class: 'stt-act', type: 'button', 'aria-label': 'Copy transcript', onclick: (e) => copy(card, e.currentTarget) }, icon('copy', 15), 'Copy'),
        ),
      ),
    );
  }

  return () => {
    disposed = true;
    startToken++;
    loadAbort?.abort();
    txAbort?.abort();
    mic?.cancel();
    cancelAnimationFrame(raf);
    window.removeEventListener('keydown', onKeyDown);
    window.removeEventListener('keyup', onKeyUp);
    unsubscribeGpu();
    wave.dispose();
    player.pause();
    player.removeAttribute('src');
    for (const c of state.cards) URL.revokeObjectURL(c.url);
  };
}
