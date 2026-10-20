import { getModel } from '../config/models.js';
import { gpuState } from '../gpu/context.js';
import { h, replace } from '../lib/dom.js';
import { formatBytes, formatInt, formatMs, formatRate } from '../lib/format.js';
import { renderMarkdown } from '../lib/markdown.js';
import { ExecuTorchLLM } from '../runtime/llm.js';
import { resolveModelSource } from '../runtime/sources.js';
import { attributionEl, licenseBadgeEl } from './badges.js';
import { browseMore, detailHeader } from './detail-shell.js';
import { gpuFallbackPanel } from './gpu-fallback.js';
import { appIcon } from './app-icons.js';
import { hfBadge } from './hf-badge.js';
import { icon } from './icons.js';
import { onDeviceNote } from './on-device.js';
import { metricTile } from './metric.js';
import { attachVoiceInput } from './voice-input.js';

const SUGGESTED_PROMPTS = [
  'Explain WebGPU to a 10-year-old',
  'Write a haiku about running AI on a laptop',
  'Give me three fun facts about octopuses',
  'What are the pros and cons of on-device AI?',
];
const SYSTEM_PROMPT = 'You are a helpful, concise assistant running fully on the user’s device. Keep answers short unless asked for detail.';
const STYLES = [
  { id: 'precise', label: 'Precise', sampling: { temperature: 0.2, topP: 0.8 } },
  { id: 'balanced', label: 'Balanced', sampling: { temperature: 0.7, topP: 0.9 } },
  { id: 'creative', label: 'Creative', sampling: { temperature: 1.0, topP: 0.95 } },
];
const LOAD_STAGES = [
  { id: 'runtime', label: 'Start the ExecuTorch WebGPU runtime', weight: 0.05 },
  { id: 'weights', label: 'Download the model (cached afterwards)', weight: 0.75 },
  { id: 'tokenizer', label: 'Load the tokenizer', weight: 0.05 },
  { id: 'gpu', label: 'Upload weights and build the GPU graph', weight: 0.15 },
];
const MAX_NEW_TOKENS = 768;

// The loaded model outlives the view, so leaving the page and coming back does
// not download or rebuild anything. Only one stays on the GPU: each holds a KV
// cache for its whole context (about 2 GB for Qwen 3), and two at once can
// exhaust GPU memory.
const loaded = new Map();

export function mountChat(root, demo) {
  const models = demo.models.map(getModel);
  const state = {
    model: models[0],
    source: null,
    phase: 'idle', // idle | loading | ready | error | unavailable
    load: { stage: null, fractions: {}, bytes: null },
    loadError: null,
    loadAbort: null,
    lm: null,
    loadMs: null,
    style: STYLES[1],
    messages: [],
    queued: null,
    generating: false,
    genAbort: null,
    metrics: {},
  };
  let disposed = false;

  const modelSelect = h(
    'select',
    { class: 'select', 'aria-label': 'Model', onchange: (e) => switchModel(e.target.value) },
    models.map((m) => h('option', { value: m.id }, `${m.shortName} · ${formatBytes(m.approxBytes, { approx: true })}`)),
  );
  const header = detailHeader(demo, { controls: h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Model'), modelSelect) });

  const styleButtons = STYLES.map((s) =>
    h('button', { class: 'segmented__btn', type: 'button', 'aria-pressed': String(s === state.style), onclick: () => setStyle(s) }, s.label),
  );
  const newChatBtn = h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => newChat() }, icon('plus', 15), 'New chat');
  const toolbar = h(
    'div',
    { class: 'toolbar' },
    h('div', { class: 'segmented segmented--sm', role: 'group', 'aria-label': 'Response style' }, h('span', { class: 'segmented__label' }, 'Style'), styleButtons),
    h('div', { class: 'toolbar__right' }, newChatBtn),
  );

  const notes = h('div', { class: 'truth' });
  const stage = h('div', { class: 'chat__stage', 'aria-live': 'polite' });
  const composerInput = h('textarea', { class: 'composer__input', rows: 1, placeholder: 'Message the model…', 'aria-label': 'Message' });
  const composerHint = h('span', { class: 'composer__hint' });
  const micSlot = h('span', { class: 'composer__mic' });
  const sendBtn = h('button', { class: 'send-btn', type: 'button', 'aria-label': 'Send' }, icon('arrowUp', 18));
  const composer = h(
    'form',
    { class: 'composer', onsubmit: (e) => (e.preventDefault(), onSendClick()) },
    composerInput,
    h('div', { class: 'composer__bar' }, composerHint, micSlot, sendBtn),
  );
  const metricsPanel = h('aside', { class: 'metrics', 'aria-label': 'About this run' });

  replace(root,
    h(
      'div',
      { class: 'page detail chat-page' },
      header,
      toolbar,
      notes,
      h('div', { class: 'detail-grid' }, h('div', { class: 'chat' }, stage, composer), metricsPanel),
      browseMore(demo.id),
    ),
  );

  const detachVoice = attachVoiceInput(micSlot, {
    onText: (text) => {
      composerInput.value = composerInput.value ? `${composerInput.value} ${text}` : text;
      autosize();
      composerInput.focus();
    },
  });

  composerInput.addEventListener('input', autosize);
  composerInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      onSendClick();
    }
  });
  sendBtn.addEventListener('click', (e) => {
    e.preventDefault();
    onSendClick();
  });
  const unsubscribeGpu = gpuState.subscribe(() => renderAll());

  // ---------- actions ----------
  function autosize() {
    composerInput.style.height = 'auto';
    composerInput.style.height = `${Math.min(composerInput.scrollHeight, 220)}px`;
  }

  function setStyle(s) {
    state.style = s;
    styleButtons.forEach((b, i) => b.setAttribute('aria-pressed', String(STYLES[i] === s)));
  }

  function newChat() {
    state.genAbort?.abort();
    state.messages = [];
    state.queued = null;
    state.metrics = {};
    renderAll();
  }

  async function resolveSource() {
    const model = state.model;
    const source = await resolveModelSource(model);
    if (disposed || model !== state.model) return;
    state.source = source;
    const cached = loaded.get(model.id);
    if (cached) Object.assign(state, { lm: cached.lm, loadMs: cached.loadMs, phase: 'ready' });
    else state.phase = source.kind === 'unavailable' ? 'unavailable' : 'idle';
    renderAll();
  }

  function switchModel(id) {
    if (id === state.model.id) return;
    state.loadAbort?.abort();
    state.genAbort?.abort();
    Object.assign(state, { model: getModel(id), source: null, phase: 'idle', lm: null, loadMs: null, loadError: null, queued: null, generating: false, metrics: {} });
    renderAll();
    resolveSource();
  }

  async function load() {
    if (gpuState.get().status !== 'ready' || state.phase === 'loading' || !state.source || state.source.kind === 'unavailable') return;
    const model = state.model;
    const controller = new AbortController();
    Object.assign(state, { phase: 'loading', loadError: null, loadAbort: controller, load: { stage: 'runtime', fractions: {}, bytes: null } });
    renderAll();
    const started = performance.now();
    for (const [id, other] of loaded) {
      other.lm.dispose();
      loaded.delete(id);
    }
    try {
      const lm = await ExecuTorchLLM.load(state.source, {
        signal: controller.signal,
        onProgress: (p) => {
          if (model !== state.model) return;
          const idx = LOAD_STAGES.findIndex((s) => s.id === p.stage);
          const cur = LOAD_STAGES.findIndex((s) => s.id === state.load.stage);
          if (idx > cur) state.load.stage = p.stage;
          state.load.fractions[p.stage] = p.fraction;
          if (p.stage === 'weights') state.load.bytes = p;
          renderLoader();
        },
      });
      const loadMs = performance.now() - started;
      loaded.set(model.id, { lm, loadMs });
      if (disposed || model !== state.model) return;
      Object.assign(state, { lm, phase: 'ready', loadMs });
      renderAll();
      if (state.queued) {
        state.queued.status = null;
        state.queued = null;
        await generate();
      }
    } catch (err) {
      if (err.name === 'AbortError' || model !== state.model) return;
      Object.assign(state, { phase: 'error', loadError: err.message });
      renderAll();
    } finally {
      state.loadAbort = null;
    }
  }

  function onSendClick() {
    if (state.generating) {
      state.genAbort?.abort();
      return;
    }
    const text = composerInput.value.trim();
    if (!text) return;
    composerInput.value = '';
    autosize();
    if (state.queued) {
      state.queued.text += `\n\n${text}`;
      renderStage();
      return;
    }
    const msg = { role: 'user', text };
    state.messages.push(msg);
    if (state.phase !== 'ready') {
      msg.status = 'queued';
      state.queued = msg;
      renderAll();
      if (state.phase === 'idle') load();
      return;
    }
    generate();
  }

  function chatHistory() {
    const history = [{ role: 'system', content: SYSTEM_PROMPT }];
    for (const m of state.messages) {
      if (m.role === 'user') history.push({ role: 'user', content: m.text });
      else if (m.role === 'assistant' && m.text) history.push({ role: 'assistant', content: m.text });
    }
    return history;
  }

  async function generate() {
    const lm = state.lm;
    const history = chatHistory();
    const msg = { role: 'assistant', text: '', streaming: true, prefill: null };
    state.messages.push(msg);
    const controller = new AbortController();
    Object.assign(state, { generating: true, genAbort: controller });
    renderAll();
    try {
      const r = await lm.generate(history, {
        maxNewTokens: MAX_NEW_TOKENS,
        sampling: state.style.sampling,
        signal: controller.signal,
        onPrefill: (done, total) => {
          msg.prefill = total > 64 ? { done, total } : null;
          paintMessage(msg);
        },
        onText: (_delta, full) => {
          msg.prefill = null;
          msg.text = full;
          paintMessage(msg);
        },
      });
      msg.text = r.text;
      state.metrics = { ttft: r.ttftMs, decodeTps: r.decodeTps, newTokens: r.newTokens, prefilled: r.prefilledTokens };
      msg.footer = `${formatInt(r.newTokens)} tokens · ${formatRate(r.decodeTps, 'tok/s')}`;
      if (r.stop === 'aborted') msg.stopNote = 'Stopped by you.';
      else if (r.stop === 'length') msg.stopNote = 'Reached the reply length limit.';
    } catch (err) {
      state.messages.push({ role: 'system', kind: 'error', text: `Generation failed: ${err.message}` });
    } finally {
      msg.streaming = false;
      msg.prefill = null;
      Object.assign(state, { generating: false, genAbort: null });
      if (!disposed) renderAll();
    }
  }

  // ---------- rendering ----------
  function renderAll() {
    if (disposed) return;
    renderNotes();
    renderStage();
    renderComposer();
    renderMetrics();
  }

  function renderNotes() {
    const kind = state.source?.kind;
    const where =
      kind === 'remote'
        ? ['Hosted on Hugging Face', 'Model files come from Hugging Face, pinned to one version and checked after download.']
        : kind === 'dev'
          ? ['Local review build', 'Model files are served by your local dev server until they are published.']
          : null;
    replace(notes,
      h(
        'div',
        { class: 'truth__item truth__item--ok' },
        h('span', { class: 'truth__tag' }, 'On-device'),
        h('span', null, 'The model runs in this tab with ExecuTorch on WebGPU. Your messages never leave your device.'),
      ),
      where ? h('div', { class: 'truth__item' }, h('span', { class: 'truth__tag' }, where[0]), h('span', null, where[1])) : null,
    );
  }

  function renderStage() {
    const atBottom = stage.scrollHeight - stage.scrollTop - stage.clientHeight < 80;
    let content;
    if (state.phase === 'loading') content = h('div', { class: 'stage-stack' }, loaderPanel(), state.messages.length ? transcript() : null);
    else if (state.phase !== 'ready') content = h('div', { class: 'stage-stack' }, state.messages.length ? transcript() : null, preloadPanel());
    else content = state.messages.length ? transcript() : readyEmpty();
    replace(stage, content);
    if (state.phase !== 'ready' || !state.messages.length) stage.scrollTop = 0;
    else if (atBottom || state.generating) stage.scrollTop = stage.scrollHeight;
  }

  function suggestions(onPick) {
    return h(
      'div',
      { class: 'preload__chips' },
      SUGGESTED_PROMPTS.map((p) => h('button', { class: 'suggest', type: 'button', onclick: () => onPick(p) }, p)),
    );
  }

  // The chat models side by side, so the choice sits next to the Load button.
  function modelChooser() {
    if (models.length < 2) return null;
    return h(
      'div',
      { class: 'model-choice', role: 'radiogroup', 'aria-label': 'Model' },
      models.map((m) =>
        h(
          'button',
          {
            class: 'model-choice__opt',
            type: 'button',
            role: 'radio',
            'aria-checked': String(m.id === state.model.id),
            disabled: state.phase === 'loading',
            onclick: () => {
              modelSelect.value = m.id;
              switchModel(m.id);
            },
          },
          h(
            'span',
            { class: 'model-choice__top' },
            h('span', { class: 'model-choice__name' }, m.shortName),
            h('span', { class: 'model-choice__size' }, formatBytes(m.approxBytes, { approx: true })),
          ),
          m.tagline ? h('span', { class: 'model-choice__meta' }, m.tagline) : null,
        ),
      ),
    );
  }

  function preloadPanel() {
    const gpu = gpuState.get();
    const model = state.model;
    const size = formatBytes(model.approxBytes, { approx: true });
    const unavailable = state.phase === 'unavailable';
    const a = model.arch;
    return h(
      'div',
      { class: 'preload' },
      h('div', { class: 'app-glyph', 'aria-hidden': 'true' }, appIcon(demo.id, 64)),
      h('h2', { class: 'preload__title' }, model.name),
      a ? h('p', { class: 'preload__sub' }, `${a.layers} layers · ${formatInt(a.dModel)} wide · 4-bit weights · runs on your GPU`) : null,
      modelChooser(),
      state.phase === 'error' ? h('div', { class: 'inline-error', role: 'alert' }, icon('alert', 16), h('span', null, `Load failed: ${state.loadError}`)) : null,
      gpu.status === 'ready'
        ? h(
            'button',
            { class: 'btn btn--primary btn--xl', type: 'button', disabled: unavailable || !state.source, onclick: load },
            icon('bolt', 18),
            unavailable ? 'Not available in this build' : `Load model · ${size}`,
          )
        : gpu.status === 'probing'
          ? h('button', { class: 'btn btn--primary btn--xl', type: 'button', disabled: true }, 'Detecting GPU…')
          : gpuFallbackPanel(gpu, { compact: true }),
      unavailable
        ? h('p', { class: 'preload__note' }, 'Weights are not hosted yet ', licenseBadgeEl(model))
        : h('p', { class: 'preload__note' }, 'One-time download, then cached on your device. Nothing you type leaves the tab.'),
      hfBadge(state.source),
      attributionEl(model),
      onDeviceNote(state.model),
      suggestions((p) => {
        composerInput.value = p;
        autosize();
        composerInput.focus();
      }),
    );
  }

  let loaderEls = null;
  function loaderPanel() {
    const bar = h('div', { class: 'progress__bar' });
    const detail = h('p', { class: 'loader__detail' });
    const steps = LOAD_STAGES.map((s) => h('li', { class: 'steps__item', dataset: { stage: s.id } }, h('span', { class: 'steps__dot' }), s.label));
    loaderEls = { bar, detail, steps };
    const panel = h(
      'div',
      { class: 'loader' },
      h('h2', { class: 'loader__title' }, `Loading ${state.model.shortName}`),
      h('div', { class: 'progress', role: 'progressbar', 'aria-label': 'Load progress' }, bar),
      detail,
      h('ol', { class: 'steps' }, steps),
      hfBadge(state.source),
      h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => (state.loadAbort?.abort(), (state.phase = 'idle'), renderAll()) }, 'Cancel'),
    );
    queueMicrotask(renderLoader);
    return panel;
  }

  function renderLoader() {
    if (!loaderEls) return;
    const { fractions, stage: current, bytes } = state.load;
    const idx = LOAD_STAGES.findIndex((s) => s.id === current);
    let total = 0;
    LOAD_STAGES.forEach((s, i) => {
      const f = fractions[s.id] ?? (i < idx ? 1 : 0);
      total += s.weight * Math.min(1, f);
      loaderEls.steps[i].classList.toggle('is-done', f >= 1);
      loaderEls.steps[i].classList.toggle('is-active', i === idx && f < 1);
    });
    loaderEls.bar.style.width = `${(total * 100).toFixed(1)}%`;
    loaderEls.bar.parentElement.setAttribute('aria-valuenow', String(Math.round(total * 100)));
    loaderEls.detail.textContent =
      current === 'weights' && bytes
        ? bytes.fromCache
          ? 'Loaded from your device cache'
          : `${formatBytes(bytes.loaded)}${bytes.total ? ` of ${formatBytes(bytes.total)}` : ''}`
        : current === 'gpu'
          ? 'Compiling WGSL kernels and uploading weights…'
          : '';
  }

  function readyEmpty() {
    return h(
      'div',
      { class: 'ready-empty' },
      h('div', { class: 'preload__glyph preload__glyph--ok', 'aria-hidden': 'true' }, icon('check', 24)),
      h('h2', { class: 'preload__title' }, `${state.model.shortName} is running on your GPU`),
      h('p', { class: 'preload__sub' }, 'Ask anything. Replies stream token by token from ExecuTorch.'),
      attributionEl(state.model),
      suggestions((p) => {
        composerInput.value = p;
        onSendClick();
      }),
    );
  }

  function transcript() {
    return h('div', { class: 'transcript' }, state.messages.map(messageEl));
  }

  function messageEl(m) {
    if (m.role === 'user') {
      return h(
        'div',
        { class: 'msg msg--user' },
        h('div', { class: 'msg__bubble' }, m.text),
        m.status === 'queued'
          ? h('div', { class: 'msg__queued' }, icon('info', 13), state.phase === 'loading' ? 'Sends when the model finishes loading' : 'Press Load model to send')
          : null,
      );
    }
    if (m.kind === 'error') {
      return h('div', { class: 'msg msg--error', role: 'alert' }, icon('alert', 16), h('span', null, m.text));
    }
    const textEl = h('div', { class: 'msg__text md' });
    const prefillEl = h('div', { class: 'msg__prefill' });
    const el = h(
      'div',
      { class: 'msg msg--assistant msg--live' },
      h('div', { class: 'msg__label' }, `${state.model.shortName} · on your GPU`),
      h('div', { class: 'msg__bubble' }, prefillEl, textEl, m.streaming ? h('span', { class: 'caret' }) : null),
      m.stopNote ? h('div', { class: 'msg__foot' }, m.stopNote) : null,
      m.footer ? h('div', { class: 'msg__foot' }, m.footer) : null,
    );
    m.textEl = textEl;
    m.prefillEl = prefillEl;
    paintMessage(m, false);
    return el;
  }

  function paintMessage(m, scroll = true) {
    if (!m.textEl) return;
    m.textEl.innerHTML = renderMarkdown(m.text.trimStart());
    if (m.prefill) {
      const pct = (100 * m.prefill.done) / m.prefill.total;
      replace(m.prefillEl,
        h('span', null, `Reading ${formatInt(m.prefill.total)} tokens`),
        h('span', { class: 'mini-progress' }, h('span', { class: 'mini-progress__bar', style: { width: `${pct}%` } })),
      );
    } else {
      m.prefillEl.replaceChildren();
    }
    if (scroll) stage.scrollTop = stage.scrollHeight;
  }

  function renderComposer() {
    replace(sendBtn, icon(state.generating ? 'stop' : 'arrowUp', 18));
    sendBtn.setAttribute('aria-label', state.generating ? 'Stop generating' : 'Send');
    sendBtn.classList.toggle('send-btn--stop', state.generating);
    composerHint.textContent =
      state.phase === 'ready' && state.lm
        ? `${formatInt(state.lm.pos)} / ${formatInt(state.lm.maxCtx)} context`
        : state.phase === 'loading'
          ? 'Loading… you can type now'
          : 'Model not loaded';
  }

  function renderMetrics() {
    const m = state.metrics;
    const spec = state.source?.spec ?? null;
    replace(metricsPanel,
      h('h2', { class: 'metrics__title' }, 'This conversation'),
      h(
        'div',
        { class: 'metric-grid' },
        metricTile('Speed', formatRate(m.decodeTps, 'tok/s'), 'tokens generated per second'),
        metricTile('First token', formatMs(m.ttft), 'after you press send'),
        metricTile('Context', state.lm ? formatInt(state.lm.pos) : '—', state.lm ? `of ${formatInt(state.lm.maxCtx)} tokens` : 'tokens in memory'),
        metricTile('Load', formatMs(state.loadMs), 'download + GPU setup'),
      ),
      h('h3', { class: 'metrics__h' }, 'How it works'),
      h(
        'ul',
        { class: 'opt-list' },
        [
          'An ExecuTorch .pte with 4-bit grouped weights',
          'WebGPU backend: every layer runs as WGSL compute',
          spec?.prefillMax > 1 ? `Your prompt is read in batches of up to ${spec.prefillMax} tokens` : 'Your prompt is read token by token',
          'The KV cache stays on the GPU and is reused across turns',
          'Sampling (temperature, top-p) happens in JavaScript',
        ].map((t) => h('li', null, icon('check', 14), t)),
      ),
      h('p', { class: 'metrics__note' }, 'Speed depends on your GPU. Close other GPU-heavy tabs for the best experience.'),
    );
  }

  renderAll();
  resolveSource();

  return () => {
    disposed = true;
    unsubscribeGpu();
    detachVoice();
    state.loadAbort?.abort();
    state.genAbort?.abort();
  };
}
