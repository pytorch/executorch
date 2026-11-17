import { FAILURE_COPY } from '../gpu/adapter.js';
import { initGPU } from '../gpu/context.js';
import { h } from '../lib/dom.js';
import { icon } from './icons.js';

const STILL_WORKS = [
  'Browse every demo and read exactly what each one does.',
  'Use the readable preview in On-device Chat to see the experience with scripted text.',
  'Run the plain-JavaScript side of Compare Engines to see what this page costs without a GPU.',
];

export function gpuFallbackPanel(state, { compact = false } = {}) {
  const copy = FAILURE_COPY[state.reason] ?? FAILURE_COPY['no-api'];
  const lost = state.status === 'lost';
  return h(
    'section',
    { class: compact ? 'fallback fallback--compact' : 'fallback', role: 'status' },
    h(
      'div',
      { class: 'fallback__visual', 'aria-hidden': 'true' },
      h('div', { class: 'fallback__chip' }, icon('chip', compact ? 28 : 40)),
      h('div', { class: 'fallback__slash' }),
    ),
    h(
      'div',
      { class: 'fallback__body' },
      h('p', { class: 'eyebrow' }, lost ? 'GPU reset' : 'WebGPU unavailable'),
      h('h2', { class: 'fallback__title' }, copy.title),
      h('p', { class: 'fallback__text' }, copy.body),
      state.detail ? h('p', { class: 'fallback__detail' }, h('span', { class: 'mono' }, state.detail)) : null,
      h(
        'div',
        { class: 'fallback__cols' },
        h('div', null, h('h3', { class: 'fallback__h' }, 'How to fix it'), h('ul', { class: 'fallback__list' }, copy.remedies.map((r) => h('li', null, r)))),
        compact
          ? null
          : h('div', null, h('h3', { class: 'fallback__h' }, 'What still works here'), h('ul', { class: 'fallback__list fallback__list--ok' }, STILL_WORKS.map((r) => h('li', null, r)))),
      ),
      lost
        ? h('button', { class: 'btn btn--primary', type: 'button', onclick: () => initGPU() }, icon('refresh', 16), 'Reconnect to the GPU')
        : null,
    ),
  );
}
