import { CATEGORIES, COMPUTE, LISTED_CATEGORIES, LISTED_DEMOS, demosIn } from '../config/catalog.js';
import { gpuState } from '../gpu/context.js';
import { h, replace } from '../lib/dom.js';
import { appIcon } from './app-icons.js';
import { computeBadgeEl } from './badges.js';
import { categoryIcon } from './category-icons.js';
import { gpuFallbackPanel } from './gpu-fallback.js';
import { icon } from './icons.js';
import { illustration } from './illustrations.js';

const CATEGORY_LABEL = Object.fromEntries(CATEGORIES.map((c) => [c.id, c.label]));

export function demoCard(demo, { compact = false } = {}) {
  return h(
    'a',
    { class: compact ? 'card card--compact' : 'card', href: `#/demo/${demo.id}`, dataset: { compute: demo.compute } },
    h('div', { class: 'card__art' }, illustration(demo.art), computeBadgeEl(demo)),
    h(
      'div',
      { class: 'card__body' },
      h('span', { class: `chip chip--${demo.category}` }, categoryIcon(demo.category, 13), CATEGORY_LABEL[demo.category]),
      h('div', { class: 'card__heading' }, appIcon(demo.id, 28), h('h3', { class: 'card__title' }, demo.title)),
      h('p', { class: 'card__blurb' }, demo.blurb),
      compact ? null : h('p', { class: 'card__meta' }, h('span', { class: 'card__meta-label' }, `${demo.meta.label}:`), ' ', demo.meta.value),
      h('span', { class: 'card__launch' }, 'Launch', icon('arrowRight', 15)),
    ),
  );
}

function featureCard(demo) {
  return h(
    'a',
    { class: 'feature', href: `#/demo/${demo.id}` },
    h('div', { class: 'feature__art' }, illustration(demo.art)),
    h(
      'div',
      { class: 'feature__body' },
      h('p', { class: 'eyebrow' }, 'Featured · Live camera'),
      h('h2', { class: 'feature__title' }, demo.title),
      h('p', { class: 'feature__blurb' }, demo.blurb),
      h('p', { class: 'card__meta' }, h('span', { class: 'card__meta-label' }, `${demo.meta.label}:`), ' ', demo.meta.value),
      h('span', { class: 'btn btn--primary feature__cta' }, 'Try it', icon('arrowRight', 16)),
    ),
  );
}

export function mountCatalog(root, { category } = {}) {
  const real = LISTED_DEMOS.filter((d) => d.compute === COMPUTE.EXECUTORCH).length;

  const fallbackSlot = h('div', { class: 'catalog__fallback' });
  const unsubscribe = gpuState.subscribe((s) => {
    replace(fallbackSlot, s.status === 'unavailable' || s.status === 'lost' ? gpuFallbackPanel(s) : '');
  });

  replace(root,
    h(
      'div',
      { class: 'page catalog' },
      h(
        'header',
        { class: 'page-head' },
        h(
          'nav',
          { class: 'crumbs', 'aria-label': 'Breadcrumb' },
          h('a', { class: 'crumbs__link', href: '../' }, 'Playground'),
          h('span', { class: 'crumbs__sep', 'aria-hidden': 'true' }, '›'),
          h('span', { class: 'crumbs__here', 'aria-current': 'page' }, 'WebGPU'),
          h('span', { class: 'crumbs__tag' }, 'Experimental'),
        ),
        h('h1', { class: 'page-title' }, 'Run PyTorch models locally in your browser'),
        h(
          'p',
          { class: 'page-lede' },
          'Powered by ExecuTorch on WebGPU: real language, vision, and speech models running in this tab on your GPU. No server and no install, and your prompts, photos, and voice never leave your device.',
        ),
        h(
          'div',
          { class: 'experimental', role: 'note' },
          h('span', { class: 'experimental__tag' }, 'Experimental'),
          h(
            'span',
            null,
            'The ExecuTorch WebGPU backend is under active development. Expect rough edges, and expect performance and model support to change. ',
            h('a', { href: 'https://github.com/pytorch/executorch/issues', target: '_blank', rel: 'noopener' }, 'Report an issue'),
            '.',
          ),
        ),
        h(
          'div',
          { class: 'honesty' },
          h('span', { class: 'honesty__item honesty__item--live' }, h('span', { class: 'badge__pulse' }), `${real} ExecuTorch models`),
        ),
      ),
      fallbackSlot,
      LISTED_DEMOS.filter((d) => d.featured).map(featureCard),
      LISTED_CATEGORIES.map((c) =>
        h(
          'section',
          { class: 'band', id: `band-${c.id}`, 'aria-labelledby': `band-title-${c.id}` },
          h('div', { class: 'band__head' }, h('h2', { class: 'band__title', id: `band-title-${c.id}` }, categoryIcon(c.id, 16), c.label), h('span', { class: 'band__blurb' }, c.blurb)),
          h('div', { class: 'card-grid' }, demosIn(c.id).map((d) => demoCard(d))),
        ),
      ),
    ),
  );

  if (category) {
    requestAnimationFrame(() => root.querySelector(`#band-${category}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
  }
  return unsubscribe;
}
