import { LISTED_DEMOS } from '../config/catalog.js';
import { h } from '../lib/dom.js';
import { appIcon } from './app-icons.js';
import { computeBadgeEl } from './badges.js';
import { demoCard } from './catalog.js';
import { icon } from './icons.js';

export function backLink() {
  return h('a', { class: 'back', href: '#/examples' }, icon('chevronLeft', 16), 'Back to Examples');
}

export function detailHeader(demo, { controls = null } = {}) {
  return h(
    'header',
    { class: 'detail-head' },
    backLink(),
    h(
      'div',
      { class: 'detail-head__row' },
      h('div', { class: 'detail-head__titles' }, appIcon(demo.id, 36), h('h1', { class: 'detail-title' }, demo.title), computeBadgeEl(demo, { long: true })),
      controls ? h('div', { class: 'detail-head__controls' }, controls) : null,
    ),
  );
}

export function browseMore(currentId) {
  return h(
    'section',
    { class: 'more', 'aria-label': 'Browse more examples' },
    h('h2', { class: 'more__title' }, 'Browse more examples'),
    h('div', { class: 'more__strip' }, LISTED_DEMOS.filter((d) => d.id !== currentId).map((d) => demoCard(d, { compact: true }))),
  );
}
