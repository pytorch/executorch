import { computeBadge } from '../config/catalog.js';
import { h } from '../lib/dom.js';
import { icon } from './icons.js';

export function computeBadgeEl(demo, { long = false } = {}) {
  const badge = computeBadge(demo);
  return h('span', { class: 'badge badge--live', title: badge.description }, h('span', { class: 'badge__pulse' }), long ? badge.label : badge.short);
}

export function licenseBadgeEl(model) {
  const status = model.license.status;
  const cls = status === 'cleared' ? 'license license--cleared' : status === 'blocked' ? 'license license--blocked' : 'license license--todo';
  const text = status === 'cleared' ? 'License cleared' : status === 'blocked' ? 'Blocked' : 'License review: to do';
  return h('span', { class: cls }, icon(status === 'cleared' ? 'check' : 'lock', 12), text);
}

// "Original model by …" line with links to the upstream release and its paper.
export function creditEl(model) {
  const c = model.credit;
  if (!c) return null;
  return h(
    'p',
    { class: 'credit' },
    'Original model by ',
    h('a', { href: c.url, target: '_blank', rel: 'noopener' }, c.by),
    c.paper ? [' · ', h('a', { href: c.paper, target: '_blank', rel: 'noopener' }, 'paper')] : null,
  );
}

// Attribution a model's license requires us to display, e.g. "Built with Llama".
export function attributionEl(model) {
  return model.attribution ? h('p', { class: 'attribution' }, model.attribution) : null;
}
