import { h } from '../lib/dom.js';

// Stat tile: label, value, hint. A trailing unit ("117 tok/s") is set smaller so
// the number keeps its size in narrow tiles instead of truncating.
export function metricTile(label, value, hint) {
  const m = /^([~≈]?[\d.,]+(?:\.\d+)?)\s+(.+)$/.exec(String(value));
  return h(
    'div',
    { class: 'metric' },
    h('span', { class: 'metric__label' }, label),
    h('span', { class: 'metric__value' }, m ? [m[1], h('span', { class: 'metric__unit' }, ` ${m[2]}`)] : value),
    h('span', { class: 'metric__hint' }, hint),
  );
}
