// Small control builders shared by the vision task panels.
import { h } from '../lib/dom.js';

export function segmented(label, options, value, onChange) {
  return h(
    'div',
    { class: 'vision-control' },
    h('span', { class: 'vision-control__label' }, label),
    h(
      'div',
      { class: 'segmented segmented--sm vision-segmented', role: 'group', 'aria-label': label },
      options.map((o) =>
        h(
          'button',
          {
            class: o.id === value ? 'segmented__btn is-active' : 'segmented__btn',
            type: 'button',
            'aria-pressed': String(o.id === value),
            onclick: () => o.id !== value && onChange(o.id),
          },
          o.label,
        ),
      ),
    ),
  );
}

export function slider(label, { min, max, step = 1, value, format = String, onInput }) {
  const out = h('span', { class: 'vision-control__value' }, format(value));
  const input = h('input', {
    class: 'range__input vision-range',
    type: 'range',
    min,
    max,
    step,
    value,
    'aria-label': label,
    oninput: (e) => {
      const v = Number(e.target.value);
      out.textContent = format(v);
      onInput(v);
    },
  });
  return h('label', { class: 'vision-control' }, h('span', { class: 'vision-control__label' }, label, out), input);
}
