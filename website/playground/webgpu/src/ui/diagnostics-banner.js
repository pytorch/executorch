import { diagnostics, dismissDiagnostic } from '../gpu/context.js';
import { h, replace } from '../lib/dom.js';
import { icon } from './icons.js';

// Shader compile errors, validation errors, device loss, and self-test
// mismatches are shown on every page; they are never console-only.
export function mountDiagnosticsBanner(el) {
  return diagnostics.subscribe(({ items }) => {
    const visible = items.filter((d) => d.level === 'error' || d.level === 'warning');
    replace(el,
      ...visible.map((d) =>
        h(
          'div',
          { class: d.level === 'error' ? 'diag diag--error' : 'diag diag--warning', role: 'alert' },
          icon(d.level === 'error' ? 'alert' : 'info', 18),
          h(
            'div',
            { class: 'diag__body' },
            h('strong', { class: 'diag__title' }, d.title),
            d.detail ? h('details', { class: 'diag__details' }, h('summary', null, 'Details'), h('pre', { class: 'diag__pre' }, d.detail)) : null,
          ),
          h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Dismiss', onclick: () => dismissDiagnostic(d.id) }, icon('close', 16)),
        ),
      ),
    );
  });
}
