// Asks before loading a model that likely needs more memory than a phone or
// tablet gives one browser tab. iOS Safari closes the tab outright (no error the
// page can catch), so the check has to happen before the download starts.
import { h } from '../lib/dom.js';
import { formatBytes } from '../lib/format.js';
import { icon } from './icons.js';

// Above this estimate, phones and tablets are warned. The chat models
// (estimated 1.8-2.9 GB) close the tab on iPhones; the vision models stay
// under 0.3 GB.
export const MOBILE_PEAK_LIMIT = 400e6;

const accepted = new Set();

export function isMobileDevice(nav = globalThis.navigator) {
  if (!nav) return false;
  const ua = nav.userAgent ?? '';
  if (/iPhone|iPad|iPod|Android/i.test(ua)) return true;
  // iPadOS Safari reports a desktop Mac user agent.
  return /Macintosh/.test(ua) && (nav.maxTouchPoints ?? 0) > 1;
}

// Rough peak while loading: the downloaded bytes, the runtime's copy of the
// program, and GPU buffers for the weights plus any preallocated KV cache.
export function estimatePeakBytes(model) {
  return 2 * (model.approxBytes ?? 0) + (model.kvCacheBytes ?? 0);
}

export function needsMemoryWarning(model, nav = globalThis.navigator) {
  const lowMemory = typeof nav?.deviceMemory === 'number' && nav.deviceMemory <= 4;
  return (isMobileDevice(nav) || lowMemory) && estimatePeakBytes(model) > MOBILE_PEAK_LIMIT;
}

// Resolves true to go ahead. Asks once per model per page load.
export function confirmModelMemory(model) {
  if (accepted.has(model.id) || !needsMemoryWarning(model) || typeof HTMLDialogElement === 'undefined') {
    return Promise.resolve(true);
  }
  return new Promise((resolve) => {
    const finish = (ok) => {
      if (ok) accepted.add(model.id);
      dialog.close();
      dialog.remove();
      resolve(ok);
    };
    const cancel = h('button', { class: 'btn btn--ghost', type: 'button', onclick: () => finish(false) }, 'Cancel');
    const dialog = h(
      'dialog',
      { class: 'memory-guard', 'aria-labelledby': 'memory-guard-title' },
      h('div', { class: 'memory-guard__icon', 'aria-hidden': 'true' }, icon('alert', 22)),
      h('h2', { class: 'memory-guard__title', id: 'memory-guard-title' }, 'This model may be too big for this device'),
      h(
        'p',
        { class: 'memory-guard__body' },
        `${model.name} needs about ${formatBytes(estimatePeakBytes(model))} of memory while it loads. `,
        'Phones and tablets often close the tab when a page uses that much. It works best on a laptop or desktop.',
      ),
      h(
        'div',
        { class: 'memory-guard__actions' },
        cancel,
        h('button', { class: 'btn btn--primary', type: 'button', onclick: () => finish(true) }, 'Load anyway'),
      ),
    );
    dialog.addEventListener('cancel', (e) => {
      e.preventDefault();
      finish(false);
    });
    document.body.append(dialog);
    dialog.showModal();
    cancel.focus();
  });
}
