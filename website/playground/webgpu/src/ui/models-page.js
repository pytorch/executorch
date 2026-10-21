import { LISTED_DEMOS } from '../config/catalog.js';
import { LICENSE_STATUS, MODELS as ALL_MODELS } from '../config/models.js';
import { PHOTO_CREDITS } from '../config/photo-credits.js';
import { h, replace } from '../lib/dom.js';
import { formatBytes } from '../lib/format.js';
import { deleteStoredModel, storedFiles } from '../runtime/model-storage.js';
import { resolveModelSource } from '../runtime/sources.js';
import { attributionEl, creditEl } from './badges.js';
import { hfBadge } from './hf-badge.js';
import { icon } from './icons.js';

const MODELS = ALL_MODELS.filter((m) => LISTED_DEMOS.some((d) => d.models.includes(m.id)));

// Two-step delete: the first click arms the button, a second click within a few
// seconds removes the files.
const CONFIRM_MS = 4000;

// "Apache-2.0 (Small only; …)" → "Apache-2.0"; the full text stays in Details.
function shortLicense(text) {
  return text.split(/ \(|;/)[0];
}

export function mountModels(root) {
  const usedBy = (id) => LISTED_DEMOS.filter((d) => d.models.includes(id));
  let disposed = false;
  const stored = new Map();
  const summary = h('div', { class: 'storage-summary', 'aria-live': 'polite' });
  const slots = new Map(MODELS.map((m) => [m.id, h('div', { class: 'model-row__storage' })]));
  const hfSlots = new Map(MODELS.map((m) => [m.id, h('div', { class: 'model-row__hf' })]));
  const rows = new Map();
  const rowFor = (m, el) => (rows.set(m.id, el), el);
  // Models already in this browser come first, largest first; the rest keep catalog order.
  function sortRows() {
    const order = MODELS.map((m, i) => ({ m, i, bytes: stored.get(m.id)?.bytes ?? 0 })).sort((a, b) => (b.bytes > 0) - (a.bytes > 0) || b.bytes - a.bytes || a.i - b.i);
    const list = rows.get(MODELS[0].id)?.parentElement;
    for (const { m } of order) {
      const row = rows.get(m.id);
      row.classList.toggle('model-row--stored', (stored.get(m.id)?.bytes ?? 0) > 0);
      list?.append(row);
    }
  }
  for (const m of MODELS) resolveModelSource(m).then((src) => !disposed && replace(hfSlots.get(m.id), hfBadge(src)));

  function renderSummary() {
    const total = [...stored.values()].reduce((a, s) => a + s.bytes, 0);
    const count = [...stored.values()].filter((s) => s.urls.length).length;
    replace(summary,
      h('div', { class: 'storage-summary__text' }, icon('download', 16), count ? `${count} model${count > 1 ? 's' : ''} on this device · ${formatBytes(total)}` : 'No models stored on this device yet'),
      count ? confirmButton('Delete all', `Delete all ${formatBytes(total)}?`, async () => {
        for (const m of MODELS) if (stored.get(m.id)?.urls.length) await deleteStoredModel(m);
        await refreshAll();
      }) : null,
    );
  }

  function confirmButton(label, confirmLabel, onConfirm, { iconOnly = false, ariaLabel = label } = {}) {
    let armed = null;
    const btn = h('button', { class: iconOnly ? 'trash-btn' : 'btn btn--ghost btn--sm trash-btn--text', type: 'button', 'aria-label': ariaLabel, title: ariaLabel });
    const idle = () => {
      armed = null;
      btn.classList.remove('is-armed');
      replace(btn, icon('trash', 16), iconOnly ? null : label);
    };
    btn.addEventListener('click', async () => {
      if (!armed) {
        btn.classList.add('is-armed');
        replace(btn, icon('trash', 16), confirmLabel);
        armed = setTimeout(idle, CONFIRM_MS);
        return;
      }
      clearTimeout(armed);
      btn.disabled = true;
      replace(btn, 'Deleting…');
      await onConfirm();
    });
    idle();
    return btn;
  }

  function renderSlot(m) {
    const slot = slots.get(m.id);
    const s = stored.get(m.id);
    if (!s) return replace(slot, h('span', { class: 'model-row__store-text' }, 'Checking storage…'));
    if (!s.urls.length) return replace(slot, h('span', { class: 'model-row__store-text' }, 'Not on this device'));
    replace(slot,
      h('span', { class: 'model-row__store-text is-stored' }, h('span', { class: 'model-row__store-dot' }), `On this device · ${formatBytes(s.bytes)}`),
      confirmButton('Delete', `Delete ${formatBytes(s.bytes)}?`, async () => {
        await deleteStoredModel(m);
        await refresh(m);
        sortRows();
      }, { iconOnly: true, ariaLabel: `Delete ${m.name} from this device` }),
    );
  }

  async function refresh(m) {
    const s = await storedFiles(m).catch(() => ({ urls: [], bytes: 0 }));
    if (disposed) return;
    stored.set(m.id, s);
    renderSlot(m);
    renderSummary();
  }

  async function refreshAll() {
    await Promise.all(MODELS.map(refresh));
    if (!disposed) sortRows();
  }
  replace(root,
    h(
      'div',
      { class: 'page models-page' },
      h(
        'header',
        { class: 'page-head' },
        h('p', { class: 'eyebrow' }, 'Playground'),
        h('h1', { class: 'page-title' }, 'Models'),
        h('p', { class: 'page-lede' }, 'The models behind the playground. Each one downloads once, then stays in your browser so it loads instantly next time. Remove any of them below to free up space.'),
      ),
      summary,
      h(
        'div',
        { class: 'model-list' },
        MODELS.map((m) => {
          return rowFor(m, h(
            'article',
            { class: 'model-row' },
            h(
              'div',
              { class: 'model-row__main' },
              h('h2', { class: 'model-row__name' }, m.name),
              h('p', { class: 'model-row__meta' }, `${m.task} · ${formatBytes(m.approxBytes, { approx: true })}${m.approxBytesIsEstimate ? ' (estimate)' : ''}`),
              h('p', { class: 'model-row__used' }, 'Used by: ', usedBy(m.id).map((d, i) => [i ? ', ' : '', h('a', { href: `#/demo/${d.id}` }, d.title)])),
              creditEl(m),
              attributionEl(m),
            ),
            h(
              'div',
              { class: 'model-row__license' },
              h('p', { class: 'model-row__upstream' }, h('span', { class: 'model-row__label' }, 'License'), shortLicense(m.license.upstream)),
              m.license.status === LICENSE_STATUS.CLEARED ? null : h('span', { class: 'tag tag--muted' }, 'Local preview only'),
              hfSlots.get(m.id),
            ),
            slots.get(m.id),
            h('details', { class: 'model-row__prov' }, h('summary', null, 'Details'), h('p', null, m.license.upstream), h('p', null, m.license.provenance)),
          ));
        }),
      ),
      h(
        'section',
        { class: 'photo-credits', id: 'photo-credits' },
        h('h2', { class: 'photo-credits__title' }, 'Sample photo credits'),
        h(
          'ul',
          { class: 'photo-credits__list' },
          PHOTO_CREDITS.map((c) =>
            h(
              'li',
              null,
              h('a', { href: c.source, target: '_blank', rel: 'noopener' }, c.title),
              ' by ',
              c.authorUrl ? h('a', { href: c.authorUrl, target: '_blank', rel: 'noopener' }, c.author) : c.author,
              ' · ',
              c.licenseUrl ? h('a', { href: c.licenseUrl, target: '_blank', rel: 'noopener' }, c.license) : c.license,
              c.via ? ` · via ${c.via}` : null,
            ),
          ),
        ),
      ),
    ),
  );
  MODELS.forEach(renderSlot);
  renderSummary();
  refreshAll();
  return () => {
    disposed = true;
  };
}
