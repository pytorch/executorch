import { LISTED_CATEGORIES, demosIn } from '../config/catalog.js';
import { MODELS } from '../config/models.js';
import { FAILURE_COPY } from '../gpu/adapter.js';
import { gpuState } from '../gpu/context.js';
import { h, replace } from '../lib/dom.js';
import { formatBytes } from '../lib/format.js';
import { theme, toggleTheme } from '../lib/theme.js';
import { storedFiles } from '../runtime/model-storage.js';
import { appIcon } from './app-icons.js';
import { categoryIcon } from './category-icons.js';
import { icon } from './icons.js';

// Browsers report terse adapter strings ("apple" / "metal-3"); show them the way
// a person would say them.
const VENDORS = { apple: 'Apple', nvidia: 'NVIDIA', amd: 'AMD', intel: 'Intel', qualcomm: 'Qualcomm', arm: 'Arm', google: 'Google', microsoft: 'Microsoft' };
const titleCase = (s) => s.replace(/[-_]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

export function friendlyGpu(info) {
  if (!info) return { name: 'Your GPU', detail: '' };
  const vendor = VENDORS[info.vendor?.toLowerCase()] ?? (info.vendor ? titleCase(info.vendor) : '');
  const arch = info.architecture ? titleCase(info.architecture).replace(/^Metal (\d)/, 'Metal $1') : '';
  if (info.isFallbackAdapter || /swiftshader/i.test(info.architecture ?? '')) return { name: 'Software renderer', detail: 'SwiftShader · slow, for testing' };
  return { name: vendor ? `${vendor} GPU` : info.description || 'Your GPU', detail: arch || info.description || '' };
}

function adapterChip() {
  const glyph = h('span', { class: 'adapter__glyph' }, icon('chip', 18));
  const label = h('span', { class: 'adapter__label' }, 'Detecting your GPU…');
  const sub = h('span', { class: 'adapter__sub' }, 'WebGPU');
  const status = h('span', { class: 'adapter__status' });
  const chip = h('a', { class: 'adapter adapter--probing', href: '#/examples', title: 'Detecting GPU' }, glyph, h('span', { class: 'adapter__text' }, label, sub), status);
  gpuState.subscribe((s) => {
    chip.classList.remove('adapter--probing', 'adapter--ready', 'adapter--unavailable', 'adapter--lost');
    if (s.status === 'ready') {
      chip.classList.add('adapter--ready');
      const g = friendlyGpu(s.info);
      label.textContent = g.name;
      sub.textContent = g.detail ? `${g.detail} · WebGPU ready` : 'WebGPU ready';
      replace(status, icon('check', 14));
      const fields = Object.entries(s.info ?? {}).filter(([, v]) => v !== '' && v !== false);
      chip.title = fields.length ? fields.map(([k, v]) => `${k}: ${v}`).join('\n') : 'The browser did not report adapter details';
    } else if (s.status === 'probing') {
      chip.classList.add('adapter--probing');
      label.textContent = 'Detecting your GPU…';
      replace(status);
    } else {
      chip.classList.add(s.status === 'lost' ? 'adapter--lost' : 'adapter--unavailable');
      label.textContent = FAILURE_COPY[s.reason]?.chip ?? 'No WebGPU';
      sub.textContent = 'Tap for how to fix it';
      replace(status, icon('alert', 14));
      chip.title = FAILURE_COPY[s.reason]?.body ?? '';
    }
  });
  return chip;
}

export function createSidebar({ onToggleCollapse, onNavigate }) {
  const navLink = (href, iconName, text, key) =>
    h('a', { class: 'nav__item', href, dataset: { nav: key }, onclick: onNavigate }, icon(iconName, 18), h('span', { class: 'nav__text' }, text));

  const storageBadge = h('span', { class: 'nav__badge' });
  const refreshStorage = () =>
    Promise.all(MODELS.map((m) => storedFiles(m).catch(() => ({ bytes: 0 })))).then((all) => {
      const bytes = all.reduce((a, s) => a + s.bytes, 0);
      storageBadge.textContent = bytes ? formatBytes(bytes) : '';
    });

  const themeLabel = h('span', { class: 'nav__text' });
  const themeIcon = h('span', { class: 'theme-btn__icon' });
  const themeBtn = h('button', { class: 'theme-btn', type: 'button', onclick: toggleTheme }, themeIcon, themeLabel);
  theme.subscribe(({ value }) => {
    replace(themeIcon, icon(value === 'dark' ? 'sun' : 'moon', 18));
    themeLabel.textContent = value === 'dark' ? 'Light mode' : 'Dark mode';
    themeBtn.setAttribute('aria-label', `Switch to ${value === 'dark' ? 'light' : 'dark'} mode`);
  });

  const sidebar = h(
    'aside',
    { class: 'sidebar', 'aria-label': 'Primary' },
    h(
      'div',
      { class: 'sidebar__head' },
      h(
        'a',
        { class: 'brand', href: '#/examples', onclick: onNavigate },
        h('img', { class: 'brand__logo', src: './assets/executorch-chip-logo.svg', alt: '', width: 34, height: 34 }),
        h(
          'span',
          { class: 'brand__text' },
          h('span', { class: 'brand__name' }, 'WebGPU'),
          h('span', { class: 'brand__product' }, 'ExecuTorch Playground'),
        ),
      ),
      h('button', { class: 'icon-btn sidebar__collapse', type: 'button', 'aria-label': 'Collapse sidebar', onclick: onToggleCollapse }, icon('sidebar', 18)),
    ),
    adapterChip(),
    h(
      'div',
      { class: 'privacy-card privacy-card--compact', title: 'Models run on your GPU. What you type, say, or show never leaves this tab.' },
      h('span', { class: 'privacy-card__icon' }, icon('shield', 15)),
      h('span', { class: 'nav__text privacy-card__text' }, h('strong', null, 'Private by design'), 'Runs on your GPU. Nothing leaves this tab.'),
    ),
    h(
      'nav',
      { class: 'nav' },
      navLink('#/examples', 'grid', 'All examples', 'examples'),
      LISTED_CATEGORIES.map((c) =>
        h(
          'div',
          { class: 'nav__group' },
          h('a', { class: 'nav__group-label', href: `#/examples/${c.id}`, dataset: { nav: `examples/${c.id}` }, onclick: onNavigate }, categoryIcon(c.id, 13), h('span', { class: 'nav__text' }, c.label)),
          demosIn(c.id).map((d) =>
            h(
              'a',
              { class: 'nav__demo', href: `#/demo/${d.id}`, dataset: { nav: `demo/${d.id}` }, onclick: onNavigate, title: d.title },
              appIcon(d.id, 22),
              h('span', { class: 'nav__text' }, d.title),
            ),
          ),
        ),
      ),
      h('div', { class: 'nav__rule' }),
      h('a', { class: 'nav__item', href: '#/models', dataset: { nav: 'models' }, onclick: onNavigate }, icon('layers', 18), h('span', { class: 'nav__text' }, 'Models'), storageBadge),
    ),
    h(
      'div',
      { class: 'sidebar__foot' },
      h(
        'div',
        { class: 'help-card' },
        h('span', { class: 'nav__text help-card__title' }, 'Found a bug or have feedback?'),
        h(
          'a',
          { class: 'help-card__link', href: 'https://github.com/pytorch/executorch/issues', target: '_blank', rel: 'noopener', title: 'Report on GitHub' },
          icon('bug', 16),
          h('span', { class: 'nav__text' }, 'Report on GitHub'),
          h('span', { class: 'nav__text help-card__ext' }, icon('external', 13)),
        ),
        h(
          'a',
          { class: 'help-card__link', href: 'https://discord.gg/GbraDsuqCW', target: '_blank', rel: 'noopener', title: 'Chat with us on Discord' },
          icon('chat', 16),
          h('span', { class: 'nav__text' }, 'Chat with us on Discord'),
          h('span', { class: 'nav__text help-card__ext' }, icon('external', 13)),
        ),
      ),
      themeBtn,
    ),
  );

  sidebar.setActive = (route) => {
    const current = route.name === 'demo' ? `demo/${route.id}` : route.nav;
    for (const a of sidebar.querySelectorAll('[data-nav]')) a.classList.toggle('is-active', a.dataset.nav === current);
    refreshStorage();
  };
  return sidebar;
}
