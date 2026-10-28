import { getDemo } from './config/catalog.js';
import { initGPU } from './gpu/context.js';
import { h, replace } from './lib/dom.js';
import './lib/theme.js';
import { mountCatalog } from './ui/catalog.js';
import { mountChat } from './ui/demo-chat.js';
import { mountEffects } from './ui/demo-effects.js';
import { mountSegment } from './ui/demo-segment.js';
import { mountStt } from './ui/demo-stt.js';
import { mountVision } from './ui/demo-vision.js';
import { mountVlm } from './ui/demo-vlm.js';
import { mountDiagnosticsBanner } from './ui/diagnostics-banner.js';
import { icon } from './ui/icons.js';
import { mountModels } from './ui/models-page.js';
import { createSidebar } from './ui/sidebar.js';

const DEMO_VIEWS = {
  chat: mountChat,
  vision: mountVision,
  stt: mountStt,
  vlm: mountVlm,
  segment: mountSegment,
  effects: mountEffects,
};

// Hash routing keeps the site a folder of static files that works under any
// subpath (e.g. /examples/) without server rewrites.
function parseRoute() {
  const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  if (parts[0] === 'demo' && parts[1]) return { name: 'demo', id: parts[1], nav: 'demo' };
  if (parts[0] === 'models') return { name: 'models', nav: 'models' };
  if (parts[0] === 'examples' && parts[1]) return { name: 'catalog', category: parts[1], nav: `examples/${parts[1]}` };
  return { name: 'catalog', nav: 'examples' };
}

const COLLAPSE_KEY = 'etwg-sidebar-collapsed';
const app = document.getElementById('app');
const main = h('main', { class: 'main', id: 'main', tabindex: '-1' });
const banner = h('div', { class: 'diag-stack', 'aria-live': 'assertive' });
const view = h('div', { class: 'view' });

function setCollapsed(collapsed) {
  app.classList.toggle('app--collapsed', collapsed);
  try {
    localStorage.setItem(COLLAPSE_KEY, collapsed ? '1' : '0');
  } catch {
    // Non-persistent is fine.
  }
}

const sidebar = createSidebar({
  onToggleCollapse: () => setCollapsed(!app.classList.contains('app--collapsed')),
  onNavigate: () => app.classList.remove('app--drawer-open'),
});
const mobileBar = h(
  'div',
  { class: 'mobilebar' },
  h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Open navigation', onclick: () => app.classList.add('app--drawer-open') }, icon('menu', 20)),
  h('span', { class: 'mobilebar__title' }, 'WebGPU Playground'),
);
const scrim = h('div', { class: 'scrim', onclick: () => app.classList.remove('app--drawer-open') });

main.append(mobileBar, banner, view);
replace(app, sidebar, scrim, main);
try {
  if (localStorage.getItem(COLLAPSE_KEY) === '1') app.classList.add('app--collapsed');
} catch {
  // Storage unavailable: start expanded.
}
mountDiagnosticsBanner(banner);

let cleanup = null;
let lastKey = '';
function render() {
  const route = parseRoute();
  const key = route.name === 'demo' ? `demo/${route.id}` : route.name;
  sidebar.setActive(route);
  if (key === lastKey && route.name === 'catalog') {
    if (route.category) view.querySelector(`#band-${route.category}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    else window.scrollTo({ top: 0, behavior: 'smooth' });
    return;
  }
  lastKey = key;
  cleanup?.();
  cleanup = null;
  window.scrollTo({ top: 0 });

  if (route.name === 'demo') {
    const demo = getDemo(route.id);
    if (!demo) {
      replace(view, h('div', { class: 'page' }, h('h1', { class: 'page-title' }, 'Not found'), h('p', null, h('a', { href: '#/examples' }, 'Back to Examples'))));
      return;
    }
    document.title = `${demo.title} · ExecuTorch Playground`;
    cleanup = DEMO_VIEWS[demo.view](view, demo);
  } else if (route.name === 'models') {
    document.title = 'Models · ExecuTorch Playground';
    cleanup = mountModels(view);
  } else {
    document.title = 'WebGPU · ExecuTorch Playground';
    cleanup = mountCatalog(view, { category: route.category });
  }
}

window.addEventListener('hashchange', render);
render();
initGPU();
