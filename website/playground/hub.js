import { DEMOS } from './webgpu/src/config/catalog.js';
import { h } from './webgpu/src/lib/dom.js';
import { appIcon } from './webgpu/src/ui/app-icons.js';
import { illustration } from './webgpu/src/ui/illustrations.js';

// The card is taller than the 16:9 art, so show the whole scene instead of cropping it.
for (const el of document.querySelectorAll('[data-art]')) {
  const art = illustration(el.dataset.art);
  art.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  el.append(art);
}

for (const list of document.querySelectorAll('[data-demos]')) {
  for (const id of list.dataset.demos.split(' ')) {
    const demo = DEMOS.find((d) => d.id === id);
    if (demo) list.append(h('li', { class: 'track__demo' }, appIcon(id, 22), demo.title));
  }
}
