import { svg } from '../lib/dom.js';

const PATHS = {
  language: '<path d="M4 5.5A1.5 1.5 0 0 1 5.5 4h13A1.5 1.5 0 0 1 20 5.5v9a1.5 1.5 0 0 1-1.5 1.5H10l-4.5 4v-4h0A1.5 1.5 0 0 1 4 14.5v-9Z"/><path d="M8 8.5h8M8 12h5"/>',
  vision: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z"/><circle cx="12" cy="12" r="3"/>',
  generative: '<path d="M11 4Q11.6 10.4 18 11 11.6 11.6 11 18 10.4 11.6 4 11 10.4 10.4 11 4Z"/><path d="M18.5 16v4M16.5 18h4"/>',
  audio: '<path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 10.5v3"/>',
  training: '<path d="M4 4v16h16"/><path d="m7.5 15 3.5-4 3 2.5 5-6.5"/>',
};

export function categoryIcon(categoryId, size = 16) {
  return svg(
    `<svg class="icon cat-icon" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PATHS[categoryId] ?? ''}</svg>`,
  );
}
