import { createStore } from './store.js';

// Follows the OS until the visitor picks a side, then persists their choice.
const KEY = 'etwg-theme';
const media = matchMedia('(prefers-color-scheme: dark)');

function readSaved() {
  try {
    const v = localStorage.getItem(KEY);
    return v === 'dark' || v === 'light' ? v : null;
  } catch {
    return null;
  }
}

export const theme = createStore({ value: readSaved() ?? (media.matches ? 'dark' : 'light'), explicit: readSaved() !== null });

theme.subscribe(({ value }) => {
  document.documentElement.dataset.theme = value;
});

media.addEventListener('change', (e) => {
  if (!theme.get().explicit) theme.set({ value: e.matches ? 'dark' : 'light' });
});

export function toggleTheme() {
  const value = theme.get().value === 'dark' ? 'light' : 'dark';
  try {
    localStorage.setItem(KEY, value);
  } catch {
    // Storage can be unavailable (private mode); the choice still applies for this visit.
  }
  theme.set({ value, explicit: true });
}
