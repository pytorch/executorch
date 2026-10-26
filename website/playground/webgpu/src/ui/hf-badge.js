import { h } from '../lib/dom.js';
import { icon } from './icons.js';

// Shows where a hosted model downloads from: the Hugging Face repo address, linked.
export function hfBadge(source) {
  if (source?.kind !== 'remote' || !source.repoUrl) return null;
  const repo = source.repoUrl.replace('https://huggingface.co/', '');
  return h(
    'a',
    { class: 'hf-badge', href: source.repoUrl, target: '_blank', rel: 'noopener', title: `Open ${repo} on Hugging Face` },
    h('img', { class: 'hf-badge__logo', src: './assets/huggingface-logo.svg', alt: '', width: 22, height: 20 }),
    h('span', { class: 'hf-badge__text' }, h('span', { class: 'hf-badge__label' }, 'Downloads from Hugging Face'), h('span', { class: 'hf-badge__repo' }, `huggingface.co/${repo}`)),
    icon('external', 13),
  );
}
