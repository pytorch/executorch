import { h, replace } from '../lib/dom.js';
import { isOnDevice } from '../runtime/model-storage.js';
import { icon } from './icons.js';

// Shown on a load screen once the model's files are known to be in this browser;
// it also relabels the screen's primary button, since nothing will download.
export function onDeviceNote(model) {
  const note = h('p', { class: 'on-device', hidden: true });
  isOnDevice(model).then((stored) => {
    if (!stored || !note.isConnected) return;
    replace(note, icon('check', 14), 'On this device · no download needed');
    note.hidden = false;
    const button = note.parentElement?.querySelector('.btn--primary');
    const label = [...(button?.childNodes ?? [])].reverse().find((n) => n.nodeType === Node.TEXT_NODE && n.textContent.trim());
    if (label) label.textContent = 'Start';
  });
  return note;
}
