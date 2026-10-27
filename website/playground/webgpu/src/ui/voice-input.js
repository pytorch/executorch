// Mic button for the chat composer: records, transcribes with Whisper on the
// GPU, and hands the text to the composer. Hidden when no speech model resolves.
import { recordMic } from '../audio/mic.js';
import { h, replace, svg } from '../lib/dom.js';
import { isWhisperAvailable, isWhisperLoaded, loadWhisper } from '../runtime/whisper.js';

const MIC = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>';

export function attachVoiceInput(slot, { onText } = {}) {
  let phase = 'idle'; // idle | loading | recording | transcribing
  let recording = null;
  let disposed = false;
  const abort = new AbortController();
  const status = h('span', { class: 'voice__status', 'aria-live': 'polite' });
  const btn = h('button', { class: 'voice-btn', type: 'button', 'aria-label': 'Speak your message', title: 'Speak your message (Whisper, on your GPU)', onclick: onClick });

  function render(text = '') {
    if (disposed) return;
    btn.classList.toggle('voice-btn--rec', phase === 'recording');
    btn.classList.toggle('voice-btn--busy', phase === 'loading' || phase === 'transcribing');
    btn.setAttribute('aria-label', phase === 'recording' ? 'Stop and transcribe' : 'Speak your message');
    replace(btn, phase === 'recording' ? h('span', { class: 'voice-btn__stop' }) : svg(MIC));
    status.textContent = text;
  }

  async function onClick() {
    if (phase === 'recording') {
      phase = 'transcribing';
      render('Transcribing…');
      try {
        const pcm = await recording.stop();
        recording = null;
        const asr = await loadWhisper({ signal: abort.signal });
        const { text } = await asr.transcribe(pcm, { signal: abort.signal });
        if (text.trim()) onText?.(text.trim());
        phase = 'idle';
        render(text.trim() ? '' : 'Didn’t catch that, try again');
      } catch (err) {
        phase = 'idle';
        if (err.name !== 'AbortError') render(err.message);
      }
      return;
    }
    if (phase !== 'idle') return;
    try {
      if (!isWhisperLoaded()) {
        phase = 'loading';
        render('Loading speech model…');
        await loadWhisper({
          signal: abort.signal,
          onProgress: (p) => {
            if (p.stage === 'download' && p.total) render(`Loading speech model ${Math.round((100 * p.loaded) / p.total)}%`);
          },
        });
      }
      recording = await recordMic({ maxSeconds: 30, onAutoStop: () => onClick() });
      phase = 'recording';
      render('Listening… click to stop');
    } catch (err) {
      phase = 'idle';
      if (err.name !== 'AbortError') render(err.message);
    }
  }

  isWhisperAvailable().then((ok) => {
    if (!ok || disposed) return;
    replace(slot, status, btn);
    render();
  });

  return () => {
    disposed = true;
    abort.abort();
    recording?.cancel();
  };
}
