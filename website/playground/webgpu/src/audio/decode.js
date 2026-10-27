// Decodes an uploaded audio file in the browser's own decoder, straight to
// 16 kHz (decodeAudioData resamples to the context's rate).
import { downmix, SPEECH_RATE } from './pcm.js';

export const MAX_FILE_SECONDS = 600;

export async function decodeAudioFile(fileOrBuffer) {
  const bytes = fileOrBuffer instanceof ArrayBuffer ? fileOrBuffer : await fileOrBuffer.arrayBuffer();
  const Ctx = globalThis.OfflineAudioContext ?? globalThis.webkitOfflineAudioContext;
  if (!Ctx) throw new Error('This browser cannot decode audio files.');
  let buffer;
  try {
    buffer = await new Ctx(1, 1, SPEECH_RATE).decodeAudioData(bytes);
  } catch {
    throw new Error('This file could not be decoded. Try WAV, MP3, M4A, OGG or WebM audio.');
  }
  if (buffer.duration > MAX_FILE_SECONDS) {
    throw new Error(`This clip is ${Math.round(buffer.duration / 60)} minutes long; the demo transcribes up to ${MAX_FILE_SECONDS / 60} minutes.`);
  }
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, i) => buffer.getChannelData(i));
  return downmix(channels);
}
