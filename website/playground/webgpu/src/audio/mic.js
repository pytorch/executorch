// Microphone capture to raw PCM (no MediaRecorder, so nothing is encoded and
// decoded again), resampled to 16 kHz mono when stopped.
import { concat, resample, SPEECH_RATE } from './pcm.js';

export class MicError extends Error {
  // kind: 'unsupported' | 'insecure' | 'denied' | 'no-device' | 'busy' | 'failed'
  constructor(kind, message) {
    super(message);
    this.name = 'MicError';
    this.kind = kind;
  }
}

function toMicError(err) {
  switch (err?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return new MicError('denied', 'Microphone access was blocked. Allow it in your browser’s site settings, or upload a file instead.');
    case 'NotFoundError':
    case 'OverconstrainedError':
      return new MicError('no-device', 'No microphone was found. Connect one, or upload a file instead.');
    case 'NotReadableError':
    case 'AbortError':
      return new MicError('busy', 'The microphone is in use by another app or could not be started.');
    default:
      return new MicError('failed', `Could not start the microphone: ${err?.message ?? err}`);
  }
}

// Batches 128-frame render quanta into ~40 ms messages.
const TAP_SOURCE = `
class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(2048); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      if (this.n + ch.length > this.buf.length) { this.port.postMessage(this.buf.slice(0, this.n)); this.n = 0; }
      this.buf.set(ch, this.n); this.n += ch.length;
    }
    return true;
  }
}
registerProcessor('pcm-tap', Tap);
`;

async function tapNode(ctx, onChunk) {
  if (ctx.audioWorklet && typeof AudioWorkletNode === 'function') {
    const url = URL.createObjectURL(new Blob([TAP_SOURCE], { type: 'text/javascript' }));
    try {
      await ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    const node = new AudioWorkletNode(ctx, 'pcm-tap', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    node.port.onmessage = (e) => onChunk(e.data);
    return node;
  }
  const node = ctx.createScriptProcessor(2048, 1, 1);
  node.onaudioprocess = (e) => onChunk(Float32Array.from(e.inputBuffer.getChannelData(0)));
  return node;
}

export function micSupport() {
  if (typeof window !== 'undefined' && !window.isSecureContext) return 'insecure';
  if (!globalThis.navigator?.mediaDevices?.getUserMedia) return 'unsupported';
  return 'ok';
}

/**
 * Starts recording from the default microphone.
 * onLevel({ rms, peak, seconds }) fires ~25×/s. Recording stops by itself at
 * `maxSeconds` (onAutoStop is then called; stop() still returns the audio).
 * Returns { stop(): Promise<Float32Array> (16 kHz mono), cancel(), seconds }.
 * Throws MicError.
 */
export async function recordMic({ onLevel, maxSeconds = 300, onAutoStop } = {}) {
  const support = micSupport();
  if (support === 'insecure') throw new MicError('insecure', 'The microphone needs a secure page (https or localhost).');
  if (support === 'unsupported') throw new MicError('unsupported', 'This browser does not expose a microphone API.');

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    throw toMicError(err);
  }

  const ctx = new AudioContext();
  const chunks = [];
  let frames = 0;
  let capturing = true;
  let release = null;
  const done = new Promise((r) => (release = r));

  const teardown = async () => {
    capturing = false;
    for (const t of stream.getTracks()) t.stop();
    try {
      await ctx.close();
    } catch {
      // Already closed.
    }
    release();
  };

  try {
    const node = await tapNode(ctx, (chunk) => {
      if (!capturing) return;
      chunks.push(chunk);
      frames += chunk.length;
      let s = 0;
      let p = 0;
      for (let i = 0; i < chunk.length; i++) {
        const x = chunk[i];
        s += x * x;
        if (Math.abs(x) > p) p = Math.abs(x);
      }
      const seconds = frames / ctx.sampleRate;
      onLevel?.({ rms: Math.sqrt(s / chunk.length), peak: p, seconds });
      if (seconds >= maxSeconds) {
        capturing = false;
        onAutoStop?.();
      }
    });
    ctx.createMediaStreamSource(stream).connect(node);
    // The node writes no output; the connection only keeps it scheduled.
    node.connect(ctx.destination);
    await ctx.resume();
  } catch (err) {
    await teardown();
    throw new MicError('failed', `Could not start audio capture: ${err.message}`);
  }

  let stopped = null;
  return {
    get seconds() {
      return frames / ctx.sampleRate;
    },
    stop() {
      stopped ??= (async () => {
        const rate = ctx.sampleRate;
        await teardown();
        return resample(concat(chunks), rate, SPEECH_RATE);
      })();
      return stopped;
    },
    cancel() {
      if (!stopped) stopped = teardown().then(() => new Float32Array(0));
      return done;
    },
  };
}
