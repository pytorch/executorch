// The catalog table. Tiles, routes, and compute badges are all generated from
// this one list, so a badge can never disagree with what a demo actually does.
import { getModel, isPublished } from './models.js';

export const COMPUTE = Object.freeze({
  EXECUTORCH: 'executorch',
});

export const COMPUTE_BADGES = Object.freeze({
  [COMPUTE.EXECUTORCH]: {
    label: 'ExecuTorch · real model',
    short: 'ExecuTorch',
    description: 'A real ExecuTorch .pte runs on the WebGPU backend in this tab. Nothing is sent to a server.',
  },
});

export const CATEGORIES = [
  { id: 'language', label: 'Language', blurb: 'Chat with a model running on your GPU.' },
  { id: 'vision', label: 'Vision', blurb: 'Pixels in, structure out. Works on your camera, live.' },
  { id: 'generative', label: 'Generative', blurb: 'Images made and remade in the tab.' },
  { id: 'audio', label: 'Audio', blurb: 'Speech to text, without a server.' },
];

export const DEMOS = [
  {
    id: 'chat',
    category: 'language',
    title: 'On-device Chat',
    blurb: 'Chat with a real language model running entirely on your GPU. Talk to it, too.',
    meta: { label: 'Models', value: 'Qwen 3 0.6B, Llama 3.2 1B' },
    models: ['qwen3-0.6b', 'llama-3.2-1b-instruct'],
    compute: COMPUTE.EXECUTORCH,
    view: 'chat',
    art: 'chat',
  },
  {
    id: 'effects',
    category: 'vision',
    title: 'Camera Effects',
    blurb: 'Your camera, six ways at once: portrait blur, pixelated or swapped backgrounds, neon scanlines, color pop.',
    meta: { label: 'Model', value: 'MODNet person segmentation · live camera' },
    models: ['modnet'],
    compute: COMPUTE.EXECUTORCH,
    view: 'effects',
    art: 'effects',
    featured: true,
  },
  {
    id: 'segment',
    category: 'vision',
    title: 'Segment Anything',
    blurb: 'Click any object and get a pixel-accurate mask back in milliseconds.',
    meta: { label: 'Model', value: 'SAM 2.1 Hiera Tiny' },
    models: ['sam2-hiera-tiny'],
    compute: COMPUTE.EXECUTORCH,
    view: 'segment',
    art: 'segment',
  },
  {
    id: 'vlm',
    unlisted: 'The public WebGPU runtime has no 8-bit linear kernel yet, and the 4-bit export loses too much answer quality.',
    category: 'vision',
    title: 'VLM Q&A',
    blurb: 'Show it a picture and ask about it. The image never leaves your device.',
    meta: { label: 'Model', value: 'SmolVLM-500M · photo or camera' },
    models: ['smolvlm-500m-instruct'],
    compute: COMPUTE.EXECUTORCH,
    view: 'vlm',
    art: 'vlm',
  },
  {
    id: 'depth',
    category: 'vision',
    title: 'Depth Estimation',
    blurb: 'See the third dimension: a depth map for any photo, or your camera in real time.',
    meta: { label: 'Model', value: 'Depth Anything V2 Small · photo or camera' },
    models: ['depth-anything-v2-small'],
    compute: COMPUTE.EXECUTORCH,
    view: 'vision',
    task: 'depth',
    art: 'depth',
  },
  {
    id: 'detect',
    category: 'vision',
    title: 'Object Detection',
    blurb: 'Boxes and labels for 80 everyday objects, on a photo or a live camera feed.',
    meta: { label: 'Model', value: 'D-FINE S · photo, video, or camera' },
    models: ['dfine-small'],
    compute: COMPUTE.EXECUTORCH,
    view: 'vision',
    task: 'detect',
    art: 'detect',
  },
  {
    id: 'bgremove',
    category: 'generative',
    title: 'Background Removal',
    blurb: 'Cut people out of a photo, or swap your camera background, without a green screen.',
    meta: { label: 'Model', value: 'MODNet · photo or camera' },
    models: ['modnet'],
    compute: COMPUTE.EXECUTORCH,
    view: 'vision',
    task: 'bgremove',
    art: 'bgremove',
  },
  {
    id: 'superres',
    category: 'generative',
    title: 'Super-Resolution',
    blurb: 'Upscale a small image 4× and recover detail, with a before-and-after slider.',
    meta: { label: 'Model', value: 'Real-ESRGAN ×4' },
    models: ['real-esrgan'],
    compute: COMPUTE.EXECUTORCH,
    view: 'vision',
    task: 'superres',
    art: 'superres',
  },
  {
    id: 'speech-to-text',
    category: 'audio',
    title: 'Speech to Text',
    blurb: 'Record your voice or drop in an audio file and get a transcript, decoded entirely in the tab.',
    meta: { label: 'Model', value: 'Whisper base' },
    models: ['whisper-base'],
    compute: COMPUTE.EXECUTORCH,
    view: 'stt',
    art: 'stt',
  },
];

// Visitors see a demo only when it is listed and every model it needs is published.
export const LISTED_DEMOS = DEMOS.filter((d) => !d.unlisted && d.models.every((id) => isPublished(getModel(id))));
export const LISTED_CATEGORIES = CATEGORIES.filter((c) => LISTED_DEMOS.some((d) => d.category === c.id));

const BY_ID = new Map(LISTED_DEMOS.map((d) => [d.id, d]));

export function getDemo(id) {
  return BY_ID.get(id) ?? null;
}

export function computeBadge(demo) {
  const badge = COMPUTE_BADGES[demo.compute];
  if (!badge) throw new Error(`Demo ${demo.id} has unknown compute kind: ${demo.compute}`);
  return badge;
}

export function demosIn(categoryId) {
  return LISTED_DEMOS.filter((d) => d.category === categoryId);
}
