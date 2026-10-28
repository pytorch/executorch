// Sample media, served from this site (sources and licenses in config/photo-credits.js,
// also assets/samples/CREDITS.md), and the pinned tokenizer library.
import { TOKENIZER_LIB } from './runtime.js';
const at = (path) => new URL(`../../${path}`, import.meta.url).href;
const sample = (file, label) => ({ path: at(`assets/samples/${file}`), label });

export const SHARED = Object.freeze({
  tokenizerLib: TOKENIZER_LIB,
  visionSamples: {
    depth: [sample('city.jpg', 'City at night'), sample('giraffes.jpg', 'Giraffes'), sample('husky.jpg', 'Husky'), sample('astronaut.jpg', 'Astronaut')],
    detect: [sample('street.jpg', 'Street'), sample('table.jpg', 'Table'), sample('cat.jpg', 'Cat')],
    bgremove: [sample('astronaut.jpg', 'Astronaut'), sample('esa-astronaut.jpg', 'ESA astronaut'), sample('skier.jpg', 'Skier')],
    superres: [sample('husky.jpg', 'Husky'), sample('butterfly.jpg', 'Butterfly'), sample('chihuahuas.jpg', 'Chihuahuas'), sample('giraffes.jpg', 'Giraffes')],
    effects: [sample('astronaut.jpg', 'Astronaut'), sample('esa-astronaut.jpg', 'ESA astronaut'), sample('skier.jpg', 'Skier')],
  },
  segmentSamples: [sample('chihuahuas.jpg', 'Chihuahuas'), sample('city.jpg', 'City at night'), sample('street.jpg', 'Street'), sample('astronaut.jpg', 'Astronaut')],
  vlmSamples: [sample('astronaut.jpg', 'Astronaut'), sample('receipt.jpg', 'Receipt'), sample('city.jpg', 'City street'), sample('husky.jpg', 'Husky')],
});
