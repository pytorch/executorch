// The camera effects. Each is the body of one fragment shader: `effect(uv)`
// returns the output color, using video(uv), maskAt(uv) (1 = person) and the
// helpers declared in effects-gl.js.
export const EFFECTS = [
  {
    id: 'original',
    label: 'Original',
    glsl: `vec3 effect(vec2 uv) { return video(uv); }`,
  },
  {
    id: 'bokeh',
    label: 'Portrait blur',
    // A golden-angle disk blur that skips samples landing on the person, so the
    // subject does not smear into the background around its edges.
    glsl: `vec3 effect(vec2 uv) {
  float m = maskAt(uv);
  vec2 r = vec2(uSrcRes.y / uSrcRes.x, 1.0) * 0.028;
  vec3 acc = vec3(0.0);
  float wsum = 0.0;
  for (int i = 0; i < 40; i++) {
    float f = float(i) + 0.5;
    float a = f * 2.39996;
    vec2 o = vec2(cos(a), sin(a)) * sqrt(f / 40.0) * r;
    vec3 c = video(uv + o);
    float w = (1.0 - rawMask(uv + o) * 0.95) * (1.0 + 2.0 * pow(dot(c, LUMA), 4.0));
    acc += c * w;
    wsum += w;
  }
  vec3 bg = acc / max(wsum, 1e-3);
  return mix(bg, video(uv), m);
}`,
  },
  {
    id: 'pixelate',
    label: 'Pixelate background',
    glsl: `vec3 effect(vec2 uv) {
  float block = max(6.0, uRes.y / 24.0);
  vec2 cell = (floor(uv * uRes / block) + 0.5) * block / uRes;
  return mix(video(cell), video(uv), maskAt(uv));
}`,
  },
  {
    id: 'neon',
    label: 'TV waves',
    glsl: `vec3 effect(vec2 uv) {
  vec2 w = uv + vec2(sin(uv.y * 38.0 + uTime * 5.0) * 0.006 + sin(uv.y * 7.0 - uTime * 1.3) * 0.004, 0.0);
  float m = maskAt(w);
  float lum = dot(video(w), LUMA);
  vec3 neon = mix(vec3(0.05, 0.95, 1.0), vec3(1.0, 0.18, 0.85), clamp(uv.y + 0.25 * sin(uTime * 0.8 + uv.x * 3.0), 0.0, 1.0));
  float scan = 0.6 + 0.4 * sin(uv.y * uRes.y * 1.15 - uTime * 9.0);
  float edge = clamp(m * (1.0 - m) * 4.0, 0.0, 1.0);
  vec3 person = neon * (0.25 + 1.5 * lum) * scan;
  vec3 bg = vec3(0.02, 0.02, 0.05) * (0.7 + 0.3 * sin(uv.y * uRes.y * 0.5));
  return mix(bg, person, m) + neon * edge * 1.4;
}`,
  },
  {
    id: 'colorpop',
    label: 'Color pop',
    glsl: `vec3 effect(vec2 uv) {
  vec3 c = video(uv);
  vec3 g = vec3(dot(c, LUMA)) * 0.92;
  return mix(g, c, maskAt(uv));
}`,
  },
  {
    id: 'virtual',
    label: 'Virtual background',
    glsl: `vec3 effect(vec2 uv) {
  vec3 bg;
  if (uHasBg > 0.5) {
    float s = max(uRes.x / uBgRes.x, uRes.y / uBgRes.y);
    vec2 bgUv = (uv * uRes - 0.5 * (uRes - uBgRes * s)) / (uBgRes * s);
    bg = texture(uBg, bgUv).rgb;
  } else {
    bg = gradientBg(uv);
  }
  return mix(bg, video(uv), maskAt(uv));
}`,
  },
];

const hex = (s) => [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16) / 255);

export const BACKDROPS = [
  { id: 'sunset', label: 'Sunset', stops: ['#ffb36b', '#ff6a88', '#7b4bd6'] },
  { id: 'ocean', label: 'Ocean', stops: ['#5ee7df', '#2a78d6', '#1b2a6b'] },
  { id: 'mint', label: 'Mint', stops: ['#f1fff8', '#a8edea', '#5fbf9f'] },
  { id: 'studio', label: 'Studio', stops: ['#3a4256', '#232a3a', '#121620'] },
].map((b) => ({ ...b, rgb: b.stops.map(hex) }));
