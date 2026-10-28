import { mulberry32 } from '../lib/prng.js';
import { svg, uid } from '../lib/dom.js';

// Hand-built 16:9 illustrations: a browser window with a mini product UI on a
// pastel field, plus a floating panel where it helps. Theme-dependent colors
// are CSS variables (`--art-*`, set per `data-cat` in app.css), so one markup
// serves light and dark; "photo" content keeps fixed colors in both themes.
// Every gradient/clip id is suffixed per instance, so the same art can appear
// several times on one page.
//
// Cards crop the art to ~16:8.2 (about 10px off top and bottom), and the
// compute badge covers roughly x > 272, y < 54 on narrow cards: keep key shapes
// out of both.
const W = 400;
const H = 225;

const C = {
  blue: '#4285F4', blue2: '#8AB4F8',
  green: '#34A853', green2: '#81C995',
  yellow: '#F9AB00', yellow2: '#FDD663',
  coral: '#EA4335', coral2: '#F28B82',
  purple: '#A142F4', purple2: '#C58AF9',
  skin: '#F2C29E', hair: '#4A3B36', scrim: '#202124',
};
const CARD = 'var(--art-card)';
const INK = 'var(--art-ink)';
const LINE = 'var(--art-line)';
const FIELD = 'var(--art-field)';
const FONT = 'font-family="Open Sans, Helvetica, Arial, sans-serif"';
const MONO = 'font-family="ui-monospace, SF Mono, Menlo, Consolas, monospace"';

const CATEGORY = {
  chat: 'language',
  effects: 'vision', segment: 'vision', vlm: 'vision', depth: 'vision', detect: 'vision',
  cppn: 'generative', bgremove: 'generative', superres: 'generative',
  stt: 'audio',
  trainer: 'training', kernels: 'training',
};

function frame(id, name, blobs, body) {
  return `
<svg class="art" data-cat="${CATEGORY[name]}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid slice" role="img" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg-${id}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="var(--art-bg-1)"/><stop offset="1" stop-color="var(--art-bg-2)"/></linearGradient>
  </defs>
  <rect width="${W}" height="${H}" fill="url(#bg-${id})"/>
  <g fill="var(--art-blob)">${blobs}</g>
  ${body}
</svg>`;
}

// Rounded rect path with per-corner radii [tl, tr, br, bl].
function rr(x, y, w, h, r) {
  const [tl, tr, br, bl] = Array.isArray(r) ? r : [r, r, r, r];
  return `M${x + tl} ${y}H${x + w - tr}Q${x + w} ${y} ${x + w} ${y + tr}V${y + h - br}Q${x + w} ${y + h} ${x + w - br} ${y + h}H${x + bl}Q${x} ${y + h} ${x} ${y + h - bl}V${y + tl}Q${x} ${y} ${x + tl} ${y}Z`;
}

const circle = (cx, cy, r) => `<circle cx="${cx}" cy="${cy}" r="${r}"/>`;
const box = (x, y, w, h, r, fill, extra = '') => `<path d="${rr(x, y, w, h, r)}" fill="${fill}" ${extra}/>`;
const float = (x, y, w, h, r = 10) => `<path class="art__float" d="${rr(x, y, w, h, r)}" fill="${CARD}"/>`;
const spark = (cx, cy, r, fill) => `<path d="M${cx} ${cy - r}Q${cx} ${cy} ${cx + r} ${cy}Q${cx} ${cy} ${cx} ${cy + r}Q${cx} ${cy} ${cx - r} ${cy}Q${cx} ${cy} ${cx} ${cy - r}Z" fill="${fill}"/>`;

const text = (x, y, str, { size = 8, weight = 400, anchor = 'start', fill = INK, opacity = 1, font = FONT } = {}) =>
  `<text x="${x}" y="${y}" text-anchor="${anchor}" ${font} font-size="${size}" font-weight="${weight}" fill="${fill}" fill-opacity="${opacity}">${str}</text>`;

// A label pill on a dark scrim, for captions that sit on top of photos.
const scrimPill = (x, y, w, str, { anchor = 'middle' } = {}) =>
  `${box(x, y, w, 14, 7, C.scrim, 'fill-opacity=".55"')}${text(anchor === 'middle' ? x + w / 2 : x + 8, y + 10, str, { size: 7.5, weight: 600, fill: '#fff', anchor })}`;

const sendBtn = (cx, cy, fill) =>
  `<circle cx="${cx}" cy="${cy}" r="8.5" fill="${fill}"/><path d="M${cx} ${cy + 4}v-8m-3.4 3.4l3.4 -3.4l3.4 3.4" stroke="#fff" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`;

const avatar = (id, cx, cy, from, to) =>
  `<defs><linearGradient id="av-${id}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${from}"/><stop offset="1" stop-color="${to}"/></linearGradient></defs>
   <circle cx="${cx}" cy="${cy}" r="8" fill="url(#av-${id})"/>${spark(cx, cy, 4.2, '#fff')}`;

// Browser window: traffic lights, address pill, hairline; `body` is clipped to
// the content area below the 18px title bar.
function win(id, x, y, w, h, body) {
  const k = `wc-${id}`;
  return `
  <defs><clipPath id="${k}"><path d="${rr(x, y + 18, w, h - 18, [0, 0, 10, 10])}"/></clipPath></defs>
  <path class="art__card" d="${rr(x, y, w, h, 10)}" fill="${CARD}"/>
  <circle cx="${x + 11}" cy="${y + 9}" r="2.5" fill="#F28B82"/><circle cx="${x + 19}" cy="${y + 9}" r="2.5" fill="#FDD663"/><circle cx="${x + 27}" cy="${y + 9}" r="2.5" fill="#81C995"/>
  ${box(x + w / 2 - w * 0.17, y + 5, w * 0.34, 8, 4, FIELD)}
  <g clip-path="url(#${k})">${body}</g>
  <path d="M${x} ${y + 18.5}h${w}" stroke="${LINE}"/>`;
}

const person = (cx, top, { shirt = '#E8776A', skin = C.skin, hair = C.hair, scale = 1 } = {}) => {
  const s = scale;
  return `
    <path d="M${cx - 50 * s} ${top + 96 * s} q${4 * s} ${-44 * s} ${50 * s} ${-46 * s} q${46 * s} ${2 * s} ${50 * s} ${46 * s} z" fill="${shirt}"/>
    <path d="${rr(cx - 8 * s, top + 36 * s, 16 * s, 18 * s, 6 * s)}" fill="${skin}" fill-opacity=".9"/>
    <ellipse cx="${cx}" cy="${top + 26 * s}" rx="${17 * s}" ry="${20 * s}" fill="${skin}"/>
    <path d="M${cx - 18 * s} ${top + 26 * s} q0 ${-26 * s} ${19 * s} ${-26 * s} q${18 * s} 0 ${18 * s} ${25 * s} q${-6 * s} ${-13 * s} ${-18 * s} ${-13 * s} q${-12 * s} 0 ${-19 * s} ${14 * s}z" fill="${hair}"/>`;
};

const ART = {
  chat(id) {
    const reply = ['WebGPU gives web pages fast, parallel access', 'to your GPU. This model is answering you', 'from this tab; nothing is sent to a server.'];
    // Estimated line widths; fitTyping() swaps in measured ones once on screen.
    const widths = reply.map((s) => Math.round(s.length * 4.3));
    const ys = [118, 130.5, 143];
    return frame(id, 'chat', `${circle(346, 30, 86)}${circle(40, 206, 64)}`, `
      ${win(id, 62, 22, 278, 180, `
        ${box(204, 62, 124, 24, [11, 11, 3, 11], C.purple2, 'fill-opacity=".38"')}
        ${text(215, 77.5, 'What can WebGPU do?', { size: 8.5, weight: 500 })}
        ${avatar(id, 84, 108, '#C58AF9', '#8E44E8')}
        ${text(99, 105, 'Assistant', { size: 7.5, weight: 700, opacity: 0.55 })}
        <g class="chat-reply">
          ${reply.map((s, i) => text(99, ys[i], s, { size: 8.5, opacity: 0.88 }).replace('<text ', '<text data-typed ')).join('')}
          ${reply.map((s, i) => `<rect class="chat-ov chat-ov--${i + 1}" x="97" y="${ys[i] - 9.5}" width="244" height="13" fill="${CARD}" style="--w:${widths[i] + 4}px"/>`).join('')}
          <g class="chat-caret" style="--w1:${widths[0] + 2}px;--w2:${widths[1] + 2}px;--w3:${widths[2] + 2}px"><rect class="art-blink" x="99.5" y="${ys[0] - 8}" width="1.6" height="10" rx=".8" fill="${C.purple}"/></g>
          <g class="chat-after">${box(99, 150, 9, 9, 3, FIELD)}${box(112, 150, 9, 9, 3, FIELD)}</g>
        </g>
        ${box(76, 168, 250, 24, 12, FIELD)}
        ${text(91, 183, 'Message', { size: 8, opacity: 0.45 })}
        ${sendBtn(313, 180, C.purple)}`)}
      ${float(28, 52, 98, 22, 11)}
      <circle cx="41" cy="63" r="5.5" fill="${C.green}" fill-opacity=".18"/><circle cx="41" cy="63" r="2.8" fill="${C.green}"/>
      ${text(50, 66, 'Runs on your GPU', { size: 8, weight: 600 })}`);
  },

  effects(id) {
    const rng = mulberry32(5);
    const warm = ['#EBCFAE', '#E2BE98', '#F3DDC2', '#D9AE86', '#F6E6D1', '#CFA27A'];
    const big = [];
    for (let r = 0; r < 8; r++) for (let c = 0; c < 20; c++) big.push(`<rect x="${60 + c * 14}" y="${44 + r * 14}" width="14" height="14" fill="${warm[Math.floor(rng() * warm.length)]}"/>`);
    const room = (wall, win, pot, leaf) => `
      <rect x="60" y="44" width="280" height="108" fill="${wall}"/>
      <rect x="78" y="54" width="58" height="72" rx="3" fill="${win}"/>
      <path d="M107 54v72M78 90h58" stroke="${wall}" stroke-width="3"/>
      <path d="${rr(294, 116, 24, 24, 4)}" fill="${pot}"/>
      <ellipse cx="300" cy="104" rx="8" ry="16" fill="${leaf}" transform="rotate(-20 300 104)"/><ellipse cx="314" cy="102" rx="8" ry="17" fill="${leaf}" transform="rotate(18 314 102)"/>`;
    const scan = Array.from({ length: 27 }, (_, i) => `<rect x="60" y="${45 + i * 4}" width="280" height="1.6" fill="#1F2125" fill-opacity=".6"/>`).join('');
    const states = [
      ['Original', 50, room('#EAD3B8', '#F7EBDB', '#D9A57E', '#8CC49E') + person(200, 58)],
      ['Pixelate background', 98, `<g shape-rendering="crispEdges">${big.join('')}</g>${person(200, 58)}`],
      ['TV waves', 56, `<rect x="60" y="44" width="280" height="108" fill="#1F2125"/>${person(200, 58, { shirt: '#8AB4F8', skin: '#8AB4F8', hair: '#4285F4' })}${scan}`],
      ['Color pop', 58, room('#D3D5D9', '#E6E8EB', '#BDC1C6', '#C4C7CC') + person(200, 58)],
    ];
    // Each state shows for 2.5s of a 10s loop; Pixelate is the static frame.
    const phase = (k) => `style="animation-delay:${k === 0 ? 0 : -(10 - 2.5 * k)}s"`;
    const live = states.map(([label, w, body], k) => {
      const cls = k === 1 ? 'fx-state' : 'fx-state fx-state--rest';
      return `
      <g class="${cls}" ${phase(k)}>${body}${scrimPill(68, 52, w, label, { anchor: 'start' })}</g>`;
    }).join('');
    const mini = (x, y, shirt, skin = C.skin, hair = C.hair) => `
      <path d="M${x + 16} ${y + 32} q1 -12 16 -12.5 q15 .5 16 12.5z" fill="${shirt}"/>
      <circle cx="${x + 32}" cy="${y + 12.5}" r="6.5" fill="${skin}"/>
      <path d="M${x + 25.3} ${y + 12.5} q0 -8 6.8 -8 q6.6 0 6.6 8 q-2.5 -4 -6.6 -4 q-4.4 0 -6.8 4z" fill="${hair}"/>`;
    const thumbs = [
      ['Original', (x, y) => `<rect x="${x}" y="${y}" width="64" height="32" fill="#EAD3B8"/><rect x="${x + 6}" y="${y + 4}" width="14" height="20" fill="#F7EBDB"/>${mini(x, y, '#E8776A')}`],
      ['Pixelate', (x, y) => `${Array.from({ length: 16 }, (_, i) => `<rect x="${x + (i % 8) * 8}" y="${y + Math.floor(i / 8) * 16}" width="8" height="16" fill="${warm[(i * 5) % 6]}"/>`).join('')}${mini(x, y, '#E8776A')}`],
      ['TV waves', (x, y) => `<rect x="${x}" y="${y}" width="64" height="32" fill="#1F2125"/>${mini(x, y, '#8AB4F8', '#8AB4F8', '#4285F4')}${Array.from({ length: 8 }, (_, i) => `<rect x="${x}" y="${y + i * 4 + 1}" width="64" height="1.4" fill="#1F2125" fill-opacity=".6"/>`).join('')}`],
      ['Color pop', (x, y) => `<rect x="${x}" y="${y}" width="64" height="32" fill="#D3D5D9"/><rect x="${x + 6}" y="${y + 4}" width="14" height="20" fill="#E6E8EB"/>${mini(x, y, '#E8776A')}`],
    ];
    const strip = thumbs.map(([label, draw], i) => {
      const x = 60 + i * 72;
      const y = 158;
      const ring = i === 1 ? 'fx-ring' : 'fx-ring fx-ring--rest';
      return `
        <defs><clipPath id="th${i}-${id}"><rect x="${x}" y="${y}" width="64" height="32" rx="5"/></clipPath></defs>
        <g clip-path="url(#th${i}-${id})" shape-rendering="crispEdges">${draw(x, y)}</g>
        <path class="${ring}" ${phase(i)} d="${rr(x - 2, y - 2, 68, 36, 7)}" fill="none" stroke="${C.blue}" stroke-width="2"/>
        ${text(x + 32, y + 42, label, { size: 7, weight: 600, anchor: 'middle', opacity: 0.65 })}`;
    }).join('');
    return frame(id, 'effects', `${circle(40, 30, 76)}${circle(372, 200, 82)}`, `
      ${win(id, 52, 18, 296, 194, `
        <defs><clipPath id="v-${id}"><rect x="60" y="44" width="280" height="108" rx="7"/></clipPath></defs>
        <g clip-path="url(#v-${id})">${live}</g>
        ${box(68, 130, 40, 14, 7, C.scrim, 'fill-opacity=".55"')}
        <circle class="fx-live-ring" cx="77" cy="137" r="2.6" fill="${C.coral}"/><circle cx="77" cy="137" r="2.6" fill="${C.coral}"/>
        ${text(83, 140, 'Live', { size: 7.5, weight: 700, fill: '#fff' })}
        ${strip}`)}`);
  },

  segment(id) {
    const cat = 'M158 178 C 154 150, 162 130, 172 122 C 164 114, 162 100, 166 92 L 168 70 L 182 83 C 187 81, 193 81, 198 83 L 212 70 L 214 92 C 218 100, 216 114, 208 122 C 218 130, 226 150, 222 178 Z';
    const tail = 'M222 172 C 248 170, 250 146, 238 138';
    return frame(id, 'segment', `${circle(352, 30, 84)}${circle(40, 210, 66)}`, `
      ${win(id, 40, 22, 252, 182, `
        <defs><linearGradient id="wall-${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F4EADF"/><stop offset="1" stop-color="#E9DACA"/></linearGradient></defs>
        <rect x="40" y="40" width="252" height="164" fill="url(#wall-${id})"/>
        ${box(226, 52, 46, 34, 2, '#D6E4F5', 'stroke="#fff" stroke-width="3"')}
        <path d="M228 84 L244 68 L254 76 L262 70 L270 84 Z" fill="#A8D5B5"/>
        <rect x="40" y="186" width="252" height="18" fill="#E0CFBB"/>
        ${box(74, 106, 186, 46, 16, '#BCCDE6')}
        ${box(62, 140, 210, 40, 12, '#AFC3E2')}
        ${box(52, 122, 26, 58, 11, '#A3B8DA')}${box(256, 122, 26, 58, 11, '#A3B8DA')}
        <rect x="70" y="180" width="6" height="8" rx="2" fill="#8C7B6B"/><rect x="258" y="180" width="6" height="8" rx="2" fill="#8C7B6B"/>
        <path d="${tail}" stroke="#F4F5F7" stroke-width="7" fill="none" stroke-linecap="round"/>
        <path d="${cat}" fill="#F4F5F7"/>
        <path d="M184 84 l4 10 M196 84 l-4 10 M168 126 q8 4 6 14 M212 126 q-8 4 -6 14" stroke="#D5D8DD" stroke-width="2.2" fill="none" stroke-linecap="round"/>
        <circle cx="182" cy="102" r="2.6" fill="#3C4043"/><circle cx="198" cy="102" r="2.6" fill="#3C4043"/>
        <path d="M187.5 108 h5 l-2.5 3z" fill="#E8A0A0"/>
        <path class="seg-fill" d="${tail}" stroke="${C.blue}" stroke-opacity=".3" stroke-width="7" fill="none" stroke-linecap="round"/>
        <path class="seg-fill" d="${cat}" fill="${C.blue}" fill-opacity=".3"/>
        <path class="seg-outline" d="${cat}" pathLength="100" fill="none" stroke="#fff" stroke-width="4" stroke-linejoin="round"/>
        <path class="seg-outline" d="${cat}" pathLength="100" fill="none" stroke="${C.blue}" stroke-width="1.8" stroke-linejoin="round"/>
        <circle class="seg-ripple" cx="190" cy="140" r="8" fill="none" stroke="${C.green}" stroke-width="2"/>
        <circle class="seg-point" cx="190" cy="140" r="6" fill="${C.green}" stroke="#fff" stroke-width="2"/>
        <g class="seg-cursor"><path d="M195 145 L195 160 L199 156 L202 163 L205 161.6 L202 155 L207.5 155 Z" fill="#fff" stroke="#202124" stroke-width="1.2" stroke-linejoin="round"/></g>`)}
      ${float(262, 66, 104, 104)}
      ${text(274, 84, 'Objects', { size: 8.5, weight: 700 })}
      ${box(270, 92, 88, 24, 6, C.blue, 'fill-opacity=".1"')}
      ${box(276, 99, 10, 10, 3, C.blue)}
      ${text(292, 103, 'Object 1', { size: 8, weight: 600 })}
      ${text(292, 111.5, '1 point', { size: 6.5, opacity: 0.55 })}
      ${box(270, 121, 88, 18, 6, 'none', `stroke="${LINE}" stroke-width="1.2" stroke-dasharray="3 2.5"`)}
      ${text(314, 133, '+ Add object', { size: 7.5, weight: 500, anchor: 'middle', opacity: 0.6 })}
      ${box(270, 145, 88, 17, 8.5, C.blue)}
      ${text(314, 156.5, 'Cut out PNG', { size: 7.5, weight: 700, anchor: 'middle', fill: '#fff' })}`);
  },

  vlm(id) {
    return frame(id, 'vlm', `${circle(44, 34, 74)}${circle(370, 204, 84)}`, `
      ${win(id, 44, 22, 312, 182, `
        <defs>
          <clipPath id="p-${id}"><rect x="52" y="48" width="152" height="148" rx="6"/></clipPath>
          <linearGradient id="sky-${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#B9D2F7"/><stop offset="1" stop-color="#E8F0FC"/></linearGradient>
        </defs>
        <g clip-path="url(#p-${id})">
          <rect x="52" y="48" width="152" height="148" fill="url(#sky-${id})"/>
          <circle cx="176" cy="74" r="16" fill="#FFF4D0" fill-opacity=".6"/><circle cx="176" cy="74" r="9" fill="#FFE29A"/>
          <path d="M52 146 L86 106 L106 122 L134 86 L162 116 L182 102 L204 122 V196 H52Z" fill="#9DBFB0"/>
          <path d="M126 96 L134 86 L143 98 L138 95 L134 100 L130 95Z M80 113 L86 106 L92 114 L88 112 L86 115 L83 112Z" fill="#fff"/>
          <path d="M52 162 Q 100 140 150 158 T 204 152 V196 H52Z" fill="#6FA88A"/>
          <path d="M52 176 Q 128 168 204 176 V196 H52Z" fill="#A7C8EF"/>
          <path d="M80 182 h22 M120 186 h30 M160 181 h18" stroke="#fff" stroke-opacity=".7" stroke-width="1.4" stroke-linecap="round"/>
        </g>
        ${box(254, 62, 92, 22, [10, 10, 3, 10], C.green2, 'fill-opacity=".4"')}
        ${text(263, 76.5, 'How many peaks?', { size: 8, weight: 500 })}
        <g class="hv-fade">${avatar(id, 222, 104, '#81C995', '#1E8E3E')}</g>
        ${text(235, 101, 'Assistant', { size: 7, weight: 700, opacity: 0.55 })}
        <g class="hv-fade" style="animation-delay:.2s">${text(235, 114, 'Two peaks. The taller', { size: 8, opacity: 0.88 })}
        ${text(235, 125.5, 'one, on the right, has', { size: 8, opacity: 0.88 })}
        ${text(235, 137, 'snow on its summit.', { size: 8, opacity: 0.88 })}</g>
        ${box(212, 174, 136, 22, 11, FIELD)}
        ${text(224, 188, 'Ask about the photo', { size: 7.5, opacity: 0.45 })}
        ${sendBtn(337, 185, C.green)}`)}`);
  },

  depth(id) {
    const scene = (c) => `
      <rect x="52" y="36" width="296" height="140" fill="${c.sky}"/>
      <path d="M52 112 L96 80 L130 100 L170 70 L214 98 L256 76 L300 96 L348 84 V176 H52Z" fill="${c.far}"/>
      <path d="M52 136 Q 120 112 190 130 T 348 122 V176 H52Z" fill="${c.mid}"/>
      <path d="M52 158 Q 150 146 240 156 T 348 152 V176 H52Z" fill="${c.near}"/>
      <rect x="115" y="126" width="6" height="18" rx="2" fill="${c.trunk}"/><circle cx="118" cy="118" r="13" fill="${c.tree}"/>
      <circle cx="270" cy="118" r="7" fill="${c.skin}"/><path d="${rr(261, 126, 18, 30, 8)}" fill="${c.body}"/>
      <rect x="263" y="150" width="6" height="18" rx="3" fill="${c.body}"/><rect x="271" y="150" width="6" height="18" rx="3" fill="${c.body}"/>`;
    const ramp = ['#F4A24A', '#A7D45E', '#3DB6A6', '#4F74D6', '#463B98'];
    return frame(id, 'depth', `${circle(358, 24, 84)}${circle(32, 206, 64)}`, `
      ${win(id, 52, 18, 296, 192, `
        <defs>
          <clipPath id="r-${id}"><rect class="depth-sweep" x="200" y="36" width="220" height="140"/></clipPath>
          <linearGradient id="sky-${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#C2D8F8"/><stop offset="1" stop-color="#EEF4FD"/></linearGradient>
          <linearGradient id="cm-${id}" x1="0" y1="0" x2="1" y2="0">${ramp.map((c, i) => `<stop offset="${i / 4}" stop-color="${c}"/>`).join('')}</linearGradient>
        </defs>
        <g>${scene({ sky: `url(#sky-${id})`, far: '#AFC6DC', mid: '#8CC49E', near: '#5FA77A', tree: '#3E8E5E', trunk: '#7A5A44', skin: C.skin, body: '#E8776A' })}</g>
        <g clip-path="url(#r-${id})">${scene({ sky: ramp[4], far: ramp[3], mid: ramp[2], near: ramp[1], tree: '#6CC58A', trunk: '#6CC58A', skin: ramp[0], body: ramp[0] })}</g>
        <g class="depth-sweep">
          <rect x="199" y="36" width="2" height="140" fill="#fff"/>
          <circle class="art__float" cx="200" cy="106" r="11" fill="${CARD}"/>
          <path d="M196 102 L192 106 L196 110 M204 102 L208 106 L204 110" stroke="${INK}" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
        </g>
        ${scrimPill(60, 156, 38, 'Photo')}${scrimPill(302, 156, 38, 'Depth')}
        ${text(64, 195.5, 'Near', { size: 7, weight: 600, opacity: 0.6 })}
        ${box(90, 189.5, 220, 7, 3.5, `url(#cm-${id})`)}
        ${text(318, 195.5, 'Far', { size: 7, weight: 600, opacity: 0.6 })}`)}`);
  },

  detect(id) {
    const label = (x, y, w, h, color, str, ink = '#fff') => `
      <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="${color}" fill-opacity=".1" stroke="${color}" stroke-width="1.8"/>
      ${box(x - 0.9, y - 13, str.length * 4.3 + 10, 13, [3, 3, 3, 0], color)}
      ${text(x + 4, y - 4, str, { size: 7.5, weight: 700, fill: ink })}`;
    const windows = (x, y, cols, rows) => Array.from({ length: cols * rows }, (_, i) => `<rect x="${x + (i % cols) * 10}" y="${y + Math.floor(i / cols) * 12}" width="6" height="7" rx="1" fill="#fff" fill-opacity=".65"/>`).join('');
    return frame(id, 'detect', `${circle(346, 198, 90)}${circle(44, 28, 64)}`, `
      ${win(id, 46, 18, 308, 192, `
        <defs><linearGradient id="sky-${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#CFE1F8"/><stop offset="1" stop-color="#EEF4FD"/></linearGradient></defs>
        <rect x="46" y="36" width="308" height="174" fill="url(#sky-${id})"/>
        <rect x="56" y="70" width="48" height="78" fill="#D9D3E8"/>${windows(62, 78, 4, 5)}
        <rect x="108" y="56" width="40" height="92" fill="#E6DCCF"/>${windows(113, 64, 3, 7)}
        <rect x="292" y="62" width="62" height="86" fill="#D4DEEA"/>${windows(298, 70, 5, 6)}
        <rect x="46" y="146" width="308" height="10" fill="#E3E1DC"/>
        <rect x="46" y="156" width="308" height="54" fill="#C9CED6"/>
        <path d="M56 186 h24 M100 186 h24 M144 186 h24 M330 186 h24" stroke="#fff" stroke-opacity=".75" stroke-width="2.5" stroke-linecap="round"/>
        <rect x="174" y="118" width="5" height="30" rx="2" fill="#7A5A44"/><circle cx="176" cy="110" r="14" fill="#8CC49E"/>
        <circle cx="137" cy="98" r="8" fill="${C.skin}"/><path d="M129 97 q0 -10 8 -10 q8 0 8 9 q-3 -5 -8 -5 q-5 0 -8 6z" fill="${C.hair}"/>
        <path d="${rr(127, 106, 20, 32, 8)}" fill="#5B8DEF"/>
        <rect x="128" y="134" width="7" height="22" rx="3" fill="#3C4A63"/><rect x="139" y="134" width="7" height="22" rx="3" fill="#3C4A63"/>
        <path d="${rr(72, 138, 30, 12, 6)}" fill="#E9B949"/><circle cx="104" cy="136" r="7" fill="#E9B949"/><path d="M106 130 l5 -3 l-1 7z" fill="#D49E2E"/>
        <rect x="75" y="146" width="4" height="10" rx="2" fill="#E9B949"/><rect x="94" y="146" width="4" height="10" rx="2" fill="#E9B949"/>
        <path d="M73 141 q-6 -2 -7 -9" stroke="#E9B949" stroke-width="3.5" fill="none" stroke-linecap="round"/>
        <path d="M194 184 h124 v-16 a10 10 0 0 0 -10 -10 h-14 l-14 -18 h-48 l-14 18 h-14 a10 10 0 0 0 -10 10 z" fill="#EE7B6E"/>
        <path d="M232 144 h20 v14 h-31z M258 144 h18 l11 14 h-29z" fill="#DDE8F6"/>
        <circle cx="222" cy="184" r="10" fill="#3C4043"/><circle cx="222" cy="184" r="4" fill="#9AA0A6"/>
        <circle cx="292" cy="184" r="10" fill="#3C4043"/><circle cx="292" cy="184" r="4" fill="#9AA0A6"/>
        <g class="hv-pop">${label(122, 86, 30, 72, C.blue, 'person 0.97')}</g>
        <g class="hv-pop" style="animation-delay:.15s">${label(190, 136, 132, 60, C.coral, 'car 0.94')}</g>
        <g class="hv-pop" style="animation-delay:.3s">${label(64, 125, 50, 33, C.yellow, 'dog 0.88', '#3C4043')}</g>`)}
      ${float(26, 46, 80, 22, 11)}
      <path d="M36 54v-2.5a1.5 1.5 0 0 1 1.5 -1.5h2.5 M47 50h2.5a1.5 1.5 0 0 1 1.5 1.5v2.5 M51 60v2.5a1.5 1.5 0 0 1 -1.5 1.5h-2.5 M40 64h-2.5a1.5 1.5 0 0 1 -1.5 -1.5v-2.5" stroke="${C.blue}" stroke-width="1.6" fill="none" stroke-linecap="round"/>
      ${text(57, 60, '3 objects', { size: 8, weight: 600 })}`);
  },

  bgremove(id) {
    const checks = [];
    for (let y = 36; y < 210; y += 10) for (let x = 92; x < 308; x += 10) checks.push(`<rect x="${x}" y="${y}" width="10" height="10" fill="${((x - 92) / 10 + (y - 36) / 10) % 2 ? '#ECEEF1' : '#FFFFFF'}"/>`);
    const sw = [[146, `url(#ck-${id})`], [170, `url(#bl-${id})`], [194, '#F6B8A8'], [218, '#C9B3F2']];
    return frame(id, 'bgremove', `${circle(348, 196, 90)}${circle(44, 28, 66)}`, `
      <defs>
        <pattern id="ck-${id}" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="4" fill="#fff"/><rect width="2" height="2" fill="#D5D8DD"/><rect x="2" y="2" width="2" height="2" fill="#D5D8DD"/></pattern>
        <linearGradient id="bl-${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#CFE3F7"/><stop offset="1" stop-color="#A9D7A0"/></linearGradient>
      </defs>
      ${win(id, 92, 18, 216, 192, `
        <defs><linearGradient id="sky-${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#CFE3F7"/><stop offset="1" stop-color="#EAF3FB"/></linearGradient></defs>
        <rect x="92" y="36" width="108" height="174" fill="url(#sky-${id})"/>
        <circle cx="112" cy="104" r="22" fill="#8CC49E"/><circle cx="140" cy="96" r="18" fill="#6FB287"/><circle cx="178" cy="108" r="20" fill="#8CC49E"/>
        <rect x="92" y="120" width="108" height="90" fill="#A9D7A0"/>
        <defs><clipPath id="ck2-${id}"><rect class="hv-slide" style="--to:-96px" x="200" y="36" width="216" height="174"/></clipPath></defs>
        <g clip-path="url(#ck2-${id})" shape-rendering="crispEdges">${checks.join('')}</g>
        <g class="hv-slide" style="--to:-96px"><rect x="199" y="36" width="2" height="174" fill="#fff"/></g>
        <path d="M140 210 q6 -54 60 -56 q54 2 60 56z" fill="#6E9CF0"/>
        <path d="M188 154 q12 10 24 0" stroke="#5A86D8" stroke-width="2" fill="none"/>
        <path d="${rr(191, 128, 18, 26, 8)}" fill="${C.skin}" fill-opacity=".92"/>
        <ellipse cx="200" cy="108" rx="24" ry="28" fill="${C.skin}"/>
        <path d="M175 108 q-3 -40 26 -40 q27 0 24 38 q-4 -18 -24 -20 q-18 2 -26 22z" fill="${C.hair}"/>
        <g class="hv-slide" style="--to:-96px">
          <circle class="art__float" cx="200" cy="54" r="10" fill="${CARD}"/>
          <path d="M196.5 50.5 L193 54 L196.5 57.5 M203.5 50.5 L207 54 L203.5 57.5" stroke="${INK}" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
        </g>`)}
      ${float(126, 176, 150, 28, 14)}
      ${sw.map(([x, fill]) => `<circle cx="${x}" cy="190" r="8" fill="${fill}" stroke="${LINE}" stroke-width="1"/>`).join('')}
      <circle cx="146" cy="190" r="10.5" fill="none" stroke="${C.coral}" stroke-width="1.6"/>
      <path d="M232 182 v16" stroke="${LINE}"/>
      ${text(238, 193, 'Background', { size: 7.5, weight: 600, opacity: 0.7 })}`);
  },

  superres(id) {
    const [fx, fy] = [200, 116];
    const petalsAt = 8;
    const inPetal = (x, y) => {
      for (let i = 0; i < petalsAt; i++) {
        const a = (-i * 2 * Math.PI) / petalsAt;
        const dx = x - fx;
        const dy = y - fy;
        const lx = dx * Math.cos(a) - dy * Math.sin(a);
        const ly = dx * Math.sin(a) + dy * Math.cos(a);
        if ((lx / 15) ** 2 + ((ly + 34) / 34) ** 2 < 1) return true;
      }
      return false;
    };
    const color = (x, y) => {
      const d = Math.hypot(x - fx, y - fy);
      if (d < 10) return '#F2B33D';
      if (d < 18) return '#F7C952';
      if (inPetal(x, y)) return d < 42 ? '#EF8E8E' : '#F7B4B1';
      if (Math.abs(x - fx) < 3 && y > fy) return '#5FA77A';
      if (Math.hypot(x - 96, y - 72) < 22 || Math.hypot(x - 312, y - 176) < 26) return '#E4F2E1';
      return y < 120 ? '#D3EAD0' : '#BFDFBB';
    };
    const px = [];
    for (let y = 36; y < 210; y += 12) for (let x = 52; x < 200; x += 12) px.push(`<rect x="${x}" y="${y}" width="12" height="12" fill="${color(x + 6, y + 6)}"/>`);
    const petals = Array.from({ length: petalsAt }, (_, i) => `<ellipse cx="${fx}" cy="${fy - 34}" rx="15" ry="34" fill="url(#pt-${id})" transform="rotate(${(i * 360) / petalsAt} ${fx} ${fy})"/>`).join('');
    return frame(id, 'superres', `${circle(370, 34, 76)}${circle(28, 200, 62)}`, `
      ${win(id, 52, 18, 296, 192, `
        <defs>
          <clipPath id="l-${id}"><rect class="hv-slide" style="--to:-140px" x="52" y="36" width="148" height="174"/></clipPath>
          <linearGradient id="bg2-${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#D9EDD6"/><stop offset="1" stop-color="#BBDDB6"/></linearGradient>
          <radialGradient id="pt-${id}" gradientUnits="userSpaceOnUse" cx="${fx}" cy="${fy}" r="70"><stop offset=".25" stop-color="#EA8080"/><stop offset="1" stop-color="#FBCDC9"/></radialGradient>
          <radialGradient id="ct-${id}" cx=".4" cy=".35" r=".7"><stop offset="0" stop-color="#FCDA6B"/><stop offset="1" stop-color="#EDA634"/></radialGradient>
        </defs>
        <g>
          <rect x="52" y="36" width="296" height="174" fill="url(#bg2-${id})"/>
          <circle cx="96" cy="72" r="22" fill="#E4F2E1" fill-opacity=".8"/>
          <circle cx="312" cy="176" r="26" fill="#E4F2E1" fill-opacity=".8"/>
          <path d="M200 132 V210" stroke="#5FA77A" stroke-width="5"/>
          <path d="M200 176 q22 -18 40 -6 q-20 14 -40 6z" fill="#6FB287"/>
          ${petals}
          <circle cx="${fx}" cy="${fy}" r="18" fill="url(#ct-${id})"/>
          ${Array.from({ length: 10 }, (_, i) => `<circle cx="${(fx + 9 * Math.cos(i * 0.63)).toFixed(1)}" cy="${(fy + 9 * Math.sin(i * 0.63)).toFixed(1)}" r="1.3" fill="#C98A1E" fill-opacity=".55"/>`).join('')}
        </g>
        <g clip-path="url(#l-${id})" shape-rendering="crispEdges">${px.join('')}</g>
        <g class="hv-slide" style="--to:-140px">
          <rect x="199" y="36" width="2" height="174" fill="#fff"/>
          <circle class="art__float" cx="200" cy="164" r="11" fill="${CARD}"/>
          <path d="M196 160 L192 164 L196 168 M204 160 L208 164 L204 168" stroke="${INK}" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
        </g>
        ${scrimPill(60, 188, 40, 'Input')}${scrimPill(292, 188, 48, '4× output')}`)}
      ${float(26, 48, 82, 22, 11)}
      ${spark(40, 59, 5, C.coral)}
      ${text(50, 62, '4× upscale', { size: 8, weight: 600 })}`);
  },

  stt(id) {
    const bars = Array.from({ length: 42 }, (_, i) => {
      const env = 0.35 + 0.65 * Math.abs(Math.sin(i * 0.37) * Math.cos(i * 0.11 + 0.4));
      const h = 3 + env * 18;
      return `<rect class="stt-bar" style="animation-delay:${(-((i * 0.37) % 0.9)).toFixed(2)}s" x="${(102 + i * 4.5).toFixed(1)}" y="${(64 - h / 2).toFixed(1)}" width="2.4" height="${h.toFixed(1)}" rx="1.2" fill="${i < 28 ? C.blue : C.blue2}" fill-opacity="${i < 28 ? 1 : 0.55}"/>`;
    }).join('');
    const ts = (y, t) => text(72, y, t, { size: 7, weight: 500, opacity: 0.45, font: MONO });
    return frame(id, 'stt', `${circle(40, 198, 72)}${circle(354, 30, 84)}`, `
      ${win(id, 56, 22, 288, 182, `
        ${box(66, 48, 268, 32, 16, FIELD)}
        <circle cx="82" cy="64" r="12" fill="${C.blue}"/>
        <rect x="78.5" y="56" width="7" height="12" rx="3.5" fill="#fff"/>
        <path d="M75.5 64.5a6.5 6.5 0 0 0 13 0M82 71v3" stroke="#fff" stroke-width="1.5" fill="none" stroke-linecap="round"/>
        ${bars}
        <rect x="228.6" y="52" width="1.4" height="24" rx=".7" fill="${INK}" fill-opacity=".6"/>
        ${text(300, 67, '0:07', { size: 8, weight: 600, opacity: 0.7, font: MONO })}
        <g class="stt-row stt-row--1">${ts(102, '0:00')}${text(98, 102, 'Okay, so here is the idea.', { size: 8.5, opacity: 0.88 })}</g>
        <g class="stt-row stt-row--2">${text(98, 115, 'The model runs right here in the browser,', { size: 8.5, opacity: 0.88 })}</g>
        <g class="stt-row stt-row--3">${ts(136, '0:04')}${text(98, 136, 'and nothing I say leaves this laptop.', { size: 8.5, opacity: 0.88 })}</g>
        <g class="stt-row stt-row--4">${ts(157, '0:07')}${text(98, 157, 'It even works offline', { size: 8.5, opacity: 0.45 })}
        <rect class="art-blink" x="190" y="149" width="1.8" height="11" rx=".9" fill="${C.blue}"/></g>
        ${box(70, 176, 54, 16, 8, FIELD)}${text(97, 187, 'English', { size: 7, weight: 600, anchor: 'middle', opacity: 0.7 })}
        ${box(130, 176, 58, 16, 8, FIELD)}${text(159, 187, 'Copy text', { size: 7, weight: 600, anchor: 'middle', opacity: 0.7 })}`)}`);
  },

  cppn(id) {
    const palette = ['#F7B6A6', '#FBD5C4', '#FDE5B0', '#F9CF8A', '#F4A9C0', '#DDBDF4', '#BFA3EE', '#F9C7D6'];
    const bands = Array.from({ length: 14 }, (_, i) => {
      const y0 = 26 + i * 13.5;
      let d = `M 28 ${y0}`;
      for (let x = 28; x <= 296; x += 4) {
        const y = y0 + 9 * Math.sin(x / 26 + i * 0.7) + 5 * Math.sin(x / 11 - i * 1.3) + 3 * Math.cos(x / 47 + i);
        d += ` L ${x} ${y.toFixed(1)}`;
      }
      return `<path class="hv-drift" style="animation-delay:${(-i * 0.37).toFixed(2)}s" d="${d} L 296 214 L 28 214 Z" fill="${palette[i % palette.length]}"/>`;
    }).join('');
    const slider = (y, label, fill) => `
      ${text(266, y, label, { size: 7, weight: 600, opacity: 0.6 })}
      ${box(266, y + 5, 88, 4, 2, LINE)}${box(266, y + 5, fill, 4, 2, C.coral)}
      <circle class="art__card" cx="${266 + fill}" cy="${y + 7}" r="5" fill="#fff"/>`;
    return frame(id, 'cppn', `${circle(46, 200, 80)}${circle(360, 26, 72)}`, `
      ${win(id, 44, 18, 236, 192, `
        <defs><radialGradient id="sh-${id}" cx=".2" cy=".1" r=".9"><stop offset="0" stop-color="#fff" stop-opacity=".45"/><stop offset=".6" stop-color="#fff" stop-opacity="0"/></radialGradient></defs>
        <rect x="44" y="36" width="236" height="174" fill="#FDE5B0"/>
        ${bands}
        <rect x="44" y="36" width="236" height="174" fill="url(#sh-${id})"/>`)}
      ${float(254, 62, 112, 114)}
      ${text(266, 80, 'Network', { size: 8.5, weight: 700 })}
      ${slider(95, 'Zoom', 52)}
      ${slider(117, 'Speed', 34)}
      ${text(266, 146, 'Animate', { size: 7.5, weight: 600, opacity: 0.75 })}
      ${box(330, 138, 24, 13, 6.5, C.coral)}<circle cx="347.5" cy="144.5" r="4.8" fill="#fff"/>
      ${box(266, 156, 88, 14, 7, FIELD)}
      ${text(310, 166, 'New network', { size: 7.5, weight: 600, anchor: 'middle' })}`);
  },

  trainer(id) {
    const colors = [C.blue, C.coral, C.green];
    const tints = [C.blue2, C.coral2, C.green2];
    const [ox, oy] = [158, 122];
    const pts = [];
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < 22; i++) {
        const t = i / 22;
        const r = 9 + t * 66;
        const a = t * 3.6 + (c * 2 * Math.PI) / 3;
        const j = c * 22 + i;
        pts.push(`<circle class="hv-settle" style="--dx:${(Math.sin(j * 12.9898) * 22).toFixed(1)}px;--dy:${(Math.cos(j * 78.233) * 22).toFixed(1)}px;animation-delay:${j * 6}ms" cx="${(ox + r * Math.cos(a)).toFixed(1)}" cy="${(oy + r * Math.sin(a) * 0.95).toFixed(1)}" r="3" fill="${colors[c]}" stroke="${CARD}" stroke-width="1.1"/>`);
      }
    }
    const grid = [];
    for (let x = 64; x < 276; x += 28) grid.push(`<path d="M${x} 36V210" stroke="${LINE}" stroke-width=".8"/>`);
    for (let y = 52; y < 210; y += 28) grid.push(`<path d="M40 ${y}H276" stroke="${LINE}" stroke-width=".8"/>`);
    return frame(id, 'trainer', `${circle(360, 24, 84)}${circle(30, 206, 70)}`, `
      ${win(id, 40, 18, 236, 192, `
        ${grid.join('')}
        <g fill-opacity=".16">
          <path d="M${ox} ${oy} L40 20 L276 10 Z" fill="${tints[0]}"/>
          <path d="M${ox} ${oy} L276 10 L290 214 Z" fill="${tints[1]}"/>
          <path d="M${ox} ${oy} L290 214 L30 214 L40 20 Z" fill="${tints[2]}"/>
        </g>
        ${pts.join('')}`)}
      ${float(244, 72, 122, 90)}
      ${text(256, 89, 'Loss', { size: 7.5, weight: 600, opacity: 0.6 })}
      ${text(256, 106, '0.083', { size: 13, weight: 700 })}
      ${text(354, 105, 'epoch 120', { size: 7, weight: 500, opacity: 0.5, anchor: 'end' })}
      <path d="M256 116 C 268 140, 282 146, 300 148 S 340 151, 354 151 V152 H256Z" fill="${C.yellow}" fill-opacity=".14"/>
      <path d="M256 116 C 268 140, 282 146, 300 148 S 340 151, 354 151" stroke="${C.yellow}" stroke-width="2" fill="none" stroke-linecap="round"/>
      <path d="M256 152.5h98" stroke="${LINE}"/>
      ${float(244, 170, 88, 24, 12)}
      <circle cx="258" cy="182" r="6.5" fill="${C.green}"/><path d="M256 179.5v5M260 179.5v5" stroke="#fff" stroke-width="1.6" stroke-linecap="round"/>
      ${text(270, 185, 'Training…', { size: 8, weight: 600 })}`);
  },

  kernels(id) {
    const K = '#A142F4';
    const A = C.blue;
    const N = '#E8710A';
    const code = [
      [['@compute', A], [' '], ['@workgroup_size', A], ['('], ['16', N], [', '], ['16', N], [')']],
      [['fn', K], [' main(']],
      [['  '], ['@builtin', A], ['(global_invocation_id) id: '], ['vec3u', C.green]],
      [[') {']],
      [['  '], ['var', K], [' acc = '], ['0.0', N], [';']],
      [['  '], ['for', K], [' ('], ['var', K], [' k = '], ['0u', N], ['; k &lt; K; k++) {']],
      [['    acc += A[id.y * K + k] * B[k * N + id.x];']],
      [['  }']],
      [['  C[id.y * N + id.x] = acc;']],
      [['}']],
    ];
    const lines = code.map((parts, i) => {
      const y = 52 + i * 15;
      const spans = parts.map(([t, c]) => `<tspan fill="${c ?? INK}">${t.replace(/ /g, ' ')}</tspan>`).join('');
      return `${text(57, y, String(i + 1), { size: 6.5, opacity: 0.35, anchor: 'end', font: MONO })}<text x="68" y="${y}" ${MONO} font-size="7.2" fill="${INK}">${spans}</text>`;
    }).join('');
    const grid = (x, y, hi) => Array.from({ length: 16 }, (_, i) => {
      const r = Math.floor(i / 4);
      const c = i % 4;
      const on = hi(r, c);
      return `<rect x="${x + c * 7}" y="${y + r * 7}" width="5.6" height="5.6" rx="1.2" fill="${on ?? LINE}"/>`;
    }).join('');
    return frame(id, 'kernels', `${circle(352, 200, 84)}${circle(40, 26, 64)}`, `
      ${win(id, 40, 18, 236, 192, `
        <rect x="40" y="36" width="22" height="174" fill="${FIELD}"/>
        <rect x="62" y="138" width="214" height="13" fill="${C.purple2}" fill-opacity=".14"/>
        ${lines}`)}
      ${float(248, 60, 118, 116)}
      ${text(260, 78, 'C = A × B', { size: 8.5, weight: 700 })}
      ${grid(260, 88, () => null)}${grid(295, 88, () => null)}${grid(330, 88, () => null)}
      <g class="kern-y">${[0, 1, 2, 3].map((c) => `<rect x="${260 + c * 7}" y="88" width="5.6" height="5.6" rx="1.2" fill="${C.yellow2}"/>`).join('')}</g>
      <g class="kern-x">${[0, 1, 2, 3].map((r) => `<rect x="295" y="${88 + r * 7}" width="5.6" height="5.6" rx="1.2" fill="${C.blue2}"/>`).join('')}</g>
      <g class="kern-y"><g class="kern-x"><rect x="330" y="88" width="5.6" height="5.6" rx="1.2" fill="${C.coral2}"/></g></g>
      ${box(260, 128, 94, 18, 9, C.green, 'fill-opacity=".14"')}
      <path d="M268 137 l3 3 l6 -6.5" stroke="${C.green}" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
      ${text(282, 140, 'Matches CPU', { size: 7.5, weight: 700 })}
      ${text(260, 163, 'fp32 · 256 × 256', { size: 7, opacity: 0.55, font: MONO })}`);
  },
};

// The typing reveal and caret stop at each line's end, which depends on the
// font that actually loaded.
function fitTyping(el) {
  const caret = el.querySelector('.chat-caret');
  el.querySelectorAll('[data-typed]').forEach((t, i) => {
    const w = Math.ceil(t.getComputedTextLength());
    if (!w) return;
    el.querySelector(`.chat-ov--${i + 1}`)?.style.setProperty('--w', `${w + 4}px`);
    caret?.style.setProperty(`--w${i + 1}`, `${w + 2}px`);
  });
}

// Pause motion for arts scrolled out of view; a removed art reports
// non-intersecting once, which is when it is released.
const visibility = typeof IntersectionObserver === 'function'
  ? new IntersectionObserver((entries) => {
    for (const e of entries) {
      e.target.classList.toggle('art--paused', !e.isIntersecting);
      if (e.isIntersecting) fitTyping(e.target);
      if (!e.target.isConnected) visibility.unobserve(e.target);
    }
  })
  : null;

export function illustration(name) {
  const make = ART[name];
  if (!make) throw new Error(`No illustration named ${name}`);
  const el = svg(make(uid('a')));
  visibility?.observe(el);
  return el;
}
