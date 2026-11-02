// Text preparation for the Phonon text-to-speech demo (no DOM), ported from the
// official runtime (the `ptts` crate: src/preprocess.rs, preprocess/rewrite/
// {numbers,currency}.rs, tts_model.rs, plan.rs) and checked against its tests.
//
// English only, and a subset: the character pass (quotes, dashes, brackets,
// emoji, a few math symbols), numbers with scale words ("1,234" -> "1 thousand
// 234"), amounts of money, spoken "@", "+" and "=". Emails, URLs, abbreviations
// and the other rewrite rules are not ported, so such text is read as written.

const PUNCTUATION = '!?.:,;…';
const DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
const SCALES = [
  [1e9, 'billion'],
  [1e6, 'million'],
  [1e3, 'thousand'],
];
const SCALE_WORDS = ['thousand', 'million', 'billion', 'trillion'];
const CURRENCY = { $: ['dollar', 'dollars'], '€': ['euro', 'euros'], '£': ['pound', 'pounds'], '¥': ['yen', 'yen'], '¢': ['cent', 'cents'] };
const SCALE_SUFFIXES = [
  ['thousand', 'thousand'], ['million', 'million'], ['billion', 'billion'], ['trillion', 'trillion'],
  ['bn', 'billion'], ['BN', 'billion'], ['mn', 'million'], ['tn', 'trillion'],
  ['k', 'thousand'], ['K', 'thousand'], ['m', 'million'], ['M', 'million'], ['b', 'billion'], ['B', 'billion'],
];
const SYMBOLS = {
  '×': 'times', '÷': 'divided by', '≤': 'less than or equal to', '≥': 'greater than or equal to',
  '≠': 'is not equal to', '≈': 'approximately', '√': 'square root of', '∞': 'infinity', '∑': 'sigma',
  '‰': 'per mille', '±': 'plus or minus', 'π': 'pi',
};
const DOUBLE_QUOTES = new Set('»«‹›“”„‟″‶⹂❝❞❠〝〞〟＂');
const SINGLE_QUOTES = new Set('‘’‚‛′‵❛❜❟＇ʼʻʹ´`');
const SLASH_UNITS = new Set(['km', 'kg', 'mg', 'ml', 'mL', 'dL', 'mol', 'kJ', 'kW', 'kWh', 'Hz', 'min', 'hr', 'hrs', 'sec', 'ft', 'lb', 'lbs', 'oz', 'mi', 'mph', 'kph', 'yd', 'cm', 'mm', 'yr', 'wk', 'mo', 'kcal', 'rpm']);

const isDigit = (c) => c >= '0' && c <= '9';
const allDigits = (s) => s.length > 0 && [...s].every(isDigit);
const isSpace = (c) => /\s/u.test(c);
const isAsciiPunct = (c) => /^[!-/:-@[-`{-~]$/.test(c);
const isAlpha = (c) => /\p{L}/u.test(c);
const isAlnum = (c) => /[\p{L}\p{N}]/u.test(c);
const isEmoji = (c) => {
  const cp = c.codePointAt(0);
  return (cp >= 0x1f000 && cp <= 0x1faff) || (cp >= 0x2600 && cp <= 0x27bf) || (cp >= 0x2b00 && cp <= 0x2bff) || cp === 0xfe0f || cp === 0x200d;
};

function splitSuffix(word) {
  let i = word.length;
  while (i > 0 && PUNCTUATION.includes(word[i - 1])) i--;
  return [word.slice(0, i), word.slice(i)];
}

const isSuffix = (rest) => [...rest].every((c) => PUNCTUATION.includes(c));
const spell = (s) => [...s].filter((c) => c !== '-').map((c) => (isDigit(c) ? DIGITS[+c] : c)).join(' ');

// "1,234.56" -> ["1 thousand 234 point 56", trailing punctuation], or null.
export function numberWords(word) {
  if (word[0] === '0' && isDigit(word[1] ?? '')) return null;
  const negative = word.startsWith('-');
  let rest = negative ? word.slice(1) : word;
  let end = 0;
  while (end < rest.length && (isDigit(rest[end]) || rest[end] === ',')) end++;
  const int = rest.slice(0, end).replace(/,+$/, '');
  if (!int) return null;
  rest = rest.slice(int.length);
  let decimal = null;
  if (rest.startsWith('.')) {
    let d = 1;
    while (d < rest.length && isDigit(rest[d])) d++;
    if (d > 1) {
      decimal = rest.slice(1, d);
      rest = rest.slice(d);
    }
  }
  if (!isSuffix(rest)) return null;
  const value = Number(int.replaceAll(',', ''));
  if (!Number.isSafeInteger(value) || value >= 1e12) return null;
  let words;
  if (value < 1000 || (value >= 1900 && value < 2100)) words = String(value);
  else {
    const parts = SCALES.map(([div, name]) => [Math.floor(value / div) % 1000, name]).filter(([n]) => n).map(([n, name]) => `${n} ${name}`);
    if (value % 1000) parts.push(String(value % 1000));
    words = parts.join(' ');
  }
  if (decimal !== null) words = `${words} point ${decimal}`;
  if (negative) words = `minus ${words}`;
  return [words, rest];
}

function range(body) {
  const i = body.indexOf('-');
  if (i < 0) return null;
  const [from, to] = [body.slice(0, i), body.slice(i + 1)];
  const small = (s) => s.length >= 2 && s.length <= 3 && allDigits(s) && s[0] !== '0';
  const year = (s) => s.length === 4 && allDigits(s) && (s.startsWith('1') || s.startsWith('20'));
  let ok = false;
  if (small(from) && small(to)) ok = Number(from) < Number(to);
  else if (from.length === 4 && to.length === 4) ok = year(from) && year(to) && from < to;
  else if (from.length === 4 && to.length === 2) ok = year(from) && allDigits(to) && from.slice(2) < to;
  return ok ? `${from} to ${to}` : null;
}

// Large numbers with scale words; codes (leading zero) spelled; "20-25" a range.
export function numbers(word) {
  const [body, end] = splitSuffix(word);
  if (body.length >= 2 && body[0] === '0' && allDigits(body)) return `${spell(body)}${end}`;
  const r = range(body);
  if (r) return `${r}${end}`;
  const n = numberWords(word);
  if (n) return n[0] + n[1];
  const dash = word.indexOf('-');
  if (dash < 0 || !isAlpha(word[dash + 1] ?? '')) return null;
  const head = numberWords(word.slice(0, dash));
  return head && head[1] === '' ? head[0] + word.slice(dash) : null;
}

function splitSymbol(word) {
  if (CURRENCY[word[0]]) return [word[0], word.slice(1)];
  const last = word.at(-1);
  return last && CURRENCY[last] ? [last, word.slice(0, -1)] : null;
}

function withScale(amount, scale, symbol) {
  const n = numberWords(amount);
  return n && n[1] === '' ? `${n[0]} ${scale} ${CURRENCY[symbol][1]}` : null;
}

function withCents(amount, symbol) {
  const dot = amount.lastIndexOf('.');
  if (dot < 0) return null;
  const [whole, cents] = [amount.slice(0, dot), amount.slice(dot + 1)];
  const grouped = whole.split(',').every((g, i) => allDigits(g) && (i === 0 ? g.length <= 3 : g.length === 3));
  if (!(allDigits(whole) || grouped) || cents.length !== 2 || !allDigits(cents)) return null;
  const n = numberWords(whole);
  if (!n) return null;
  const [one, many] = CURRENCY[symbol];
  const units = `${n[0]} ${n[0] === '1' ? one : many}`;
  if (cents === '00') return units;
  const c = cents.startsWith('0') ? cents.slice(1) : cents;
  return n[0] === '0' ? `${c} cents` : `${units} and ${c} cents`;
}

// "$1,234.56" -> "1 thousand 234 dollars and 56 cents"; "30¢" -> "30 cents".
export function currency(word) {
  const [body, suffix] = splitSuffix(word);
  if (body.startsWith('-') && CURRENCY[body[1]]) return currency(`${body[1]}-${body.slice(2)}${suffix}`);
  const split = splitSymbol(body);
  if (!split) return null;
  const [symbol, amount] = split;
  for (const [suf, scale] of SCALE_SUFFIXES) {
    if (amount.endsWith(suf) && isDigit(amount.slice(0, -suf.length).at(-1) ?? '')) {
      const spoken = withScale(amount.slice(0, -suf.length), scale, symbol);
      return spoken ? spoken + suffix : null;
    }
  }
  if (symbol !== '¢') {
    const cents = withCents(amount, symbol);
    if (cents) return cents + suffix;
  }
  const n = numberWords(amount);
  if (!n) return null;
  const [one, many] = CURRENCY[symbol];
  return `${n[0]} ${symbol === '¢' && amount === '1' ? one : many}${suffix}`;
}

// "$2 million" across two words.
function scaled(words) {
  if (words.length < 2) return null;
  const [scale, suffix] = splitSuffix(words[1]);
  const split = splitSymbol(words[0]);
  if (!split || !SCALE_WORDS.includes(scale)) return null;
  const spoken = withScale(split[1], scale, split[0]);
  return spoken ? [spoken + suffix, 2] : null;
}

// Three or more digit groups ("4 5 6"), or two after a "+" or with a leading zero, read digit by digit.
function digitRun(words) {
  const group = (s) => s.length >= 1 && s.length <= 8 && /^[0-9A-Z]+$/.test(s) && /[0-9]/.test(s);
  const plus = words[0]?.startsWith('+');
  const bodies = [];
  let end = '';
  for (const w of words) {
    let [body, suf] = splitSuffix(w);
    if (!bodies.length && plus) body = body.slice(1);
    if (!group(body)) break;
    bodies.push(body);
    if (suf) {
      end = suf;
      break;
    }
  }
  const years = bodies.every((b) => b.length === 4 && /^(19|20)/.test(b));
  const long = bodies.length >= 3 && !years;
  if (!long && !(bodies.length === 2 && (plus || bodies.some((b) => b.startsWith('0'))))) return null;
  const spelled = bodies.map(spell).join(', ');
  return [`${plus ? `plus ${spelled}` : spelled}${end}`, bodies.length];
}

const rewriteWord = (w) => numbers(w) ?? currency(w);

class Appender {
  constructor() {
    this.b = [];
  }

  keeps(last) {
    if ('"\'@+%#&'.includes(last)) return true;
    return last === '$' && this.b.length >= 2 && isDigit(this.b[this.b.length - 2]);
  }

  push(c) {
    if (c === '.' || c === ',') {
      while (this.b.length && (isSpace(this.b.at(-1)) || (isAsciiPunct(this.b.at(-1)) && !this.keeps(this.b.at(-1))))) this.b.pop();
    }
    this.b.push(c);
  }

  pushStr(s) {
    for (const c of s) this.push(c);
  }

  space() {
    if (this.b.length && !isSpace(this.b.at(-1))) this.push(' ');
  }

  popSpace() {
    while (this.b.length && isSpace(this.b.at(-1))) this.b.pop();
  }

  spoken(word) {
    this.space();
    this.pushStr(word);
    this.space();
  }

  endsWithOne() {
    let n = this.b.length;
    while (n && isSpace(this.b[n - 1])) n--;
    return this.b[n - 1] === '1' && !(n >= 2 && (isDigit(this.b[n - 2]) || this.b[n - 2] === '.' || this.b[n - 2] === ','));
  }

  endsWithSpacedDigit() {
    if (!this.b.length || !isSpace(this.b.at(-1))) return false;
    const last = [...this.b].reverse().find((c) => !isSpace(c));
    return Boolean(last && isDigit(last));
  }

  toString() {
    this.popSpace();
    return this.b.join('');
  }
}

function spellSymbols(word) {
  let j = word.length;
  while (j > 0 && isAsciiPunct(word[j - 1]) && word[j - 1] !== '/') j--;
  const body = word.slice(0, j);
  const tail = word.slice(j);
  const parts = body.split('/');
  if (parts.length > 1 && parts.every((p) => [...p].length >= 2 && [...p].every(isAlpha) && !SLASH_UNITS.has(p))) {
    return `${parts.join(' ')}${tail}`.split(' ').map((w) => rewriteWord(w) ?? w).join(' ');
  }
  if (!/[@+]/.test(word)) return word;
  const a = new Appender();
  for (const c of word) {
    if (c === '@') a.spoken('at');
    else if (c === '+') a.spoken('plus');
    else a.push(c);
  }
  return a.toString().split(' ').map((w) => rewriteWord(w) ?? w).join(' ');
}

function rewriteText(text) {
  const words = text.split(' ');
  const out = [];
  for (let i = 0; i < words.length; ) {
    const span = scaled(words.slice(i)) ?? digitRun(words.slice(i));
    if (span) {
      out.push(span[0]);
      i += span[1];
      continue;
    }
    out.push(rewriteWord(words[i]) ?? spellSymbols(words[i]));
    i++;
  }
  return out.join(' ');
}

function stripClosingQuote(text) {
  const t = text.replace(/"+$/, '');
  return t.endsWith("'") && /[.!?,;]$/.test(t.slice(0, -1)) ? t.slice(0, -1) : t;
}

// The official runtime's English normalization, for the rules ported here.
export function normalizeText(input) {
  const a = new Appender();
  const chars = [...input];
  let prev = null;
  for (let i = 0; i < chars.length; i++) {
    const c = chars[i];
    const next = chars[i + 1];
    if (DOUBLE_QUOTES.has(c)) a.push('"');
    else if (SINGLE_QUOTES.has(c)) a.push("'");
    else if ('‐‑‒―'.includes(c)) a.push('-');
    else if (c === '°') {
      a.spoken(a.endsWithOne() ? 'degree' : 'degrees');
      const scale = next === 'C' ? 'Celsius' : next === 'F' ? 'Fahrenheit' : null;
      if (scale && !isAlnum(chars[i + 2] ?? '')) {
        a.pushStr(scale);
        i++;
      }
    } else if (SYMBOLS[c]) a.spoken(SYMBOLS[c]);
    else if (c === '~' && isDigit(next ?? '')) a.spoken('about');
    else if (c === '*' && (prev === null || isSpace(prev)) && isDigit(next ?? '')) a.spoken('star');
    else if ('–*—[]{}•‣◦·→←↑↓➡➜'.includes(c)) a.space();
    else if (CURRENCY[c] && a.endsWithSpacedDigit() && !isDigit(next ?? '')) {
      a.popSpace();
      a.push(c);
    } else if (c === '…') a.push('.');
    else if (c === '=') a.spoken('equals');
    else if (c === ':' && prev !== null && !isSpace(prev) && next !== undefined && !isSpace(next)) a.push(':');
    else if (';:()'.includes(c)) {
      a.popSpace();
      a.push(',');
      a.space();
    } else if (isEmoji(c) || /\p{Cc}/u.test(c) || isSpace(c)) a.space();
    else a.push(c);
    prev = c;
  }
  return stripClosingQuote(rewriteText(a.toString()));
}

// tts_model.rs prepare_text_prompt: one line, capitalized, ending in punctuation.
// frames_after_eos: frames still generated after the model signals the end.
export function prepareTextPrompt(text) {
  const t = text.trim().replace(/[\n\r]/g, ' ').split(/\s+/).filter(Boolean).join(' ');
  if (!t) return { text: '', framesAfterEos: 3 };
  const framesAfterEos = t.split(' ').length <= 4 ? 3 : 1;
  let out = t[0].toUpperCase() + t.slice(1);
  if (isAlnum(out.at(-1))) out += '.';
  return { text: out, framesAfterEos };
}

// Upper bound on frames for a chunk: about 3 tokens per second plus 2 s, at 12.5 Hz.
export function frameBudget(numTokens, frameRate = 12.5) {
  return Math.ceil((numTokens / 3 + 2) * frameRate);
}

// Once the model reports EOS, a few more frames are kept so the voice closes out.
export class EosPolicy {
  constructor(framesAfterEos) {
    this.n = framesAfterEos;
    this.countdown = null;
  }

  shouldStop(isEos) {
    if (isEos && this.countdown === null) this.countdown = this.n;
    if (this.countdown === null) return false;
    if (this.countdown === 0) return true;
    this.countdown--;
    return false;
  }
}

// The largest chunk (in tokens) whose prompt and frames fit in `room` KV positions.
export function maxTokensFor(room, frameRate = 12.5) {
  let n = 0;
  while (n + 1 + frameBudget(n + 1, frameRate) <= room) n++;
  return n;
}

// Cuts `text` near its middle: after a sentence end, else after , ; :, else at a space.
function cutInTwo(text) {
  const mid = text.length / 2;
  const near = (re) => {
    let best = -1;
    for (const m of text.matchAll(re)) {
      const at = m.index + m[0].length;
      if (at > text.length * 0.25 && at < text.length * 0.75 && (best < 0 || Math.abs(at - mid) < Math.abs(best - mid))) best = at;
    }
    return best;
  };
  const at = [/[.!?]+\s/g, /[,;:]\s/g, /\s/g].map(near).find((x) => x > 0);
  return at > 0 ? [text.slice(0, at).trim(), text.slice(at).trim()] : [text];
}

// plan.rs: normalize, split into sentences, group them into chunks of at most
// `maxTokens` tokens, then prepare and tokenize each; a chunk still longer than
// `fitTo` tokens (the KV budget) is cut. -> [{ text, tokens, framesAfterEos, frameBudget }]
export function planChunks(tokenizer, input, { maxTokens = 50, fitTo = Infinity, normalize = true } = {}) {
  const normalized = normalize ? normalizeText(input) : input;
  const prepared = prepareTextPrompt(normalized).text.trim();
  if (!prepared) return [];
  const tokens = tokenizer.encode(prepared);
  const eos = new Set(tokenizer.encode('.!...?').slice(1));
  const bounds = [0];
  let prevEos = false;
  tokens.forEach((t, i) => {
    if (eos.has(t)) prevEos = true;
    else {
      if (prevEos) bounds.push(i);
      prevEos = false;
    }
  });
  bounds.push(tokens.length);
  const groups = [];
  let cur = '';
  let count = 0;
  for (let i = 0; i + 1 < bounds.length; i++) {
    const n = bounds[i + 1] - bounds[i];
    const sentence = tokenizer.decode(tokens.slice(bounds[i], bounds[i + 1]));
    if (!cur) [cur, count] = [sentence, n];
    else if (count + n > maxTokens) {
      groups.push(cur.trim());
      [cur, count] = [sentence, n];
    } else {
      cur += ` ${sentence}`;
      count += n;
    }
  }
  if (cur) groups.push(cur.trim());
  const chunk = (text) => {
    const p = prepareTextPrompt(text);
    const ids = tokenizer.encode(p.text);
    if (ids.length > fitTo) {
      const halves = cutInTwo(text);
      if (halves.length === 2) return halves.flatMap(chunk);
    }
    return [{ text: p.text, tokens: ids, framesAfterEos: p.framesAfterEos, frameBudget: frameBudget(ids.length) }];
  };
  return groups.filter(Boolean).flatMap(chunk);
}

// Gaussian noise for the flow head: `n` values with variance `temperature`,
// from a seeded uniform generator (Box-Muller).
export function gaussian(rng, n, temperature) {
  const out = new Float32Array(n);
  const std = Math.sqrt(Math.max(0, temperature));
  if (!std) return out;
  for (let i = 0; i < n; i += 2) {
    const u = 1 - rng();
    const v = rng();
    const r = Math.sqrt(-2 * Math.log(u)) * std;
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}
