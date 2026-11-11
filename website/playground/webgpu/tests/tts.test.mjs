import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import test from 'node:test';
import { currency, EosPolicy, frameBudget, gaussian, maxTokensFor, normalizeText, numbers, planChunks, prepareTextPrompt } from '../src/audio/tts-text.js';
import { UnigramTokenizer } from '../src/audio/unigram.js';
import { mulberry32 } from '../src/lib/prng.js';
import { embedRows, runChunk } from '../src/runtime/phonon.js';

// Expected readings below are the official runtime's own test cases (ptts 0.4.0,
// src/preprocess.rs and preprocess/rewrite/{numbers,currency}.rs), English only.

test('numbers read out with scale words, codes spelled, ranges joined', () => {
  const cases = [
    ['0', '0'], ['123', '123'], ['2024', '2024'], ['1234', '1 thousand 234'], ['1,234.56...', '1 thousand 234 point 56...'],
    ['3.14', '3 point 14'], ['12.10', '12 point 10'], ['-12.00', 'minus 12 point 00'], ['1000000', '1 million'],
    ['2500000', '2 million 500 thousand'], ['2,500,000,', '2 million 500 thousand,'], ['1002003004', '1 billion 2 million 3 thousand 4'],
    ['-4500', 'minus 4 thousand 500'], ['-12.', 'minus 12.'], ['007', 'zero zero seven'], ['0600.', 'zero six zero zero.'],
    ['7,000-ton', '7 thousand-ton'], ['20-year-old', '20-year-old'], ['20-25', '20 to 25'], ['10-5', null], ['22-22', null],
    ['3-5', null], ['12:30', null], ['10-15', '10 to 15'], ['1445-1450', '1445 to 1450'], ['2024-25', '2024 to 25'],
    ['1999-1998', null], ['3.5%', null], ['1234567890123', null], ['hello', null], ['', null], ['12,345', '12 thousand 345'],
    ['1,2,3', '123'], ['1,23', '123'], ['1234,567', '1 million 234 thousand 567'], ['1,999', '1999'], ['0,123', '123'],
  ];
  for (const [input, want] of cases) assert.equal(numbers(input), want, input);
});

test('amounts of money read with their currency', () => {
  const cases = [
    ['$1234', '1 thousand 234 dollars'], ['$1,234.56', '1 thousand 234 dollars and 56 cents'], ['$1,234.56...', '1 thousand 234 dollars and 56 cents...'],
    ['$4.59', '4 dollars and 59 cents'], ['$0.01', '1 cents'], ['$0.10', '10 cents'], ['$1.00', '1 dollar'], ['$1.10', '1 dollar and 10 cents'],
    ['$3.14', '3 dollars and 14 cents'], ['$0.00', '0 dollars'], ['$1', '1 dollars'], ['£1.50', '1 pound and 50 cents'], ['€500...', '500 euros...'],
    ['€0', '0 euros'], ['€0.5', '0 point 5 euros'], ['£250', '250 pounds'], ['30¢', '30 cents'], ['30¢.', '30 cents.'], ['1¢', '1 cent'],
    ['99¢!', '99 cents!'], ['1.50¢', '1 point 50 cents'], ['1234$', '1 thousand 234 dollars'], ['$-500', 'minus 500 dollars'],
    ['-$500.', 'minus 500 dollars.'], ['$2000000', '2 million dollars'],
  ];
  for (const [input, want] of cases) assert.equal(currency(input), want, input);
  for (const input of ['1234', '$abc', '$', '$5€', '5$5', '$007', '$1.2.3', '-$', '--$5']) assert.equal(currency(input), null, input);
});

test('the character pass folds quotes, dashes, brackets and emoji, and spells symbols', () => {
  const cases = [
    ['Hello, world!', 'Hello, world!'], ['', ''], ['“hello” world it’s', '"hello" world it\'s'], ['«hello» ‹world›', '"hello" "world'],
    ['a‐b‑c‒d―e', 'a-b-c-d-e'], ['a–b—c', 'a b c'], ['foo (bar) [baz] {qux} *quux*', 'foo, bar, baz qux quux'],
    ['• ‣ ◦ · a→b←c↑d↓e ➡ ➜', 'a b c d e'], ['wait… a…b', 'wait. a.b'], ['user@host @home', 'user at host at home'],
    ['café résumé 日本語', 'café résumé 日本語'], ['hello 😀 flag 🇫🇷 sun ☀', 'hello flag sun'], ['sun ☀️ dev 👩‍💻 done', 'sun dev done'],
    ['Fish &, chips &.', 'Fish &, chips &.'], ['a;b:c', 'a, b:c'], ['time: 10:30', 'time, 10:30'], ['a :b', 'a, b'], ['note:', 'note,'],
    [':start', ', start'], ['hello (world)', 'hello, world,'], ['foo ; bar  :  baz', 'foo, bar, baz'], ['a   b\t\tc\n\nd', 'a b c d'],
    ['“Hello”; please email user@host (now)… 🚀', '"Hello", please email user at host, now.'],
    ['The conference will be held on Tuesday, March 15th at 3:30 PM.', 'The conference will be held on Tuesday, March 15th at 3:30 PM.'],
    ['I paid 1500.', 'I paid 1 thousand 500.'], ['It was $2 million.', 'It was 2 million dollars.'], ['x = 1+2', 'x equals 1 plus 2'],
    ['and/or', 'and or'], ['10 km/h', '10 km/h'], ['21°C and 1°', '21 degrees Celsius and 1 degree'],
  ];
  for (const [input, want] of cases) assert.equal(normalizeText(input), want, input);
});

test('prompts are capitalized and terminated; short ones keep a longer tail', () => {
  const cases = [
    ['', '', 3], ['  \n ', '', 3], ['not a thing', 'Not a thing.', 3], ['one two three four five', 'One two three four five.', 1],
    ['is it?', 'Is it?', 3], ['hello world!', 'Hello world!', 3], ['call me at 5', 'Call me at 5.', 3],
    ['one\r\ntwo   three\nfour five', 'One two three four five.', 1], ['éclair au chocolat', 'Éclair au chocolat.', 3],
  ];
  for (const [input, text, frames] of cases) assert.deepEqual(prepareTextPrompt(input), { text, framesAfterEos: frames }, input);
});

test('frame budget, KV room and the EOS tail', () => {
  assert.equal(frameBudget(14), 84);
  assert.equal(frameBudget(0), 25);
  // 125 voice positions in a 512-position cache leave room for 70 tokens and their frames.
  assert.equal(maxTokensFor(387), 70);
  const eos = new EosPolicy(1);
  assert.deepEqual([false, true, false, false].map((e) => eos.shouldStop(e)), [false, false, true, true]);
  const short = new EosPolicy(3);
  assert.deepEqual([true, false, false, false].map((e) => short.shouldStop(e)), [false, false, false, true]);
});

test('runChunk keeps the EOS frame and its tail, and stops at the budget', async () => {
  const logits = [0, 1, 5, 9, 9, 9];
  const seen = [];
  const frames = await runChunk({ step: async (k) => (seen.push(k), { eosLogit: logits[k] }), frameBudget: 6, framesAfterEos: 1, eosThreshold: 4 });
  assert.equal(frames, 4);
  assert.deepEqual(seen, [0, 1, 2, 3]);
  assert.equal(await runChunk({ step: async () => ({ eosLogit: 0 }), frameBudget: 5, framesAfterEos: 1, eosThreshold: 4 }), 5);
  const ac = new AbortController();
  ac.abort();
  await assert.rejects(runChunk({ step: async () => ({ eosLogit: 0 }), frameBudget: 5, framesAfterEos: 1, eosThreshold: 4, signal: ac.signal }), { name: 'AbortError' });
});

test('bf16 embedding rows widen exactly to f32', () => {
  const bf16 = (x) => new Uint32Array(new Float32Array([x]).buffer)[0] >>> 16;
  const table = Uint16Array.from([1, -2, 0.5, 3.25, -0.125, 1024].map(bf16));
  assert.deepEqual(Array.from(embedRows(table, [2, 0], 2)), [-0.125, 1024, 1, -2]);
});

test('noise is Gaussian with the requested variance, and zero at temperature 0', () => {
  const x = gaussian(mulberry32(7), 20000, 0.3);
  const mean = x.reduce((a, b) => a + b) / x.length;
  const v = x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length;
  assert.ok(Math.abs(mean) < 0.02, mean);
  assert.ok(Math.abs(v - 0.3) < 0.02, v);
  assert.ok(gaussian(mulberry32(1), 32, 0).every((n) => n === 0));
});

// A three-piece vocabulary: segmentation, unknown characters as bytes, decoding.
const TINY = {
  pre_tokenizer: { type: 'Metaspace', replacement: '▁' },
  model: {
    type: 'Unigram', unk_id: 0, byte_fallback: true,
    vocab: [['<unk>', 0], ['▁', -3], ['▁hello', -1], ['▁he', -2], ['llo', -2], ['.', -1], ...Array.from({ length: 256 }, (_, b) => [`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`, 0])],
  },
};

test('Unigram: best segmentation, byte fallback, decode', () => {
  const tok = new UnigramTokenizer(TINY);
  assert.deepEqual(tok.encode('hello.'), [2, 5]);
  assert.deepEqual(tok.encode('hello  hello'), [2, 1, 2]);
  const ids = tok.encode('hé');
  assert.equal(tok.decode(ids), 'hé');
  assert.deepEqual(tok.encode(''), []);
});

// The ids the official runtime speaks for these inputs (normalization, sentence
// chunks, tokenization); each was checked against ptts 0.4.0 audio, sample for sample.
const PARITY = [["The quick brown fox jumps over the lazy dog.", [[776, 1417, 2600, 1657, 879, 2503, 1971, 662, 816, 665, 993, 3358, 1852, 663]]], ["Hello, world!", [[3248, 661, 956, 1367]]], ["It costs $4.59, not $1,234.56.", [[735, 1999, 662, 660, 924, 660, 1734, 662, 670, 660, 856, 905, 3226, 662, 661, 702, 660, 748, 1290, 2071, 660, 769, 873, 924, 660, 1734, 662, 670, 660, 856, 978, 3226, 662, 663]]], ["In 2024 we sold 2500000 units (about 12,345 per day).", [[999, 660, 769, 720, 769, 924, 682, 3267, 660, 769, 1472, 824, 660, 856, 720, 720, 1290, 2071, 3377, 662, 661, 710, 660, 748, 769, 1290, 2071, 660, 873, 924, 856, 1249, 871, 663]]], ["Call 007 at 3:30; it's 20-25 minutes away.", [[1463, 749, 3343, 3343, 1574, 716, 660, 873, 462, 873, 720, 661, 675, 664, 662, 660, 769, 720, 667, 660, 769, 856, 660, 1232, 662, 1065, 663]]], ["We met @noon & left \u2014 fast.", [[768, 704, 674, 716, 660, 3739, 660, 3493, 1374, 1979, 663]]], ["Temperature: 21\u00b0C, ~5 degrees warmer.", [[1909, 684, 1601, 2661, 661, 660, 769, 748, 660, 2736, 662, 840, 916, 662, 804, 1010, 661, 710, 660, 856, 660, 2736, 662, 3291, 707, 663]]], ["The quick brown fox jumps over the lazy dog. This voice is generated by a small model running entirely on your device. Please call me back when you have a moment, and we can talk about the plan.", [[776, 1417, 2600, 1657, 879, 2503, 1971, 662, 816, 665, 993, 3358, 1852, 663, 919, 1782, 677, 1271, 1139, 795, 668, 1424, 2366, 1577, 673, 660, 1857, 728, 689, 705, 660, 3616, 663], [660, 2988, 962, 704, 778, 745, 669, 690, 668, 660, 1146, 661, 670, 682, 708, 791, 710, 665, 1381, 663]]]];

const TOKENIZER = process.env.PHONON_TOKENIZER;
test('text pipeline matches the official ids (needs PHONON_TOKENIZER=<tokenizer.json>)', { skip: !TOKENIZER || !existsSync(TOKENIZER) }, () => {
  const tok = new UnigramTokenizer(JSON.parse(readFileSync(TOKENIZER, 'utf8')));
  for (const [input, chunks] of PARITY) assert.deepEqual(planChunks(tok, input).map((c) => c.tokens), chunks, input);
});
