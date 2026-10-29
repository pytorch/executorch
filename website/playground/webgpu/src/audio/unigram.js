// A Unigram (sentencepiece) tokenizer for a Hugging Face tokenizer.json, enough
// for the Phonon text-to-speech model: Prepend + Metaspace pre-tokenization,
// Viterbi segmentation, and byte fallback for characters without a piece. It
// reproduces the `tokenizers` library's ids (encode without special tokens).
const UNK_PENALTY = 10;

export class UnigramTokenizer {
  // json: the parsed tokenizer.json (model.type 'Unigram').
  constructor(json) {
    const model = json.model;
    if (model?.type !== 'Unigram') throw new Error(`Expected a Unigram tokenizer, got ${model?.type}`);
    this.space = json.pre_tokenizer?.replacement ?? '▁';
    this.unkId = model.unk_id ?? 0;
    this.byteFallback = Boolean(model.byte_fallback);
    this.pieces = new Map();
    this.byId = [];
    this.bytes = new Map();
    let minScore = Infinity;
    let maxLen = 1;
    model.vocab.forEach(([piece, score], id) => {
      this.pieces.set(piece, { id, score });
      this.byId[id] = piece;
      minScore = Math.min(minScore, score);
      maxLen = Math.max(maxLen, [...piece].length);
      const m = /^<0x([0-9A-F]{2})>$/.exec(piece);
      if (m) this.bytes.set(parseInt(m[1], 16), id);
    });
    this.maxLen = maxLen;
    this.unkScore = minScore - UNK_PENALTY;
    this.utf8 = new TextEncoder();
  }

  // Prepend normalizer, then Metaspace (spaces -> '▁', split before each '▁').
  preTokenize(text) {
    if (!text) return [];
    // The normalizer always prepends one '▁'; Metaspace then adds none.
    const s = this.space + text.replaceAll(' ', this.space);
    const out = [];
    let cur = '';
    for (const ch of s) {
      if (ch === this.space && cur) {
        out.push(cur);
        cur = '';
      }
      cur += ch;
    }
    if (cur) out.push(cur);
    return out;
  }

  // Best-scoring segmentation of one pre-token into vocabulary ids.
  viterbi(word) {
    const chars = [...word];
    const n = chars.length;
    const best = new Float64Array(n + 1).fill(-Infinity);
    const from = new Int32Array(n + 1).fill(-1);
    const tok = new Array(n + 1).fill(null);
    best[0] = 0;
    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;
      let single = false;
      let piece = '';
      for (let len = 1; len <= this.maxLen && i + len <= n; len++) {
        piece += chars[i + len - 1];
        const hit = this.pieces.get(piece);
        if (!hit) continue;
        if (len === 1) single = true;
        const s = best[i] + hit.score;
        if (s > best[i + len]) {
          best[i + len] = s;
          from[i + len] = i;
          tok[i + len] = hit.id;
        }
      }
      if (!single) {
        const s = best[i] + this.unkScore;
        if (s > best[i + 1]) {
          best[i + 1] = s;
          from[i + 1] = i;
          tok[i + 1] = -1 - i; // unknown character at i
        }
      }
    }
    const ids = [];
    for (let j = n; j > 0; j = from[j]) {
      const t = tok[j];
      if (t >= 0) ids.push([t]);
      else ids.push(this.#unknown(chars[-1 - t]));
    }
    return ids.reverse().flat();
  }

  #unknown(ch) {
    if (!this.byteFallback) return [this.unkId];
    const bytes = this.utf8.encode(ch);
    const ids = [...bytes].map((b) => this.bytes.get(b));
    return ids.every((id) => id !== undefined) ? ids : [this.unkId];
  }

  encode(text) {
    return this.preTokenize(text).flatMap((w) => this.viterbi(w));
  }

  // Metaspace + byte-fallback decoding: pieces joined, '▁' read as a space,
  // byte pieces reassembled as UTF-8, the leading space dropped.
  decode(ids) {
    const out = [];
    let pending = [];
    const flush = () => {
      if (pending.length) out.push(new TextDecoder().decode(new Uint8Array(pending)));
      pending = [];
    };
    for (const id of ids) {
      const piece = this.byId[id] ?? '';
      const m = /^<0x([0-9A-F]{2})>$/.exec(piece);
      if (m) {
        pending.push(parseInt(m[1], 16));
        continue;
      }
      flush();
      out.push(piece.replaceAll(this.space, ' '));
    }
    flush();
    const text = out.join('');
    return text.startsWith(' ') ? text.slice(1) : text;
  }
}
