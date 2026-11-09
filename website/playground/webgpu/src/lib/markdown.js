// A small Markdown subset for model replies: fenced code, inline code, bold,
// italics, headings, lists, and paragraphs. Input is HTML-escaped first and
// links are never generated, so model output cannot inject markup.
const escape = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

function inline(s) {
  return s
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
}

export function renderMarkdown(src) {
  const out = [];
  const parts = src.split(/^```[^\n]*\n?/m);
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      // An unterminated fence while streaming still renders as code.
      out.push(`<pre class="md-pre"><code>${escape(part.replace(/\n$/, ''))}</code></pre>`);
      return;
    }
    let list = null;
    let para = [];
    const flushPara = () => {
      if (para.length) out.push(`<p>${inline(escape(para.join('\n'))).replace(/\n/g, '<br>')}</p>`);
      para = [];
    };
    const flushList = () => {
      if (list) out.push(`<${list.tag}>${list.items.map((x) => `<li>${inline(escape(x))}</li>`).join('')}</${list.tag}>`);
      list = null;
    };
    for (const line of part.split('\n')) {
      const bullet = /^\s*[-*•]\s+(.*)$/.exec(line);
      const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
      const heading = /^(#{1,4})\s+(.*)$/.exec(line);
      if (bullet || numbered) {
        flushPara();
        const tag = bullet ? 'ul' : 'ol';
        if (list?.tag !== tag) flushList();
        list ??= { tag, items: [] };
        list.items.push((bullet ?? numbered)[1]);
      } else if (heading) {
        flushPara();
        flushList();
        out.push(`<h4>${inline(escape(heading[2]))}</h4>`);
      } else if (!line.trim()) {
        flushPara();
        flushList();
      } else {
        flushList();
        para.push(line);
      }
    }
    flushPara();
    flushList();
  });
  return out.join('');
}
