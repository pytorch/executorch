// Minimal element builder. Classes are always passed as `class: '...'` string
// literals so scripts/check-wiring.mjs can verify every one has a CSS rule.
export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value == null || value === false) continue;
      if (key === 'class') el.className = value;
      else if (key === 'html') el.innerHTML = value;
      else if (key === 'dataset') Object.assign(el.dataset, value);
      else if (key === 'style' && typeof value === 'object') Object.assign(el.style, value);
      else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2).toLowerCase(), value);
      else if (key === 'value') el.value = value;
      else if (value === true) el.setAttribute(key, '');
      else el.setAttribute(key, String(value));
    }
  }
  appendChildren(el, children);
  return el;
}

export function appendChildren(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

// replaceChildren() would stringify null/false into text; this skips them like h().
export function replace(el, ...children) {
  el.replaceChildren();
  return appendChildren(el, children);
}

export function svg(markup) {
  const tpl = document.createElement('template');
  tpl.innerHTML = markup.trim();
  return tpl.content.firstElementChild;
}

let uidCounter = 0;
// SVG gradient/pattern ids are document-global; suffix them per instance so
// several cards on screen never cross-wire their fills.
export function uid(prefix = 'u') {
  uidCounter += 1;
  return `${prefix}${uidCounter}`;
}
