function fxOn(teardownFns, target, eventName, handler, options) {
    if (!target || !target.addEventListener || !target.removeEventListener) return;
    target.addEventListener(eventName, handler, options);
    teardownFns.push(() => target.removeEventListener(eventName, handler, options));
}

function fxOffAll(teardownFns) {
    while (teardownFns.length > 0) {
        const off = teardownFns.pop();
        try {
            off();
        } catch (_) {}
    }
}

function fxEsc(s) {
    if (typeof s !== 'string') s = String(s);
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Pick a readable ink for a given background hex. Uses WCAG 2.x relative
// luminance and returns one of two standard inks — dark #111111 or light
// #f8f8f8 — whichever has higher contrast against the background. Returns
// null for malformed input so callers can fall back to theme defaults.
function fxReadableTextColor(hex) {
    if (typeof hex !== 'string' || hex.charAt(0) !== '#' || hex.length !== 7) return null;
    const r = parseInt(hex.substring(1, 3), 16) / 255;
    const g = parseInt(hex.substring(3, 5), 16) / 255;
    const b = parseInt(hex.substring(5, 7), 16) / 255;
    if (!isFinite(r) || !isFinite(g) || !isFinite(b)) return null;
    const lin = (c) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
    const L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
    return L > 0.179 ? '#111111' : '#f8f8f8';
}

const THEMES = {
    'light': {
        bg: '#ffffff',
        text: '#000000',
        textMuted: '#666666',
        nodeFill: '#66ccee',
        nodeInput: '#75dcfe',
        nodeOutput: '#75dcfe',
        nodeSelected: '#fbc02d ',
        edgeNormal: '#333333',
        edgeInput: '#ff9800',
        edgeOutput: '#ff9800',
        edgeHover: '#e91e63',
        minimapBox: 'rgba(255, 0, 0, 0.1)',
        minimapBorder: 'red',
        uiBg: 'rgba(255, 255, 255, 0.95)',
        uiBorder: '#cccccc',
        uiHover: '#f0f8ff',
        legendBg: 'rgba(255, 255, 255, 0.8)'
    },
    'dark': {
        bg: '#1e1e1e',
        text: '#ffffff',
        textMuted: '#cccccc',
        nodeFill: '#0277a1',
        nodeInput: '#1287b1',
        nodeOutput: '#1287b1',
        nodeSelected: '#ffeb3b',
        edgeNormal: '#cccccc',
        edgeInput: '#ffb74d',
        edgeOutput: '#ffb74d',
        edgeHover: '#ff4081',
        minimapBox: 'rgba(255, 100, 100, 0.3)',
        minimapBorder: '#ff5555',
        uiBg: 'rgba(30, 30, 30, 0.95)',
        uiBorder: '#777777',
        uiHover: '#333333',
        legendBg: 'rgba(30, 30, 30, 0.8)'
    }
};
