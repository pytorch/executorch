// WebGL2 renderer for the camera effects: the camera frame and the person mask are
// uploaded once per animation frame, then every effect is a single full-screen
// fragment pass that samples both.
import { EFFECTS } from './effects.js';

const VERTEX = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const PRELUDE = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uVideo;
uniform sampler2D uMask;
uniform sampler2D uBg;
uniform vec2 uRes;
uniform vec2 uBgRes;
uniform vec2 uSrcRes;
uniform float uTime;
uniform float uSplit;
uniform float uMirror;
uniform float uHasMask;
uniform float uHasBg;
uniform vec3 uG0;
uniform vec3 uG1;
uniform vec3 uG2;
const vec3 LUMA = vec3(0.299, 0.587, 0.114);

vec2 src(vec2 uv) { return vec2(uMirror > 0.5 ? 1.0 - uv.x : uv.x, uv.y); }
vec3 video(vec2 uv) { return texture(uVideo, src(uv)).rgb; }
float rawMask(vec2 uv) { return texture(uMask, src(uv)).r; }
// The mask is a quarter of the frame's resolution or less: a small tent filter
// plus a soft threshold turns its bilinear upsample into a feathered edge.
float maskAt(vec2 uv) {
  if (uHasMask < 0.5) return 1.0;
  vec2 t = 1.4 / vec2(textureSize(uMask, 0));
  float m = rawMask(uv) * 4.0;
  m += rawMask(uv + vec2(t.x, 0.0)) + rawMask(uv - vec2(t.x, 0.0));
  m += rawMask(uv + vec2(0.0, t.y)) + rawMask(uv - vec2(0.0, t.y));
  return smoothstep(0.12, 0.88, m / 8.0);
}
vec3 gradientBg(vec2 uv) {
  float d = clamp(uv.x * 0.55 + uv.y * 0.45, 0.0, 1.0);
  vec3 c = d < 0.5 ? mix(uG0, uG1, d * 2.0) : mix(uG1, uG2, d * 2.0 - 1.0);
  return c + 0.06 * (1.0 - length(uv - vec2(0.3, 0.25)));
}
`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) || 'Shader failed to compile');
  return shader;
}

function texture(gl, filter) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  return t;
}

export class EffectsRenderer {
  constructor() {
    this.canvas = document.createElement('canvas');
    const gl = this.canvas.getContext('webgl2', { antialias: false, alpha: false, preserveDrawingBuffer: true, premultipliedAlpha: false });
    if (!gl) throw new Error('This browser cannot run WebGL2, which the effects are drawn with.');
    this.gl = gl;
    this.vs = compile(gl, gl.VERTEX_SHADER, VERTEX);
    this.programs = new Map();
    this.vao = gl.createVertexArray();
    this.tex = { video: texture(gl, gl.LINEAR), mask: texture(gl, gl.LINEAR), bg: texture(gl, gl.LINEAR) };
    this.srcSize = [1, 1];
    this.bgSize = [1, 1];
    this.hasMask = false;
    this.hasBg = false;
  }

  program(id) {
    if (!this.programs.has(id)) {
      const gl = this.gl;
      const fx = EFFECTS.find((e) => e.id === id);
      const fs = compile(gl, gl.FRAGMENT_SHADER, `${PRELUDE}\n${fx.glsl}\nvoid main() {\n  vec3 c = vUv.x < uSplit ? video(vUv) : effect(vUv);\n  outColor = vec4(c, 1.0);\n}`);
      const p = gl.createProgram();
      gl.attachShader(p, this.vs);
      gl.attachShader(p, fs);
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) || 'Shader failed to link');
      const u = {};
      for (const name of ['uVideo', 'uMask', 'uBg', 'uRes', 'uBgRes', 'uSrcRes', 'uTime', 'uSplit', 'uMirror', 'uHasMask', 'uHasBg', 'uG0', 'uG1', 'uG2']) u[name] = gl.getUniformLocation(p, name);
      this.programs.set(id, { p, u });
    }
    return this.programs.get(id);
  }

  // `el` is a video, image, or canvas; videos must have a decoded frame.
  setSource(el, w, h) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex.video);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, el);
    this.srcSize = [w, h];
  }

  setMask(alpha, w, h) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex.mask);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, alpha);
    this.hasMask = true;
  }

  clearMask() {
    this.hasMask = false;
  }

  setBackground(el, w, h) {
    const gl = this.gl;
    this.hasBg = Boolean(el);
    if (!el) return;
    gl.bindTexture(gl.TEXTURE_2D, this.tex.bg);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, el);
    this.bgSize = [w, h];
  }

  // Draws one effect into the renderer's canvas at w x h, then copies it to `target` (a 2D canvas).
  render(effectId, target, { width, height, split = 0, mirror = false, time = 0, gradient }) {
    const gl = this.gl;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    const { p, u } = this.program(effectId);
    gl.viewport(0, 0, width, height);
    gl.useProgram(p);
    gl.bindVertexArray(this.vao);
    [this.tex.video, this.tex.mask, this.tex.bg].forEach((t, i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, t);
    });
    gl.uniform1i(u.uVideo, 0);
    gl.uniform1i(u.uMask, 1);
    gl.uniform1i(u.uBg, 2);
    gl.uniform2f(u.uRes, width, height);
    gl.uniform2f(u.uSrcRes, ...this.srcSize);
    gl.uniform2f(u.uBgRes, ...this.bgSize);
    gl.uniform1f(u.uTime, time);
    gl.uniform1f(u.uSplit, split);
    gl.uniform1f(u.uMirror, mirror ? 1 : 0);
    gl.uniform1f(u.uHasMask, this.hasMask ? 1 : 0);
    gl.uniform1f(u.uHasBg, this.hasBg ? 1 : 0);
    gl.uniform3fv(u.uG0, gradient[0]);
    gl.uniform3fv(u.uG1, gradient[1]);
    gl.uniform3fv(u.uG2, gradient[2]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (target.width !== width || target.height !== height) {
      target.width = width;
      target.height = height;
    }
    target.getContext('2d').drawImage(this.canvas, 0, 0);
  }

  dispose() {
    this.gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
