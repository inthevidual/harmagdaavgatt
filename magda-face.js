/*
 * MagdaFace — one-axis mood animation from a single photo:
 * big smile (+1) … neutral (0) … deep sad frown (-1).
 *
 * Technique: one WebGL1 full-canvas quad. The fragment shader inverse-warps the
 * photo with gaussian "handles" on facial landmarks (mouth, cheeks, brows,
 * chin), plus a weighted head nod and a lid-slide blink. State is a few floats
 * in uniforms per frame.
 *
 * Fallback: the <img> inside the container stays visible if WebGL is missing,
 * the image fails, or the context is lost.
 *
 * Usage:
 *   const face = new MagdaFace(containerEl, { mood: 0 });
 *   face.setMood(1);    // big smile
 *   face.setMood(-1);   // deep frown
 */
(function (global) {
  'use strict';

  // Landmarks are in pixel space of the ORIGINAL photo (1478 × 1723).
  var W = 1478, H = 1723;

  // [cx, cy, rx, ry] — gaussian handles. Index order is used in _draw().
  // Radii are sized so max displacement stays below ~0.9·radius (no folding).
  var HANDLES = [
    [585, 606, 70, 60],  // 0 mouth corner L
    [765, 602, 70, 60],  // 1 mouth corner R
    [675, 622, 70, 22],  // 2 lower lip
    [675, 600, 70, 18],  // 3 upper lip
    [565, 505, 80, 60],  // 4 cheek apple L
    [790, 500, 80, 60],  // 5 cheek apple R
    [610, 350, 55, 40],  // 6 brow inner L
    [730, 348, 55, 40],  // 7 brow inner R
    [480, 352, 55, 36],  // 8 brow outer L
    [855, 350, 55, 36],  // 9 brow outer R
    [675, 720, 100, 55], // 10 chin
    [558, 420, 52, 16],  // 11 lower lid L
    [770, 413, 56, 16]   // 12 lower lid R
  ];
  var N = HANDLES.length;

  // Eyes for the blink: [cx, cy, halfWidth, upperLidHeight, lowerLidHeight]
  var EYE_L = [558, 398, 50, 18, 16];
  var EYE_R = [770, 392, 56, 19, 16];

  var VERT = [
    'attribute vec2 aPos;',
    'varying vec2 vUv;',
    'void main(){',
    '  vUv = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);',
    '  gl_Position = vec4(aPos, 0.0, 1.0);',
    '}'
  ].join('\n');

  var FRAG = [
    '#ifdef GL_FRAGMENT_PRECISION_HIGH',
    'precision highp float;',
    '#else',
    'precision mediump float;',
    '#endif',
    '#define N ' + N,
    'uniform sampler2D uTex;',
    'uniform vec4 uK[N];',      // handle cx, cy, rx, ry
    'uniform vec2 uD[N];',      // handle displacement (content motion, px)
    'uniform vec3 uHead;',      // tilt (rad), nod (px), sway x (px)
    'uniform vec2 uBlink;',     // lid closure L, R (0..1)
    'uniform vec4 uEyeL;',      // cx, cy, halfW, upperH
    'uniform vec4 uEyeR;',
    'uniform vec2 uLower;',     // lowerH L, R
    'varying vec2 vUv;',
    'const vec2 RES = vec2(' + W + '.0, ' + H + '.0);',
    '',
    // Lid slides down: the lash band (LA px above the lid line) is translated
    // rigidly to the current lid position; the skin above it stretches.
    'float blinkY(vec2 p, vec4 eye, float lowerH, float b){',
    '  float ex = (p.x - eye.x) / eye.z;',
    '  float f = max(0.0, 1.0 - ex * ex);',
    '  float yTop = eye.y - eye.w * f;',
    '  float lid = yTop + b * (eye.w + lowerH) * f;',
    '  float L = 30.0;',
    '  float LA = 9.0;',
    '  float yB = yTop - L;',
    '  if (p.y < yB || f <= 0.0) return p.y;',
    '  float stretch = yB + (p.y - yB) * (L - LA) / max(lid - LA - yB, 1e-3);',
    '  float warped = p.y > lid - LA ? p.y - (lid - yTop) : stretch;',
    '  float m = smoothstep(lid + 0.8, lid - 0.8, p.y);',
    '  return mix(p.y, warped, m);',
    '}',
    '',
    'void main(){',
    '  vec2 p = vUv * RES;',
    // Head: rotate around the neck, weighted to the head so shoulders stay put.
    '  vec2 hd = (p - vec2(675.0, 430.0)) / vec2(380.0, 470.0);',
    '  float hw = exp(-dot(hd, hd));',
    '  vec2 piv = vec2(675.0, 830.0);',
    '  float a = -uHead.x * hw;',
    '  vec2 q = p - piv;',
    '  q = vec2(q.x * cos(a) - q.y * sin(a), q.x * sin(a) + q.y * cos(a));',
    '  vec2 src = piv + q - vec2(uHead.z, uHead.y) * hw;',
    // Expression handles (inverse warp: sample from where content came from).
    '  vec2 d = vec2(0.0);',
    '  for (int i = 0; i < N; i++) {',
    '    vec2 k = (src - uK[i].xy) / uK[i].zw;',
    '    d += uD[i] * exp(-dot(k, k));',
    '  }',
    '  src -= d;',
    '  src.y = blinkY(src, uEyeL, uLower.x, uBlink.x);',
    '  src.y = blinkY(src, uEyeR, uLower.y, uBlink.y);',
    '  gl_FragColor = texture2D(uTex, clamp(src / RES, 0.0, 1.0));',
    '}'
  ].join('\n');

  function lerp(a, b, t) { return a + (b - a) * t; }
  function clamp1(v) { return Math.max(-1, Math.min(1, v)); }

  function MagdaFace(container, opts) {
    opts = opts || {};
    this.container = container;
    this.img = container.querySelector('img');
    this.reduced = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.target = clamp1(opts.mood || 0);
    this.mood = this.target; this.vel = 0;
    // idle state
    this.blink = 0; this.blinkT = 0; this.nextBlink = 1.5;
    this.time = 0; this.visible = true; this.raf = 0; this.last = 0;
    this._frame = this._frame.bind(this);

    if (!this.img) return;
    var self = this;
    if (this.img.complete && this.img.naturalWidth) this._init();
    else this.img.addEventListener('load', function () { self._init(); }, { once: true });
  }

  /** mood: -1 = deep sad frown, 0 = neutral, 1 = big smile */
  MagdaFace.prototype.setMood = function (mood) {
    this.target = clamp1(mood);
    if (this.reduced) { this.mood = this.target; this.vel = 0; }
    this._kick();
  };

  MagdaFace.prototype._init = function () {
    var canvas = document.createElement('canvas');
    var gl = canvas.getContext('webgl', { premultipliedAlpha: true, antialias: false, alpha: true }) ||
             canvas.getContext('experimental-webgl');
    if (!gl) return; // static <img> stays as fallback

    function sh(type, src) {
      var s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    }
    try {
      var prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
      gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    } catch (err) {
      if (global.console) console.warn('MagdaFace: falling back to static image', err);
      return;
    }
    gl.useProgram(prog);

    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(prog, 'aPos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    var tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    try {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.img);
    } catch (err) { // e.g. cross-origin / file:// taint
      if (global.console) console.warn('MagdaFace: texture upload failed, static fallback', err);
      return;
    }

    var u = {};
    ['uK', 'uD', 'uHead', 'uBlink', 'uEyeL', 'uEyeR', 'uLower'].forEach(function (n) {
      u[n] = gl.getUniformLocation(prog, n);
    });
    var k = new Float32Array(N * 4);
    HANDLES.forEach(function (h, i) { k.set(h, i * 4); });
    gl.uniform4fv(u.uK, k);
    gl.uniform4f(u.uEyeL, EYE_L[0], EYE_L[1], EYE_L[2], EYE_L[3]);
    gl.uniform4f(u.uEyeR, EYE_R[0], EYE_R[1], EYE_R[2], EYE_R[3]);
    gl.uniform2f(u.uLower, EYE_L[4], EYE_R[4]);
    gl.clearColor(0, 0, 0, 0);

    this.gl = gl; this.u = u; this.canvas = canvas;
    this.dOff = new Float32Array(N * 2);

    canvas.setAttribute('aria-hidden', 'true');
    canvas.className = 'magda-canvas';
    this.container.appendChild(canvas);
    this.container.classList.add('magda-live');

    var self = this;
    canvas.addEventListener('webglcontextlost', function (e) {
      e.preventDefault();
      cancelAnimationFrame(self.raf); self.raf = 0; self.gl = null;
      self.container.classList.remove('magda-live');
    });

    this._resize();
    if ('ResizeObserver' in global) new ResizeObserver(function () { self._resize(); self._kick(); }).observe(this.container);
    else global.addEventListener('resize', function () { self._resize(); self._kick(); });

    if ('IntersectionObserver' in global) {
      new IntersectionObserver(function (es) {
        self.visible = es[0].isIntersecting; self._kick();
      }).observe(this.container);
    }
    document.addEventListener('visibilitychange', function () { self._kick(); });
    this._kick();
  };

  MagdaFace.prototype._resize = function () {
    if (!this.gl) return;
    var dpr = Math.min(global.devicePixelRatio || 1, 2);
    var r = this.container.getBoundingClientRect();
    var w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h;
      this.gl.viewport(0, 0, w, h);
    }
  };

  MagdaFace.prototype._kick = function () {
    if (!this.gl || this.raf) return;
    if (!this.visible || document.hidden) return;
    this.last = 0;
    this.raf = requestAnimationFrame(this._frame);
  };

  MagdaFace.prototype._frame = function (now) {
    this.raf = 0;
    if (!this.gl) return;
    var dt = this.last ? Math.min(0.05, (now - this.last) / 1000) : 1 / 60;
    this.last = now;
    this.time += dt;
    var animating = this._update(dt);
    this._draw();
    // Keep running while idle animation is on; with reduced motion, stop once settled.
    if (animating && this.visible && !document.hidden) this.raf = requestAnimationFrame(this._frame);
  };

  MagdaFace.prototype._update = function (dt) {
    // Critically damped spring toward the target mood.
    var k = 30, c = 2 * Math.sqrt(k);
    this.vel += (k * (this.target - this.mood) - c * this.vel) * dt;
    this.mood += this.vel * dt;
    var moving = Math.abs(this.vel) > 1e-3 || Math.abs(this.target - this.mood) > 1e-3;
    if (this.reduced) return moving;

    // Natural blinks every ~2–6 s, occasionally a double blink.
    this.nextBlink -= dt;
    if (this.nextBlink <= 0 && this.blinkT <= 0) {
      this.blinkT = 0.0001;
      this.nextBlink = Math.random() < 0.15 ? 0.3 : lerp(2, 6, Math.random());
    }
    if (this.blinkT > 0) {
      this.blinkT += dt / 0.16;
      this.blink = Math.sin(Math.min(1, this.blinkT) * Math.PI);
      if (this.blinkT >= 1) { this.blinkT = 0; this.blink = 0; }
    }
    return true;
  };

  MagdaFace.prototype._draw = function () {
    var gl = this.gl, t = this.time, d = this.dOff;
    var sp = Math.max(0, this.mood), sn = Math.max(0, -this.mood);
    var idle = this.reduced ? 0 : 1;

    function set(i, x, y) { d[i * 2] = x; d[i * 2 + 1] = y; }
    // Smile: corners travel outward and diagonally toward the cheekbones,
    // lips stretch, cheeks lift and push the lower lids up. No lid closing.
    // Sad: corners drop modestly, lower lip pushes up, chin bunches, inner brows rise.
    set(0, -22 * sp + 4 * sn, -20 * sp + 20 * sn);
    set(1, 22 * sp - 4 * sn, -20 * sp + 20 * sn);
    set(2, 0, 2 * sp - 6 * sn);
    set(3, 0, -2 * sp - 2 * sn);
    set(4, -6 * sp, -12 * sp + 4 * sn);
    set(5, 6 * sp, -12 * sp + 4 * sn);
    set(6, 3 * sn, -3 * sp - 12 * sn);
    set(7, -3 * sn, -3 * sp - 12 * sn);
    set(8, 0, 4 * sn);
    set(9, 0, 4 * sn);
    set(10, 0, -8 * sn);
    set(11, 0, -5 * sp);
    set(12, 0, -5 * sp);
    gl.uniform2fv(this.u.uD, d);

    var sway = idle * (Math.sin(t * 0.7) * 0.006 + Math.sin(t * 1.9) * 0.002);
    var bob = idle * Math.sin(t * 1.1) * 1.5;
    gl.uniform3f(this.u.uHead, -0.02 * sn + sway, -3 * sp + 10 * sn + bob, 0);

    // Slightly heavier upper lids when sad, plus the blink.
    var base = sn * 0.12;
    var b = Math.min(1, base + (1 - base) * this.blink);
    gl.uniform2f(this.u.uBlink, b, b);

    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  global.MagdaFace = MagdaFace;
})(window);
