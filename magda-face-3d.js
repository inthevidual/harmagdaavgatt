/*
 * MagdaFace3D — 3D face rig fitted to the photo, mood from deep frown (-1)
 * to big smile (+1).
 *
 * Pipeline (offline, see tools/): MediaPipe landmarks → ICT-FaceKit morphable
 * model fitted to the photo (identity + pose + her photo expression) → smile,
 * frown and blink morph targets from ICT's scan-based expression shapes →
 * assets/magda3d.bin.
 *
 * Runtime: raw WebGL1, no libraries.
 *   - photo drawn as background (hair, ears, clothes)
 *   - fitted mesh drawn on top, projectively textured with the same photo, so
 *     at mood 0 it is pixel-identical; morphs move skin with its texture
 *   - skin relit by the ratio of new/old normal shading → real creases
 *   - teeth, gums, mouth cavity from ICT geometry, procedurally shaded
 *   - outer face border fades into the photo
 *
 * Fallback: the <img> in the container stays visible if anything fails.
 *
 * Usage:
 *   const face = new MagdaFace3D(containerEl, { mood: 0, data: 'assets/magda3d.bin' });
 *   face.setMood(1);  // big smile      face.setMood(-1);  // deep frown
 */
(function (global) {
  'use strict';

  var VERT = [
    'attribute vec3 aPos;',  // normalized int16
    'attribute vec3 aS;',    // smile delta
    'attribute vec3 aF;',    // frown delta
    'attribute vec3 aK;',    // blink delta
    'attribute vec3 aN;',    // base normal
    'attribute vec3 aNS;',   // smile normal
    'attribute vec3 aNF;',   // frown normal
    'attribute vec2 aX;',    // material id, fade/shade (0..255)
    'uniform vec3 uLo; uniform vec3 uHi; uniform vec3 uSc;',
    'uniform vec3 uW;',      // smile, frown, blink weights
    'uniform vec2 uImg;',
    'varying vec2 vUv; varying vec3 vN; varying vec3 vN0; varying float vMat; varying float vFade;',
    'varying float vBw; varying float vSide;',
    'void main(){',
    '  vec3 base = uLo + (aPos * 32767.0 + 32768.0) / 65535.0 * (uHi - uLo);',
    '  vec3 p = base + aS * uSc.x * uW.x + aF * uSc.y * uW.y + aK * uSc.z * uW.z;',
    '  vN0 = aN;',
    '  vN = aN + (aNS - aN) * uW.x + (aNF - aN) * uW.y;',
    '  vUv = vec2(base.x / uImg.x, 1.0 - base.y / uImg.y);',
    '  vBw = clamp(length(aK), 0.0, 1.0);',      // how much this vertex moves in a blink (0..1)
    '  vSide = base.x < uImg.x * 0.457 ? 0.0 : 1.0;',
    '  vMat = aX.x; vFade = aX.y / 255.0;',
    '  gl_Position = vec4(p.x / uImg.x * 2.0 - 1.0, p.y / uImg.y * 2.0 - 1.0, -p.z / 2000.0, 1.0);',
    '}'
  ].join('\n');

  var FRAG = [
    'precision mediump float;',
    'uniform sampler2D uTex;',
    'uniform vec4 uLid;',   // lid-skin sample points (uv) left xy, right zw
    'uniform float uBlink;',
    'varying vec2 vUv; varying vec3 vN; varying vec3 vN0; varying float vMat; varying float vFade;',
    'varying float vBw; varying float vSide;',
    'vec3 lidSkin(vec2 uv){',
    '  vec2 o = vec2(0.004, 0.003);',
    '  return (texture2D(uTex, uv + o).rgb + texture2D(uTex, uv - o).rgb + texture2D(uTex, uv + vec2(o.x, -o.y)).rgb + texture2D(uTex, uv + vec2(-o.x, o.y)).rgb) * 0.25;',
    '}',
    'const vec3 LASH = vec3(0.28, 0.19, 0.16);',
    'const vec3 L = vec3(-0.25, 0.45, 0.86);',  // soft key light, front / above
    'float shade(vec3 n){ return 0.45 + 0.55 * max(dot(normalize(n), L), 0.0); }',
    'void main(){',
    '  float d = shade(vN);',
    '  if (vMat < 1.5) {',                     // skin + eyeballs: photo, relit
    '    vec4 c = texture2D(uTex, vUv);',
    // Relighting adds crease shading, but around the mouth it puffs the upper
    // lip into a pillow, so it is masked out there and kept gentle elsewhere.
    '    vec2 mp = (vUv - vec2(0.4567, 0.3523)) / vec2(0.095, 0.042);',
    '    float rl = 0.35 * smoothstep(0.8, 1.6, length(mp));',
    '    float r = mix(1.0, clamp(d / shade(vN0), 0.6, 1.3), rl);',
    // Warm shadows: skin scatters red light, so red darkens least.
    '    c.rgb *= r < 1.0 ? pow(vec3(r), vec3(0.75, 1.1, 1.25)) : vec3(r);',
    // Blink: moving lid becomes procedural skin with a soft lash line at its edge.
    '    vec3 skin = lidSkin(vSide < 0.5 ? uLid.xy : uLid.zw) * vec3(0.97, 0.9, 0.88) * (0.85 + 0.2 * d);',
    '    if (uBlink > 0.01 && vBw > 0.01) {',
    '      c.rgb = mix(c.rgb, skin, smoothstep(0.02, 0.18, vBw * uBlink));',
    '      c.rgb = mix(c.rgb, LASH, smoothstep(0.7, 0.95, vBw) * smoothstep(0.1, 0.4, uBlink) * 0.75);',
    '    }',
    // Any eyeball still peeking through a nearly closed lid reads as lid skin.
    '    if (vMat > 0.5) c.rgb = mix(c.rgb, skin, smoothstep(0.75, 0.98, uBlink));',
    '    float a = vMat < 0.5 ? vFade : 1.0;',
    '    gl_FragColor = c * a;',
    '  } else {',
    '    vec3 col = vMat < 2.5 ? vec3(0.16, 0.04, 0.05)',   // mouth socket
    '             : vMat < 3.5 ? vec3(0.62, 0.27, 0.29)',   // gums / tongue
    '             : vec3(1.0, 0.96, 0.88);',                // teeth
    '    float lit = vMat > 3.5 ? 0.55 + 0.45 * d : d;',
    '    gl_FragColor = vec4(col * lit * vFade, 1.0);',      // vFade = cavity depth shade
    '  }',
    '}'
  ].join('\n');

  var BG_VERT = 'attribute vec2 aP; varying vec2 vUv; void main(){ vUv = vec2(aP.x*0.5+0.5, 0.5-aP.y*0.5); gl_Position = vec4(aP, 0.999, 1.0); }';
  var BG_FRAG = 'precision mediump float; uniform sampler2D uTex; varying vec2 vUv; void main(){ gl_FragColor = texture2D(uTex, vUv); }';

  function clamp1(v) { return Math.max(-1, Math.min(1, v)); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function ease(x) { return x * x * (3 - 2 * x); }

  function MagdaFace3D(container, opts) {
    opts = opts || {};
    this.container = container;
    this.img = container.querySelector('img');
    this.reduced = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.target = clamp1(opts.mood || 0); this.mood = this.target; this.vel = 0;
    this.blink = 0; this.blinkT = 0; this.nextBlink = 1.5;
    this.visible = true; this.raf = 0; this.last = 0;
    this._frame = this._frame.bind(this);
    var self = this;
    var imgReady = new Promise(function (res, rej) {
      if (self.img.complete && self.img.naturalWidth) res();
      else { self.img.addEventListener('load', res, { once: true }); self.img.addEventListener('error', rej, { once: true }); }
    });
    var data = opts.buffer ? Promise.resolve(opts.buffer)
      : fetch(opts.data || 'assets/magda3d.bin').then(function (r) { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); });
    Promise.all([data, imgReady]).then(function (r) { self._init(r[0]); })
      .catch(function (e) { if (global.console) console.warn('MagdaFace3D: static fallback', e); });
  }

  MagdaFace3D.prototype.setMood = function (mood) {
    this.target = clamp1(mood);
    if (this.reduced) { this.mood = this.target; this.vel = 0; }
    this._kick();
  };

  MagdaFace3D.prototype._init = function (buf) {
    var hlen = new DataView(buf).getUint32(0, true);
    var hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
    var base = 4 + hlen, parts = {};
    hdr.layout.forEach(function (l) { parts[l.name] = { off: base + l.offset, bytes: l.bytes }; });

    var canvas = document.createElement('canvas');
    var gl = canvas.getContext('webgl', { premultipliedAlpha: true, antialias: true, alpha: true });
    if (!gl) return;

    function prog(vs, fs) {
      function sh(t, s) {
        var o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o);
        if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(o));
        return o;
      }
      var p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      return p;
    }
    var face, bg;
    try { face = prog(VERT, FRAG); bg = prog(BG_VERT, BG_FRAG); }
    catch (e) { if (global.console) console.warn('MagdaFace3D: static fallback', e); return; }

    var tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
     [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]].forEach(function (p) { gl.texParameteri(gl.TEXTURE_2D, p[0], p[1]); });
    try { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.img); }
    catch (e) { if (global.console) console.warn('MagdaFace3D: texture upload failed', e); return; }

    // One VBO holding the whole file; attributes point into it.
    var vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(buf), gl.STATIC_DRAW);
    var ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(buf, parts.index.off, parts.index.bytes / 2), gl.STATIC_DRAW);

    var attrs = [
      ['aPos', 'pos', 3, gl.SHORT, true, 6], ['aS', 'dSmile', 3, gl.SHORT, true, 6],
      ['aF', 'dFrown', 3, gl.SHORT, true, 6], ['aK', 'dBlink', 3, gl.SHORT, true, 6],
      ['aN', 'n', 3, gl.BYTE, true, 4], ['aNS', 'nSmile', 3, gl.BYTE, true, 4],
      ['aNF', 'nFrown', 3, gl.BYTE, true, 4], ['aX', 'extra', 2, gl.UNSIGNED_BYTE, false, 2]
    ].map(function (a) { return { loc: gl.getAttribLocation(face, a[0]), off: parts[a[1]].off, size: a[2], type: a[3], norm: a[4], stride: a[5] }; });

    var quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    gl.useProgram(face);
    var u = {};
    ['uLo', 'uHi', 'uSc', 'uW', 'uImg', 'uTex', 'uLid', 'uBlink'].forEach(function (n) { u[n] = gl.getUniformLocation(face, n); });
    gl.uniform3fv(u.uLo, hdr.lo); gl.uniform3fv(u.uHi, hdr.hi);
    gl.uniform3f(u.uSc, hdr.scale.smile, hdr.scale.frown, hdr.scale.blink);
    gl.uniform2f(u.uImg, hdr.img[0], hdr.img[1]);
    // Clean skin just below each eye (photo px → uv); lids are tinted slightly darker.
    gl.uniform4f(u.uLid, 560 / hdr.img[0], 446 / hdr.img[1], 772 / hdr.img[0], 440 / hdr.img[1]);

    this.gl = gl; this.canvas = canvas; this.u = u; this.attrs = attrs;
    this.progFace = face; this.progBg = bg; this.vbo = vbo; this.ibo = ibo; this.quad = quad;
    this.bgLoc = gl.getAttribLocation(bg, 'aP');
    this.nIdx = hdr.tris * 3;

    canvas.setAttribute('aria-hidden', 'true');
    canvas.className = 'magda-canvas';
    this.container.appendChild(canvas);
    this.container.classList.add('magda-live');

    var self = this;
    canvas.addEventListener('webglcontextlost', function (e) {
      e.preventDefault(); cancelAnimationFrame(self.raf); self.raf = 0; self.gl = null;
      self.container.classList.remove('magda-live');
    });
    this._resize();
    if ('ResizeObserver' in global) new ResizeObserver(function () { self._resize(); self._kick(); }).observe(this.container);
    else global.addEventListener('resize', function () { self._resize(); self._kick(); });
    if ('IntersectionObserver' in global) {
      new IntersectionObserver(function (es) { self.visible = es[0].isIntersecting; self._kick(); }).observe(this.container);
    }
    document.addEventListener('visibilitychange', function () { self._kick(); });
    this._kick();
  };

  MagdaFace3D.prototype._resize = function () {
    if (!this.gl) return;
    var dpr = Math.min(global.devicePixelRatio || 1, 2), r = this.container.getBoundingClientRect();
    var w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h; this.gl.viewport(0, 0, w, h);
    }
  };

  MagdaFace3D.prototype._kick = function () {
    if (!this.gl || this.raf || !this.visible || document.hidden) return;
    this.last = 0; this.raf = requestAnimationFrame(this._frame);
  };

  MagdaFace3D.prototype._frame = function (now) {
    this.raf = 0;
    if (!this.gl) return;
    var dt = this.last ? Math.min(0.05, (now - this.last) / 1000) : 1 / 60;
    this.last = now;
    var animating = this._update(dt);
    this._draw();
    if (animating && this.visible && !document.hidden) this.raf = requestAnimationFrame(this._frame);
  };

  MagdaFace3D.prototype._update = function (dt) {
    var k = 30, c = 2 * Math.sqrt(k);
    this.vel += (k * (this.target - this.mood) - c * this.vel) * dt;
    this.mood += this.vel * dt;
    var moving = Math.abs(this.vel) > 1e-3 || Math.abs(this.target - this.mood) > 1e-3;
    if (this.reduced) return moving;
    this.nextBlink -= dt;
    if (this.nextBlink <= 0 && this.blinkT <= 0) {
      this.blinkT = 0.0001;
      this.nextBlink = Math.random() < 0.15 ? 0.3 : lerp(2, 6, Math.random());
    }
    if (this.blinkT > 0) {
      this.blinkT += dt / 0.15;
      this.blink = Math.sin(Math.min(1, this.blinkT) * Math.PI);
      if (this.blinkT >= 1) { this.blinkT = 0; this.blink = 0; }
    }
    return true;
  };

  MagdaFace3D.prototype._draw = function () {
    var gl = this.gl;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // 1. Photo background
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    gl.useProgram(this.progBg);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.enableVertexAttribArray(this.bgLoc);
    gl.vertexAttribPointer(this.bgLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.disableVertexAttribArray(this.bgLoc);

    // 2. Face mesh
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.progFace);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    this.attrs.forEach(function (a) {
      if (a.loc < 0) return;
      gl.enableVertexAttribArray(a.loc);
      gl.vertexAttribPointer(a.loc, a.size, a.type, a.norm, a.stride, a.off);
    });
    var sp = ease(Math.max(0, this.mood)), sn = ease(Math.max(0, -this.mood));
    gl.uniform3f(this.u.uW, sp, sn, this.blink);
    gl.uniform1f(this.u.uBlink, this.blink);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.drawElements(gl.TRIANGLES, this.nIdx, gl.UNSIGNED_SHORT, 0);
    this.attrs.forEach(function (a) { if (a.loc >= 0) gl.disableVertexAttribArray(a.loc); });
  };

  global.MagdaFace3D = MagdaFace3D;
})(window);
