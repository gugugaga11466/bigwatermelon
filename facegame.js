/* ============================================================
 * facegame.js — 合成大西瓜 · 人脸换装版
 * 功能：
 *   1. 开局上传一张单人照
 *   2. 检测人脸，圆形裁剪后贴到每个水果上
 *   3. 不同水果叠加不同卡通兽耳（猫耳/狗耳/兔耳/熊耳/狐耳）
 *   4. 合成出「大西瓜」(fruitNumber 10) 时判定胜利：
 *      显示猫耳女仆照并播放一声猫叫
 * ============================================================ */
(function () {
  'use strict';

  /* ---------- 配置 ---------- */
  var MODEL_PATH = 'models/';
  var ASSET_PATH = 'facegame/assets/';
  var OVERLAY_W = 512;   // 贴图宽（人脸圆直径）
  var OVERLAY_H = 1024;  // 贴图高（人脸 + 上方兽耳空间）
  var FACE_R = 256;      // 人脸圆半径
  var FACE_CX = 256;     // 人脸圆中心 x
  var FACE_CY = 512;     // 人脸圆中心 y
  var EAR_SHIFT = 0.30;  // 兽耳相对人脸圆心向上偏移（相对 FACE_R）

  // fruitNumber 0..10 对应的兽耳贴图
  var EAR_MAP = [
    'ear_cat_grey',    // 0 葡萄 -> 猫耳(灰)
    'ear_fox_orange',  // 1 樱桃 -> 狐耳
    'ear_dog_brown',   // 2 橘子 -> 狗耳
    'ear_rabbit_cream',// 3 柠檬 -> 兔耳
    'ear_bear_brown',  // 4 猕猴桃 -> 熊耳
    'ear_cat_white',   // 5 西红柿 -> 猫耳(白)
    'ear_rabbit_pink', // 6 桃子 -> 兔耳(粉)
    'ear_fox_gold',    // 7 菠萝 -> 狐耳(金)
    'ear_dog_dark',    // 8 椰子 -> 狗耳(深)
    'ear_cat_green',   // 9 西瓜 -> 猫耳(绿)
    'ear_cat_green'    // 10 大西瓜 -> 猫耳(绿,女仆主题)
  ];

  /* ---------- 全局状态 ---------- */
  var faceapi = null;
  var modelsPromise = null;
  var faceCircleCanvas = null;   // 圆形人脸（贴到水果）
  var userFaceCanvas = null;     // 正方形脸部（换脸到女仆底图）
  var overlayFrames = [];        // 每个水果的 cc.SpriteFrame（人脸+兽耳）
  var maidImageUrl = null;       // 猫耳女仆照 dataURL
  var gameStarted = false;
  var winHandled = false;
  var pollingTimer = null;

  /* ---------- 工具 ---------- */
  function $(id) { return document.getElementById(id); }

  function loadImage(url) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () { resolve(img); };
      img.onerror = function () { reject(new Error('load failed: ' + url)); };
      img.src = url;
    });
  }

  function imageToCanvas(img, maxSize) {
    var w = img.naturalWidth || img.width;
    var h = img.naturalHeight || img.height;
    var scale = 1;
    if (maxSize && Math.max(w, h) > maxSize) scale = maxSize / Math.max(w, h);
    var c = document.createElement('canvas');
    c.width = Math.round(w * scale);
    c.height = Math.round(h * scale);
    var ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, c.width, c.height);
    return c;
  }

  /* ---------- face-api 模型加载 ---------- */
  function ensureModels() {
    if (modelsPromise) return modelsPromise;
    modelsPromise = (function () {
      return new Promise(function (resolve, reject) {
        function wait() {
          if (window.faceapi) {
            faceapi = window.faceapi;
            resolve();
          } else if (tries > 200) {
            reject(new Error('face-api.js 未加载'));
          } else {
            tries++;
            setTimeout(wait, 100);
          }
        }
        var tries = 0;
        wait();
      }).then(function () {
        var p;
        try { p = faceapi.tf.setBackend('webgl'); } catch (e) { p = Promise.reject(e); }
        return Promise.resolve(p).then(function () {
          return faceapi.tf.ready();
        }).catch(function () {
          return faceapi.tf.setBackend('cpu').then(function () {
            return faceapi.tf.ready();
          }).catch(function () { return null; });
        });
      }).then(function () {
        return Promise.all([
          faceapi.nets.tinyFaceDetector.load(MODEL_PATH),
          faceapi.nets.faceLandmark68Net.load(MODEL_PATH)
        ]);
      });
    })();
    return modelsPromise;
  }

  /* ---------- 人脸检测 ---------- */
  function detectLargestFace(canvas) {
    var options = new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.3 });
    return faceapi.detectAllFaces(canvas, options).withFaceLandmarks().then(function (results) {
      var count = results ? results.length : 0;
      if (count === 0) return { face: null, count: 0 };
      var best = null, bestArea = -1;
      for (var i = 0; i < results.length; i++) {
        var b = results[i].detection.box;
        var area = b.width * b.height;
        if (area > bestArea) { bestArea = area; best = results[i]; }
      }
      return { face: best, count: count };
    });
  }

  /* ---------- 裁剪 ---------- */
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function cropCircularFace(srcCanvas, box, size) {
    var side = Math.max(box.width, box.height) * 1.55;
    side = Math.min(side, srcCanvas.width, srcCanvas.height);
    var cx = box.x + box.width / 2;
    var cy = box.y + box.height / 2;
    var x = clamp(cx - side / 2, 0, srcCanvas.width - side);
    var y = clamp(cy - side / 2, 0, srcCanvas.height - side);
    var out = document.createElement('canvas');
    out.width = size; out.height = size;
    var ctx = out.getContext('2d');
    ctx.save();
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(srcCanvas, x, y, side, side, 0, 0, size, size);
    ctx.restore();
    // 描一圈浅色边，让脸在水果上更清晰
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size / 2 - 3, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 8;
    ctx.stroke();
    return out;
  }

  // 正方形脸部裁剪（不裁剪成圆形，用于换脸到底图上）
  function cropFaceSquare(srcCanvas, box, size) {
    var side = Math.max(box.width, box.height) * 1.18;
    side = Math.min(side, srcCanvas.width, srcCanvas.height);
    var cx = box.x + box.width / 2;
    var cy = box.y + box.height / 2;
    var x = clamp(cx - side / 2, 0, srcCanvas.width - side);
    var y = clamp(cy - side / 2, 0, srcCanvas.height - side);
    var out = document.createElement('canvas');
    out.width = size; out.height = size;
    var ctx = out.getContext('2d');
    ctx.drawImage(srcCanvas, x, y, side, side, 0, 0, size, size);
    return out;
  }

  function cropHead(srcCanvas, box, landmarks, size) {
    var pts = landmarks.positions;
    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (var i = 0; i < pts.length; i++) {
      if (pts[i].x < minX) minX = pts[i].x;
      if (pts[i].x > maxX) maxX = pts[i].x;
      if (pts[i].y < minY) minY = pts[i].y;
      if (pts[i].y > maxY) maxY = pts[i].y;
    }
    var faceW = maxX - minX;
    var faceH = maxY - minY;
    // 头部区域：脸框上方留出额头+头发，两侧留头发
    var headTop = minY - faceH * 0.85;
    var headBottom = maxY + faceH * 0.30;
    var headW = faceW * 2.2;
    var centerX = (minX + maxX) / 2;
    var side = Math.max(headBottom - headTop, headW);
    side = Math.min(side, srcCanvas.width, srcCanvas.height);
    var x = clamp(centerX - side / 2, 0, srcCanvas.width - side);
    var y = clamp(headTop, 0, srcCanvas.height - side);
    var out = document.createElement('canvas');
    out.width = size; out.height = size;
    var ctx = out.getContext('2d');
    ctx.drawImage(srcCanvas, x, y, side, side, 0, 0, size, size);
    return out;
  }

  /* ---------- 合成 ---------- */
  // 用「兽耳/头饰」贴图叠加：asset 约定为 512x512，头部圆心(256,256) 半径150
  function drawAccessory(ctx, assetImg, cx, cy, radius, shiftUp) {
    var scale = radius / 150;
    ctx.save();
    ctx.translate(cx, cy - (shiftUp || 0) * radius);
    ctx.scale(scale, scale);
    ctx.drawImage(assetImg, -256, -256);
    ctx.restore();
  }

  function makeFruitOverlay(faceCanvas, earImg) {
    var c = document.createElement('canvas');
    c.width = OVERLAY_W; c.height = OVERLAY_H;
    var ctx = c.getContext('2d');
    // 人脸圆
    ctx.save();
    ctx.beginPath();
    ctx.arc(FACE_CX, FACE_CY, FACE_R, 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(faceCanvas, FACE_CX - FACE_R, FACE_CY - FACE_R, FACE_R * 2, FACE_R * 2);
    ctx.restore();
    // 兽耳
    if (earImg) drawAccessory(ctx, earImg, FACE_CX, FACE_CY, FACE_R, EAR_SHIFT);
    return c;
  }

  // 画柔和的腮红（红晕）
  function drawBlush(ctx, cx, cy, rx, ry) {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(1, ry / rx);
    var g = ctx.createRadialGradient(0, 0, 0, 0, 0, rx);
    g.addColorStop(0, 'rgba(255, 108, 135, 0.42)');
    g.addColorStop(0.55, 'rgba(255, 108, 135, 0.22)');
    g.addColorStop(1, 'rgba(255, 108, 135, 0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(0, 0, rx, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // 猫耳女仆照：用户头像 + 红晕 + 白猫耳 + 蕾丝发带
  function makeMaidPhoto(headCanvas, catEarImg, headdressImg) {
    var size = headCanvas.width;
    var c = document.createElement('canvas');
    c.width = size; c.height = size;
    var ctx = c.getContext('2d');
    ctx.drawImage(headCanvas, 0, 0);
    // 头部圆心估计在画布中间偏上一点，半径约半宽
    var hx = size / 2;
    var hy = size * 0.46;
    var hr = size * 0.44;
    // 红晕（脸颊两侧）
    drawBlush(ctx, hx - hr * 0.48, hy + hr * 0.18, hr * 0.30, hr * 0.19);
    drawBlush(ctx, hx + hr * 0.48, hy + hr * 0.18, hr * 0.30, hr * 0.19);
    // 猫耳（画在发带后面）+ 女仆蕾丝发带
    drawAccessory(ctx, catEarImg, hx, hy, hr, 0.20);
    drawAccessory(ctx, headdressImg, hx, hy, hr, 0.0);
    return c;
  }

  /* ---------- 猫叫合成 ---------- */
  var audioCtx = null;

  // 在用户手势中尽早解锁音频（避免浏览器自动播放策略拦住猫叫）
  function ensureAudioCtx() {
    try {
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      if (!audioCtx) audioCtx = new AC();
      if (audioCtx.state === 'suspended' && audioCtx.resume) audioCtx.resume();
      return audioCtx;
    } catch (e) { return null; }
  }

  function playMeow() {
    try {
      var ctx = ensureAudioCtx();
      if (!ctx) return;
      var now = ctx.currentTime;
      var dur = 0.75;

      var master = ctx.createGain();
      master.gain.setValueAtTime(0.0001, now);
      master.gain.linearRampToValueAtTime(0.45, now + 0.10);
      master.gain.setValueAtTime(0.45, now + 0.25);
      master.gain.linearRampToValueAtTime(0.0001, now + dur);
      master.connect(ctx.destination);

      // 颤音 LFO
      var lfo = ctx.createOscillator();
      lfo.frequency.value = 21;
      var lfoGain = ctx.createGain();
      lfoGain.gain.value = 16;
      lfo.connect(lfoGain);

      [1, 2].forEach(function (harm, idx) {
        var osc = ctx.createOscillator();
        osc.type = idx === 0 ? 'sawtooth' : 'triangle';
        var f0 = 380, fpeak = 830;
        osc.frequency.setValueAtTime(f0 * harm, now);
        osc.frequency.linearRampToValueAtTime(fpeak * harm, now + 0.28);
        osc.frequency.linearRampToValueAtTime(f0 * 0.8 * harm, now + dur);
        lfoGain.connect(osc.frequency);
        var bp = ctx.createBiquadFilter();
        bp.type = 'bandpass';
        bp.frequency.value = 1250 * harm;
        bp.Q.value = 1.4;
        var g = ctx.createGain();
        g.gain.value = idx === 0 ? 1 : 0.35;
        osc.connect(bp); bp.connect(g); g.connect(master);
        osc.start(now); osc.stop(now + dur + 0.05);
      });
      lfo.start(now); lfo.stop(now + dur + 0.05);
    } catch (e) { /* ignore */ }
  }

  // 优先播放真实猫叫录音，失败则回退到合成猫叫
  var meowAudio = null;
  function playRealMeow() {
    try {
      if (!meowAudio) {
        meowAudio = new Audio('meow.mp3');
        meowAudio.volume = 0.9;
      }
      meowAudio.currentTime = 0;
      var p = meowAudio.play();
      if (p && p.catch) p.catch(function () { playMeow(); });
    } catch (e) {
      playMeow();
    }
  }

  /* ---------- DOM UI ---------- */
  function buildUploadUI() {
    var div = document.createElement('div');
    div.id = 'face-upload';
    div.innerHTML =
      '<div class="fu-card">' +
      '  <h1>🍉 合成大西瓜 · 人脸版</h1>' +
      '  <p class="fu-sub">请上传一张 <b>只有一个人像</b> 的照片</p>' +
      '  <label class="fu-btn">选择照片<input type="file" id="fu-file" accept="image/*" style="display:none"></label>' +
      '  <div class="fu-preview"><img id="fu-preview-img" alt=""></div>' +
      '  <div class="fu-status" id="fu-status"></div>' +
      '  <button class="fu-btn fu-start" id="fu-start" disabled>开始游戏</button>' +
      '</div>';
    document.body.appendChild(div);

    var style = document.createElement('style');
    style.textContent =
      '#face-upload{position:fixed;inset:0;z-index:99999;background:linear-gradient(160deg,#ffe9f0,#fff6e6,#e7f6ff);' +
      'display:flex;align-items:center;justify-content:center;font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;}' +
      '.fu-card{background:#fff;border-radius:24px;box-shadow:0 20px 60px rgba(0,0,0,.15);padding:36px 40px;text-align:center;max-width:420px;width:90%;max-height:92vh;overflow-y:auto;}' +
      '.fu-card h1{font-size:24px;margin:0 0 8px;color:#e2578a;}' +
      '.fu-sub{color:#888;margin:0 0 20px;}' +
      '.fu-btn{display:inline-block;background:#ff8fb3;color:#fff;border:0;border-radius:999px;padding:12px 26px;font-size:16px;cursor:pointer;margin:6px;}' +
      '.fu-btn:disabled{background:#ccc;cursor:not-allowed;}' +
      '.fu-preview{margin:16px auto;width:150px;height:150px;border-radius:50%;overflow:hidden;border:4px solid #ffd3e2;background:#f5f5f5;}' +
      '.fu-preview img{width:100%;height:100%;object-fit:cover;display:none;}' +
      '.fu-status{min-height:22px;color:#e2578a;font-size:14px;margin:8px 0;}' +
      '.fu-start{background:#5bd1a2;font-weight:bold;padding:14px 40px;}';
    document.head.appendChild(style);

    var fileInput = $('fu-file');
    fileInput.addEventListener('change', function () {
      ensureAudioCtx();
      var f = fileInput.files[0];
      if (!f) return;
      var reader = new FileReader();
      reader.onload = function (e) {
        var img = new Image();
        img.onload = function () {
          processPhoto(img);
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(f);
    });

    $('fu-start').addEventListener('click', function () {
      ensureAudioCtx();
      startGame();
    });
  }

  function setStatus(txt, ok) {
    var el = $('fu-status');
    if (el) { el.textContent = txt; el.style.color = ok ? '#2bb673' : '#e2578a'; }
  }

  function processPhoto(img) {
    setStatus('正在识别人脸…');
    ensureModels().then(function () {
      var src = imageToCanvas(img, 900);
      return detectLargestFace(src).then(function (res) {
        var face = res.face;
        if (!face) { setStatus('未检测到人脸，请换一张正脸清晰的照片', false); return; }
        if (res.count > 1) {
          setStatus('检测到 ' + res.count + ' 个人像，已取最大的一张（建议单人照）', false);
        }
        var box = face.detection.box;
        // 圆形人脸（贴到水果上）
        faceCircleCanvas = cropCircularFace(src, box, OVERLAY_W);
        // 猫耳女仆照：用户头像 + 红晕 + 白猫耳 + 蕾丝发带
        var headCanvas = cropHead(src, box, face.landmarks, 512);
        Promise.all([
          loadImage(ASSET_PATH + 'ear_cat_white.png'),
          loadImage(ASSET_PATH + 'maid_headdress.png')
        ]).then(function (imgs) {
          var maid = makeMaidPhoto(headCanvas, imgs[0], imgs[1]);
          maidImageUrl = maid.toDataURL('image/png');
          // 显示预览
          var prev = $('fu-preview-img');
          if (prev) { prev.src = faceCircleCanvas.toDataURL(); prev.style.display = 'block'; }
          if (res.count > 1) {
            setStatus('识别成功（已取最大人脸）✓ 点击“开始游戏”', true);
          } else {
            setStatus('识别成功 ✓ 点击“开始游戏”', true);
          }
          $('fu-start').disabled = false;
        }).catch(function () { setStatus('兽耳资源加载失败', false); });
      });
    }).catch(function (err) {
      console.error(err);
      setStatus('人脸模型加载失败：' + (err && err.message ? err.message : err), false);
    });
  }

  /* ---------- Cocos 集成 ---------- */
  function waitForCC(timeout) {
    return new Promise(function (resolve) {
      var t = 0;
      function poll() {
        if (window.cc && cc.Texture2D && cc.SpriteFrame) return resolve();
        if (t > (timeout || 20000)) return resolve();
        t += 100;
        setTimeout(poll, 100);
      }
      poll();
    });
  }

  // 把留白区域（黑边）改成柔和主题色，避免黑屏观感
  function setNiceBackground() {
    try {
      var cam = cc.Camera;
      ['Bg Camera'].forEach(function (name) {
        var n = cc.find('Canvas/' + name);
        var c = n && n.getComponent ? n.getComponent(cam) : null;
        if (c) c.backgroundColor = cc.color(255, 228, 236, 255);
      });
      document.body.style.background = '#ffe4ec';
    } catch (e) { /* ignore */ }
  }

  // 游戏在 9:16 iframe 内渲染，宽高比与设计分辨率一致，无需再改分辨率策略；
  // 这里只把可能的留白改成柔和主题色，避免黑屏观感。
  var showAllEnforced = false;
  function enforceShowAll() {
    if (showAllEnforced) return;
    try {
      if (!window.cc || !cc.find || !cc.find('Canvas')) return; // 场景未加载完成
      setNiceBackground();
      showAllEnforced = true;
    } catch (e) { /* ignore */ }
  }

  function canvasToSpriteFrame(canvas) {
    var tex = new cc.Texture2D();
    tex.initWithElement(canvas);
    var sf = new cc.SpriteFrame();
    sf.setTexture(tex);
    return sf;
  }

  function buildOverlayFrames() {
    overlayFrames = [];
    var earPromises = EAR_MAP.map(function (name) {
      return loadImage(ASSET_PATH + name + '.png');
    });
    return Promise.all(earPromises).then(function (earImgs) {
      for (var i = 0; i < earImgs.length; i++) {
        var c = makeFruitOverlay(faceCircleCanvas, earImgs[i]);
        overlayFrames[i] = canvasToSpriteFrame(c);
      }
    });
  }

  function findCanvas() {
    try { return window.cc && cc.find ? cc.find('Canvas') : null; } catch (e) { return null; }
  }

  function addOverlayToFruit(fruit, fruitNumber) {
    if (fruit.__faceOverlayAdded) return;
    fruit.__faceOverlayAdded = true;
    var frame = overlayFrames[fruitNumber];
    if (!frame) return;
    var child = new cc.Node();
    var spr = child.addComponent(cc.Sprite);
    spr.spriteFrame = frame;
    child.parent = fruit;
    child.setPosition(cc.v2(0, 0));
    child.setAnchorPoint(cc.v2(0.5, 0.5));
    var D = fruit.width || fruit.height || 100;
    child.setScale(0.88 * D / OVERLAY_W);
  }

  function scanFruits() {
    var canvas = findCanvas();
    if (!canvas) return false;
    var names = ['fruitNode', 'lineNode'];
    for (var n = 0; n < names.length; n++) {
      var node = canvas.getChildByName ? canvas.getChildByName(names[n]) : cc.find('Canvas/' + names[n]);
      if (!node) continue;
      var kids = node.children;
      for (var i = 0; i < kids.length; i++) {
        var fruit = kids[i];
        var fd = null;
        try { fd = fruit.getComponent('fruitData'); } catch (e) {}
        if (!fd) continue;
        var num = fd.fruitNumber;
        if (typeof num !== 'number') continue;
        addOverlayToFruit(fruit, num);
      }
    }
    return true;
  }

  function checkWin() {
    var canvas = findCanvas();
    if (!canvas) return false;
    var node = canvas.getChildByName ? canvas.getChildByName('fruitNode') : cc.find('Canvas/fruitNode');
    if (!node) return false;
    for (var i = 0; i < node.children.length; i++) {
      var fd = null;
      try { fd = node.children[i].getComponent('fruitData'); } catch (e) {}
      if (fd && fd.fruitNumber >= 10) return true;
    }
    return false;
  }

  /* ---------- 胜利 ---------- */
  function showWin() {
    if (winHandled) return;
    winHandled = true;
    // 暂停游戏
    try { cc.game.pause(); } catch (e) {}
    var div = document.createElement('div');
    div.id = 'maid-overlay';
    div.innerHTML =
      '<div class="mo-card">' +
      '  <h1>🎉 合成大西瓜成功！</h1>' +
      '  <div class="mo-imgwrap"><img id="mo-img" alt="猫耳女仆"></div>' +
      '  <p class="mo-sub">猫耳女仆 登场～喵！</p>' +
      '  <button class="fu-btn fu-start" id="mo-restart">再来一局</button>' +
      '</div>';
    document.body.appendChild(div);
    var style = document.createElement('style');
    style.textContent =
      '#maid-overlay{position:fixed;inset:0;z-index:99998;background:rgba(30,20,40,.82);display:flex;align-items:center;justify-content:center;font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;}' +
      '.mo-card{background:#fff;border-radius:24px;padding:28px 36px;text-align:center;max-width:360px;width:90%;max-height:92vh;overflow-y:auto;}' +
      '.mo-card h1{font-size:22px;color:#e2578a;margin:0 0 14px;}' +
      '.mo-imgwrap{width:240px;height:240px;border-radius:20px;overflow:hidden;margin:0 auto;border:5px solid #ffd3e2;}' +
      '.mo-imgwrap img{width:100%;height:100%;object-fit:cover;}' +
      '.mo-sub{color:#666;margin:12px 0;}';
    document.head.appendChild(style);
    $('mo-img').src = maidImageUrl;
    $('mo-restart').addEventListener('click', function () { location.reload(); });
    playRealMeow();
    // 再叫一声加强效果
    setTimeout(playRealMeow, 700);
  }

  /* ---------- 启动 ---------- */
  function startGame() {
    if (gameStarted) return;
    gameStarted = true;
    setStatus('正在生成人脸水果…');
    waitForCC().then(buildOverlayFrames).then(function () {
      enforceShowAll();
      var up = $('face-upload');
      if (up) up.style.display = 'none';
      // 立即扫描一遍，随后轮询
      scanFruits();
      pollingTimer = setInterval(function () {
        enforceShowAll();
        scanFruits();
        if (!winHandled && checkWin()) showWin();
      }, 120);
    }).catch(function (err) {
      console.error(err);
      setStatus('生成贴图失败：' + err.message, false);
    });
  }

  /* ---------- 入口 ---------- */
  function main() {
    buildUploadUI();
    // 供调试：在控制台执行 __facegame.getState() 查看状态
    window.__facegame = {
      getState: function () {
        return {
          gameStarted: gameStarted,
          winHandled: winHandled,
          hasFace: !!faceCircleCanvas,
          framesBuilt: overlayFrames.length,
          hasCC: !!(window.cc && window.cc.find),
          hasFaceApi: !!window.faceapi
        };
      }
    };
    // 后台预加载人脸模型，避免选完照片再等待
    setStatus('正在加载人脸模型…');
    ensureModels().then(function () {
      setStatus('人脸模型就绪，可以上传照片了', true);
    }).catch(function (err) {
      console.error('模型预加载失败', err);
      setStatus('人脸模型加载失败，请刷新重试', false);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', main);
  } else {
    main();
  }
})();
