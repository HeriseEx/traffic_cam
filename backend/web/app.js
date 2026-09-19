'use strict';
const SALT = 'traffic-hello-v1';
const REMOTE = 'https://traffic.muqin.ccwu.cc';
const LOCAL = 'http://127.0.0.1:61616';
const MAX = 50 * 1024 * 1024;
const $ = id => document.getElementById(id);
const labels = {QUEUED:'排队中',PROCESSING:'分析中',ANALYZED:'分析完成',REJECTED:'未检出',ERROR:'失败',
  CANDIDATE:'疑似候选',RED:'红灯',GREEN:'绿灯',YELLOW:'黄灯',OFF:'未见灯',RED_LIGHT:'疑似闯红灯',
  INVALID_VIDEO:'无效视频',PLATE_UNCONFIRMED:'车牌未确认',car:'汽车',truck:'卡车',bus:'公交车',motorcycle:'摩托车'};
let apiRoot = storedApi() || defaultApi();
let session = '', media, recorder, header = null, parts = [], recording = false, frameTimer = 0;
let votes = [], signals = [], lastAutoMark = 0, approachFirst = -1, approachSamples = 0, approaching = false;

function isLanHost(host) {
  return /^(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)$/.test(host);
}
function defaultApi() {
  const host = location.hostname;
  if (host === 'cam.muqin.ccwu.cc') return REMOTE;
  if (location.port === '61612') return '';
  if (host === '127.0.0.1' || host === 'localhost') return LOCAL;
  if (isLanHost(host)) return `${location.protocol}//${host}:61616`;
  return REMOTE;
}
function storedApi() {
  let saved;
  try { saved = localStorage.getItem('traffic-api'); } catch { return null; }
  if (saved == null) return null;
  if (location.protocol === 'https:' && saved.startsWith('http:')) return null;
  return saved;
}
function uuid() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const bytes = new Uint8Array(16);
  (crypto.getRandomValues || (buf => { for (let i = 0; i < buf.length; i++) buf[i] = Math.random() * 256 | 0; return buf; }))(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = [...bytes].map(b => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}
function sha256sync(bytes) {
  const rr = (x, n) => ((x >>> n) | (x << (32 - n))) >>> 0;
  const K = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
  const H = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
  const bitLen = bytes.length * 8;
  const padLen = (bytes.length + 9 + 63) & ~63;
  const buf = new Uint8Array(padLen);
  buf.set(bytes);
  buf[bytes.length] = 0x80;
  buf[padLen - 4] = bitLen >>> 24; buf[padLen - 3] = bitLen >>> 16 & 255;
  buf[padLen - 2] = bitLen >>> 8 & 255; buf[padLen - 1] = bitLen & 255;
  const w = new Array(64);
  for (let i = 0; i < padLen; i += 64) {
    for (let t = 0; t < 16; t++) {
      const o = i + t * 4;
      w[t] = (buf[o] << 24 | buf[o + 1] << 16 | buf[o + 2] << 8 | buf[o + 3]) >>> 0;
    }
    for (let t = 16; t < 64; t++) {
      const x = w[t - 15], y = w[t - 2];
      w[t] = (w[t - 16] + (rr(x, 7) ^ rr(x, 18) ^ (x >>> 3)) + w[t - 7] + (rr(y, 17) ^ rr(y, 19) ^ (y >>> 10))) | 0;
    }
    let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (let t = 0; t < 64; t++) {
      const t1 = (h + (rr(e, 6) ^ rr(e, 11) ^ rr(e, 25)) + ((e & f) ^ (~e & g)) + K[t] + w[t]) | 0;
      const t2 = ((rr(a, 2) ^ rr(a, 13) ^ rr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    H[0] = (H[0] + a) | 0; H[1] = (H[1] + b) | 0; H[2] = (H[2] + c) | 0; H[3] = (H[3] + d) | 0;
    H[4] = (H[4] + e) | 0; H[5] = (H[5] + f) | 0; H[6] = (H[6] + g) | 0; H[7] = (H[7] + h) | 0;
  }
  return H.map(n => (n >>> 0).toString(16).padStart(8, '0')).join('');
}
function toast(text) { $('toast').textContent = text; $('toast').hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => $('toast').hidden = true, 4000); }
function status(text) { $('status').textContent = text; }
function title(v) { return labels[v] || v || '—'; }
async function sha256hex(data) {
  const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : new Uint8Array(data);
  if (crypto.subtle) {
    const buf = await crypto.subtle.digest('SHA-256', bytes);
    return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('');
  }
  return sha256sync(bytes);
}
let helloPromise=null;
async function hello() {
  if(helloPromise)return helloPromise;
  helloPromise=registerDevice();
  try{await helloPromise;}finally{helloPromise=null;}
}
async function registerDevice() {
  const ts = Math.floor(Date.now() / 1000), nonce = uuid().replace(/-/g, '');
  let device;
  try { device = localStorage.getItem('traffic-device-id'); } catch {}
  if (!device) device = uuid();
  const platform = 'web';
  const code = await sha256hex(`${device}\n${platform}\n${ts}\n${nonce}\n${SALT}`);
  const response = await fetch(apiRoot + '/v1/hello', { method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device_id: device, platform, model: (navigator.userAgent || 'web').slice(0, 120), app_version: 'web-2.0', ts, nonce, code }) });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '握手失败');
  session = data.session;
  try { localStorage.setItem('traffic-device-id', data.device_id || device); } catch {}
}
async function api(path, options = {}, retried = false) {
  if (!session) await hello();
  const headers = { Authorization: 'Bearer ' + session, ...(options.headers || {}) };
  const next = { ...options };
  if (next.json !== undefined) { next.body = JSON.stringify(next.json); headers['Content-Type'] = 'application/json'; delete next.json; }
  const response = await fetch(apiRoot + path, { ...next, headers });
  if (response.status === 401 && !retried) { session = ''; return api(path, options, true); }
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `请求失败 ${response.status}`);
  return data;
}
async function connect() {
  try {
    session = '';
    await api('/v1/settings');
    $('connection').textContent = '● 已连接';
    status('已自动登录，可开启相机或导入视频');
    await tick();
  } catch (error) {
    $('connection').textContent = '○ 未连接';
    status(error.message);
  }
}
async function tick() {
  try {
    const list = await api('/v1/tasks?limit=40');
    $('recordList').replaceChildren();
    for (const task of list.tasks || []) {
      const result = task.effective_result || task.result || {};
      const item = document.createElement('button');
      item.className = 'record';
      item.innerHTML = `<strong>${result.plate || '车牌未确认'}</strong><small>${title(task.status)} · ${title(result.violation_type || result.reason || result.decision)}</small>`;
      $('recordList').append(item);
    }
    if (!(list.tasks || []).length) $('recordList').textContent = '暂无记录';
  } catch (error) { toast(error.message); }
}
function mime() {
  if (typeof MediaRecorder === 'undefined') return '';
  return ['video/mp4;codecs=avc1.42E01E', 'video/mp4', 'video/webm;codecs=vp8', 'video/webm;codecs=vp9', 'video/webm']
    .find(t => MediaRecorder.isTypeSupported(t)) || '';
}
function clipBlob() {
  if (!header) return null;
  return new Blob([header, ...parts.map(item => item.b)], { type: (recorder && recorder.mimeType) || header.type || 'video/webm' });
}
function armRecorder() {
  header = null; parts = [];
  const type = mime();
  recorder = new MediaRecorder(media, type ? { mimeType: type, videoBitsPerSecond: 2_500_000 } : { videoBitsPerSecond: 2_500_000 });
  recorder.ondataavailable = event => {
    if (!event.data || !event.data.size) return;
    if (!header) header = event.data;
    else parts.push({ t: Date.now(), b: event.data });
  };
  recorder.start(1000);
}
function stopRecorder() {
  return new Promise(resolve => {
    if (!recorder || recorder.state === 'inactive') { resolve(clipBlob()); return; }
    recorder.onstop = () => resolve(clipBlob());
    try { recorder.requestData(); } catch (error) { /* older WebViews */ }
    recorder.stop();
  });
}
async function finalizeClip() {
  const blob = await stopRecorder();
  recorder = null; header = null; parts = [];
  if (media && media.getTracks().some(track => track.readyState === 'live')) armRecorder();
  return blob;
}
function confirmedPlate() {
  const tally = {};
  for (const round of votes) for (const text of new Set(round)) tally[text] = (tally[text] || 0) + 1;
  const best = Object.entries(tally).sort((a, b) => b[1] - a[1])[0];
  return best && best[1] >= 2 ? best[0] : '';
}
function redStable() {
  return signals.length >= 2 && signals[signals.length - 1] === 'RED' && signals.slice(-2).every(v => v === 'RED');
}
function updateApproach(boxes) {
  if (!redStable()) { approachFirst = -1; approachSamples = 0; approaching = false; return; }
  const lead = boxes.slice().sort((a, b) => {
    const aa = a.box_normalized, bb = b.box_normalized;
    return (bb[2] - bb[0]) * (bb[3] - bb[1]) - (aa[2] - aa[0]) * (aa[3] - aa[1]);
  })[0];
  if (!lead || !lead.box_normalized) return;
  const bottom = lead.box_normalized[3];
  if (approachFirst < 0) approachFirst = bottom;
  approachSamples++;
  if (approachSamples >= 3 && Math.abs(bottom - approachFirst) > 0.04) approaching = true;
}
function liveHint() {
  if (recording) return '正在截取重点片段';
  if (!(redStable() && approaching)) return '';
  return confirmedPlate() ? '疑似闯红灯（本地已识别）' : '疑似闯红灯（车牌未确认）';
}
function maybeAutoMark() {
  if (recording || !media || !recorder) return;
  if (!(redStable() && approaching && confirmedPlate())) return;
  if (Date.now() - lastAutoMark < 60_000) return;
  lastAutoMark = Date.now();
  markClip('automatic');
}
function drawBox(ctx, box, vw, vh, scale, x, y, color, label) {
  ctx.strokeStyle = color; ctx.fillStyle = color;
  ctx.strokeRect(x + box[0] * vw * scale, y + box[1] * vh * scale, (box[2] - box[0]) * vw * scale, (box[3] - box[1]) * vh * scale);
  if (label) ctx.fillText(label, x + box[0] * vw * scale, y + box[1] * vh * scale - 4);
}
async function frames() {
  const video = $('live'), canvas = document.createElement('canvas'), overlay = $('overlay');
  const ctx = canvas.getContext('2d'), ox = overlay.getContext('2d');
  while (media) {
    if (video.videoWidth) {
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
      if (blob && blob.size < 2 * 1024 * 1024) {
        try { drawLive(await api('/v1/recognize-frame?vehicles=1', { method: 'POST', body: blob, headers: { 'Content-Type': 'image/jpeg' } }), ox, overlay, video); }
        catch (error) { if (!String(error.message).includes('429')) $('plate').textContent = '车牌 —'; }
      }
      if (!recording && parts.length && Date.now() - parts[0].t > 35000) await finalizeClip();
    }
    await new Promise(resolve => { frameTimer = setTimeout(resolve, 2500); });
  }
}
function drawLive(out, ctx, overlay, video) {
  const plates = out.plates || [], vehicles = out.vehicles || [], lights = out.lights || [];
  votes = [...votes, plates.map(p => p.text).filter(Boolean)].slice(-4);
  const plate = confirmedPlate();
  $('plate').textContent = plate ? `车牌 ${plate}（稳定）` : (plates[0] && plates[0].text ? `车牌 ${plates[0].text}` : '车牌 等待清晰车牌');
  signals = [...signals, out.signal_observed || 'OFF'].slice(-3);
  const signal = signals[signals.length - 1];
  $('signal').textContent = `信号灯 ${title(signal)}${redStable() || (signals.length >= 2 && signals.slice(-2).every(v => v === signal) && signal !== 'OFF') ? '（稳定）' : ''}`;
  const kinds = [...new Set(vehicles.map(v => title(v.label)))].join(' / ');
  $('vehicles').textContent = `车辆 ${kinds || '未检出'}`;
  updateApproach(vehicles.length ? vehicles : plates);
  const hint = liveHint();
  $('hint').textContent = hint;
  $('verdict').textContent = `片段判定 ${hint || '—'}`;
  overlay.width = overlay.clientWidth; overlay.height = overlay.clientHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const scale = Math.min(overlay.width / vw, overlay.height / vh);
  const x = (overlay.width - vw * scale) / 2, y = (overlay.height - vh * scale) / 2;
  ctx.lineWidth = 2; ctx.font = '14px sans-serif';
  for (const item of vehicles) {
    const box = item.box_normalized; if (!box || box.length !== 4) continue;
    drawBox(ctx, box, vw, vh, scale, x, y, '#4affaf', `${title(item.label)} ${Math.round((item.score || 0) * 100)}%`);
  }
  for (const lamp of lights) {
    const box = lamp.box_normalized; if (!box || box.length !== 4) continue;
    const color = lamp.color === 'RED' ? '#ff5050' : lamp.color === 'GREEN' ? '#50dc78' : lamp.color === 'YELLOW' ? '#ffd250' : '#ddd';
    drawBox(ctx, box, vw, vh, scale, x, y, color, title(lamp.color));
  }
  for (const row of plates) {
    const box = row.box_normalized; if (!box || box.length !== 4) continue;
    drawBox(ctx, box, vw, vh, scale, x, y, '#ffe184', row.text || '');
  }
  maybeAutoMark();
}
async function upload(blob, trigger, candidate) {
  if (!blob || blob.size < 1 || blob.size > MAX) throw new Error('视频为空或超过 50 MiB');
  const hash = await sha256hex(await blob.arrayBuffer());
  const meta = { event_id: uuid(), trigger, manual_review: false, app_version: 'web-2.0', candidate_type: candidate || (trigger === 'automatic' ? 'RED_LIGHT' : 'UNKNOWN') };
  const type = blob.type || (mime().startsWith('video/mp4') ? 'video/mp4' : 'video/webm');
  await api('/v1/tasks', { method: 'POST', body: blob, headers: { 'Content-Type': type, 'X-Video-SHA256': hash, 'X-Event-Metadata': JSON.stringify(meta) } });
}
async function markClip(trigger) {
  if (recording || !recorder || recorder.state !== 'recording') { if (trigger === 'manual') status('请先开启相机'); return; }
  if (trigger !== 'import' && !confirmedPlate()) { status('未确认车牌，已丢弃'); return; }
  recording = true; $('hint').textContent = '正在截取重点片段';
  status(trigger === 'automatic' ? '自动截取闯红灯候选' : '截取触发前缓存 + 触发后 10 秒');
  await new Promise(resolve => setTimeout(resolve, 10000));
  try {
    const blob = await finalizeClip();
    recording = false; $('hint').textContent = '';
    status('正在上传'); await upload(blob, trigger); status('已上传，服务器自动分析'); await tick();
  } catch (error) { recording = false; $('hint').textContent = ''; status(error.message); }
}
$('cam').onclick = async () => {
  if (media) {
    clearTimeout(frameTimer); media.getTracks().forEach(track => track.stop()); media = null;
    try { recorder?.stop(); } catch (error) { /* already stopped */ }
    recorder = null; header = null; parts = []; votes = []; signals = []; approaching = false; approachFirst = -1; approachSamples = 0;
    $('live').srcObject = null; $('cam').textContent = '开启相机'; $('mark').disabled = true; status('相机已关闭'); return;
  }
  const devices = navigator.mediaDevices;
  if (!devices || !devices.getUserMedia) {
    status(location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1'
      ? '当前浏览器没有摄像头接口'
      : `当前是 HTTP，浏览器关闭了摄像头。请改用 https://${location.hostname}:61612 （证书警告点继续）。导入视频仍可用。`);
    return;
  }
  if (typeof MediaRecorder === 'undefined') { status('当前浏览器不支持录像，可改用导入视频'); return; }
  try {
    media = await devices.getUserMedia({ audio: false, video: { facingMode: { ideal: 'environment' }, aspectRatio: { ideal: 16 / 9 }, width: { ideal: 1920 }, height: { ideal: 1080 } } });
    $('live').srcObject = media; armRecorder(); $('cam').textContent = '关闭相机'; $('mark').disabled = false;
    status('相机已开启：稳定红灯且车牌确认、前车仍在接近时会自动截取'); frames();
  } catch (error) { status('无法打开相机：需要允许摄像头权限。' + error.message); }
};
$('mark').onclick = () => markClip('manual');
$('file').onchange = async () => {
  const file = $('file').files[0]; $('file').value = ''; if (!file) return;
  try { status('正在上传'); await upload(file, 'import'); status('已上传，服务器自动分析'); await tick(); }
  catch (error) { status(error.message); }
};
$('api').value = apiRoot;
$('settingsBtn').onclick = () => { $('api').value = apiRoot; $('settings').showModal(); };
$('closeSettings').onclick = () => $('settings').close();
$('settingsForm').onsubmit = async event => {
  event.preventDefault();
  try {
    const raw = $('api').value.trim();
    if (!raw) {
      apiRoot = ''; localStorage.setItem('traffic-api', ''); session = ''; $('settings').close(); await connect(); return;
    }
    const url = new URL(raw);
    if (!['http:', 'https:'].includes(url.protocol) || url.pathname !== '/') throw new Error('请填写 API 根地址');
    if (url.protocol === 'http:' && !isLanHost(url.hostname)) throw new Error('远程需要 HTTPS');
    if (location.protocol === 'https:' && url.protocol === 'http:') throw new Error('HTTPS 页面不能请求 HTTP 接口，请留空走同页转发');
    apiRoot = url.origin; localStorage.setItem('traffic-api', apiRoot); session = ''; $('settings').close(); await connect();
  } catch (error) { $('settingsError').textContent = error.message; }
};
$('refresh').onclick = () => tick();
connect(); setInterval(tick, 4000);
