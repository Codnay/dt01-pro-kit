/* DT-01 dashboard.
 *
 * The page is a view. It holds no authority over the robot: the deadman, the
 * motion gate and the stop-on-every-exit-path logic all live in the server, so
 * closing this tab is a safe way to stop driving rather than a way to lose
 * control of a moving robot.
 */

const $  = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

let snapshot = { connected: false };
let params = [];
let points = [];
let pathPoints = [];
let mapLoadedFor = null;

/* -------------------------------------------------------------- identity
   The control token identifies this BROWSER, not this tab, and lives in
   localStorage on purpose: a reload must not drop control of a moving robot,
   and two tabs of the same person are the same operator. */

const me = {
  get id() {
    let id = localStorage.getItem('dt01.client');
    if (!id) {
      id = (crypto.randomUUID ? crypto.randomUUID()
        : String(Date.now()) + Math.random().toString(16).slice(2));
      localStorage.setItem('dt01.client', id);
    }
    return id;
  },
  get name() { return localStorage.getItem('dt01.name') || ''; },
  set name(value) { localStorage.setItem('dt01.name', value); },
};

let holding = false;   // does THIS browser hold the lease?
let dragging = false;  // a speed slider is under the operator's finger

/* ------------------------------------------------------------------ api */

async function api(path, body) {
  const options = body === undefined
    ? { method: 'GET' }
    : { method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Control-Token': me.id },
        body: JSON.stringify(body) };
  let response, payload;
  try {
    response = await fetch(path, options);
    payload = await response.json();
  } catch (error) {
    toast(`${path} failed: ${error.message}`, 'bad');
    throw error;
  }
  if (!response.ok || payload.ok === false) {
    const message = payload.error || `HTTP ${response.status}`;
    toast(message, 'bad');
    throw new Error(message);
  }
  return payload.data;
}

function toast(message, kind = '') {
  const node = document.createElement('div');
  node.className = kind;
  node.textContent = message;
  $('#toast').appendChild(node);
  setTimeout(() => node.remove(), kind === 'bad' ? 6500 : 3200);
}

const fmt = (value, digits = 2, fallback = '—') =>
  (value === null || value === undefined || value === '' || Number.isNaN(Number(value)))
    ? fallback : Number(value).toFixed(digits);

/* --------------------------------------------------------------- stream */

let stream = null;
let wantScan = true;

/* The server sends `{full, d}`: a whole snapshot, or only the top-level keys
 * that changed since the last frame. Merging deltas is what keeps this usable
 * on a metered phone hotspot. A full frame arrives on connect and every ten
 * seconds, so a dropped or reordered update cannot leave the page stuck. */
function connectStream() {
  if (stream) stream.close();
  const rate = Number(localStorage.getItem('dt01.hz') || '') || null;
  const query = `?scan=${wantScan ? 1 : 0}${rate ? `&hz=${rate}` : ''}`;
  stream = new EventSource(`/api/stream${query}`);
  stream.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      render(message.full ? message.d : Object.assign({}, snapshot, message.d));
    } catch (_) { /* a torn frame; the next one wins */ }
  };
  stream.onerror = () => { /* EventSource reconnects on its own */ };
}

/* --------------------------------------------------------------- render */

function render(next) {
  snapshot = next;
  renderControl();
  renderHeader();
  renderSafety();
  renderDrive();
  syncSpeedFromServer();
  renderObstacles();
  renderSpeedLive();
  renderNavigate();
  renderCharging();
  renderLinkHealth();
  renderNotices();
  renderMap();
  renderMapSummary();
}

function chip(node, kind, text) {
  node.className = `chip ${kind}`;
  node.lastElementChild.textContent = text;
}

function renderHeader() {
  const connected = snapshot.connected;
  $('#linkHost').textContent = connected
    ? `${snapshot.link.host}:${snapshot.link.port}` : 'not connected';

  // "connected" only means a robot address is configured. Whether the
  // controller is answering is a separate question, and the more useful one.
  const state = (snapshot.robot || {}).state_operation;
  const reachable = snapshot.reachable !== false;
  chip($('#chipState'),
       !connected ? 'bad' : (!reachable ? 'bad' : (state === 'RunningTask' ? 'ok' : '')),
       !connected ? 'offline' : (!reachable ? 'no reply' : (state || 'unknown')));

  const localization = snapshot.localization || {};
  const located = snapshot.localized !== undefined
    ? snapshot.localized
    : (localization.status === 'normal' || (snapshot.robot || {}).state_initialized);
  chip($('#chipLoc'), located ? 'ok' : 'warn',
       localization.status || (located ? 'localised' : 'not localised'));

  // Blind interlocks are a safety condition, not a footnote.
  const safety = $('#chipSafety');
  if (safety) {
    const blind = snapshot.connected && snapshot.safety_stale;
    safety.hidden = !blind;
    if (blind) chip(safety, 'bad', 'interlocks blind');
  }

  const feed = (snapshot.link || {}).notice_feed || {};
  chip($('#chipFeed'), feed.connected ? 'ok' : 'warn', feed.connected ? 'live' : 'polling');
  $('#chipFeed').firstElementChild.classList.toggle('pulse', !!feed.connected);

  const battery = snapshot.battery || {};
  const percent = battery.battery_capacity_percentage;
  $('#statBattery').textContent = percent === undefined ? '—' : `${fmt(percent, 0)}%`;
  $('#statSpeed').textContent = fmt((snapshot.velocity || {}).linear);
  $('#statTurn').textContent = fmt((snapshot.velocity || {}).angular);

  const braked = (snapshot.brake || {}).engaged;
  const button = $('#estopBtn');
  button.classList.toggle('released', !!braked);
  button.textContent = braked ? 'RELEASE E-STOP' : 'EMERGENCY STOP';
}

function renderSafety() {
  let reason = snapshot.connected ? snapshot.motion_blocked_by : 'not connected to a robot';
  if (reason && reason.startsWith('nobody has taken control')) reason = null;
  const bar = $('#safetyBar');
  bar.classList.toggle('show', !!reason);
  if (reason) $('#safetyText').textContent = reason;
}

/* ---------------------------------------------------------------- drive */

const held = new Set();
const KEY_MAP = {
  w: 'fwd', arrowup: 'fwd', s: 'back', arrowdown: 'back',
  a: 'left', arrowleft: 'left', d: 'right', arrowright: 'right',
};
// Mirrors snapshot.settings. Server-side so every viewer reads the same
// limits and a reload does not silently reset them.
let speed = { linear: 0.2, angular: 0.45 };
let driveTimer = null;

function driveVector() {
  let linear = 0, angular = 0;
  if (held.has('fwd')) linear += speed.linear;
  if (held.has('back')) linear -= speed.linear;
  // Positive angular swings this chassis left. The pad used to send the
  // opposite, so the left button turned the robot right and vice versa.
  if (held.has('left')) angular += speed.angular;
  if (held.has('right')) angular -= speed.angular;
  return { linear, angular };
}

function pumpDrive() {
  if (!held.size) return;
  const vector = driveVector();
  api('/api/drive', vector).catch(() => releaseAll());
}

function startPump() {
  if (driveTimer) return;
  driveTimer = setInterval(pumpDrive, 100);        // 10 Hz, matching the server loop
  pumpDrive();
}

function stopPump() {
  clearInterval(driveTimer);
  driveTimer = null;
}

function hold(action) {
  if (action === 'halt') { releaseAll(); return; }
  if (held.has(action)) return;
  held.add(action);
  paintPad();
  startPump();
}

function release(action) {
  if (!held.delete(action)) return;
  paintPad();
  if (!held.size) { stopPump(); api('/api/drive/stop', {}).catch(() => {}); }
}

function releaseAll() {
  held.clear();
  paintPad();
  stopPump();
  api('/api/drive/stop', {}).catch(() => {});
}

function paintPad() {
  $$('.pad button[data-drive]').forEach((button) => {
    button.classList.toggle('held', held.has(button.dataset.drive));
  });
}

function renderDrive() {
  const drive = snapshot.drive || {};
  const blocked = snapshot.motion_blocked_by;
  $$('.pad button[data-drive]').forEach((button) => {
    // STOP stays live for everyone, always — including read-only viewers and
    // while the gate is refusing. Everything that STARTS motion needs both a
    // clear gate and this browser holding the lease.
    const isStop = button.dataset.drive === 'halt';
    button.disabled = isStop ? !snapshot.connected
      : (!snapshot.connected || !!blocked || !holding);
  });
  const bar = $('#deadmanBar');
  const age = drive.intent_age;
  const window_ = drive.deadman_s || 0.6;
  const remaining = age === null || age === undefined ? 0
    : Math.max(0, 1 - age / window_);
  bar.firstElementChild.style.width = `${remaining * 100}%`;
  bar.classList.toggle('stale', remaining === 0);
  $('#deadmanS').textContent = window_;

  const note = $('#driveModeNote');
  const state = (snapshot.robot || {}).state_operation;
  const wrong = snapshot.connected && state && state !== 'Idle';
  note.hidden = !wrong;
  if (wrong) {
    note.innerHTML = `The robot is in <b>${state}</b>. Manual driving works best from
      <b>Idle</b> — in RunningTask the navigator issues its own velocity commands and will
      fight the pad.`;
  }
  $$('[data-state]').forEach((button) => {
    button.classList.toggle('on', button.dataset.state === state);
    button.disabled = !snapshot.connected;
  });
  renderLift();
}

function renderControl() {
  const info = snapshot.control || { held: false };
  holding = !!(info.held && info.holder === me.id);
  const button = $('#controlBtn');

  // State as a chip, like every other status. The explanation lives next to
  // the controls it affects, not across the top of every screen.
  if (holding) {
    chip($('#chipControl'), 'ok', 'you have control');
    button.textContent = 'Release';
  } else if (info.held) {
    chip($('#chipControl'), 'warn', `${info.name} has control`);
    button.textContent = 'Take over';
  } else {
    chip($('#chipControl'), 'bad', 'no control');
    button.textContent = 'Take control';
  }
  button.classList.toggle('primary', !holding);

  // Read-only means read-only: every control that could command the robot is
  // disabled, not merely ignored by the server.
  $$('[data-needs-control]').forEach((node) => { node.disabled = !holding; });
}

function escapeText(value) {
  const node = document.createElement('span');
  node.textContent = String(value === undefined || value === null ? '' : value);
  return node.innerHTML;
}

function syncSpeedFromServer() {
  // The server owns these now, so a second viewer sees the first one's changes
  // and a reload keeps them. Skipped while dragging, or the slider would fight
  // the hand holding it.
  const settings = snapshot.settings;
  if (!settings || dragging) return;
  speed.linear = settings.linear;
  speed.angular = settings.angular;
  const lin = $('#linRange');
  const ang = $('#angRange');
  if (document.activeElement !== lin) lin.value = settings.linear;
  if (document.activeElement !== ang) ang.value = settings.angular;
  $('#linVal').textContent = Number(settings.linear).toFixed(2);
  $('#angVal').textContent = Number(settings.angular).toFixed(2);
}

function renderLift() {
  const lift = snapshot.lift || {};
  const height = lift.height_mm;
  const min = lift.min_mm ?? 50;
  const max = lift.max_mm ?? 850;

  $('#liftHeight').textContent = height === undefined ? '—' : height;
  $('#liftMin').textContent = min;
  $('#liftMax').textContent = max;
  // Measured, not the motor percentage: nothing in the protocol says how fast
  // a given percent moves this column.
  $('#liftRate').textContent = lift.speed_mms
    ? `${lift.speed_mms > 0 ? '+' : ''}${lift.speed_mms} mm/s` : '';
  $('#liftAction').textContent = lift.seeking
    ? `→ ${lift.seek_target_mm} mm` : (lift.action || '');

  const bar = $('#liftBar');
  const span = Math.max(1, max - min);
  bar.style.width = height === undefined ? '0'
    : `${Math.max(0, Math.min(100, ((height - min) / span) * 100))}%`;
  bar.parentElement.classList.toggle('moving', !!lift.moving);

  const flags = ['limit_up', 'limit_down', 'stop_button', 'drive_error']
    .filter((key) => lift[key]);
  const reason = !lift.connected
    ? (lift.error || 'no feedback from the lifting column')
    : (lift.seek_error || (flags.length ? flags.join(', ').replace(/_/g, ' ') : ''));
  const note = $('#liftBlocked');
  note.hidden = !reason;
  if (reason) note.textContent = reason;

  const slider = $('#liftGoto');
  if (lift.min_mm !== undefined) { slider.min = min; slider.max = max; }
  const usable = !!lift.connected && !flags.length;
  ['#liftGotoBtn', '#liftGoto'].forEach((id) => {
    $(id).disabled = !usable || !holding;
  });
}

function renderObstacles() {
  const obstacles = snapshot.obstacles || {};
  const badge = $('#obsBadge');
  const protection = obstacles.protection || {};
  const latched = (snapshot.drive || {}).protection_latched;
  const blocked = protection.blocked_by || (obstacles.alarm && (obstacles.reasons || [])[0]);
  const message = latched || (blocked || obstacles.alarm
    ? `Manual movement blocked: ${blocked || 'obstacle detected'}`
    : protection.ready
      ? 'Protection active: configured ultrasonic and cliff readings are ready.'
      : `Warning: ${protection.warning || 'obstacle readings unavailable'}. Missing obstacle data does not block manual driving; obstacle protection is not verified.`);
  $('#protectionStatus').textContent = message;
  $('#driveProtection').textContent = message;
  $('#protectionOptional').textContent = 'Optional protections: ' + Object.entries(protection.optional || {})
    .map(([name, value]) => `${name.includes('camera') ? 'camera avoidance' : 'pre-start check'}: ${value === null ? 'unavailable' : value ? 'enabled' : 'off'}`).join(' · ');
  $('#protectionFreshness').textContent = `Sensor age limit: ${protection.sensor_stale_s ?? '—'} s. Configuration age limit: ${protection.params_stale_s ?? '—'} s.`;
  if (document.activeElement !== $('#sensorStale')) $('#sensorStale').value = protection.sensor_stale_s ?? 2;
  if (document.activeElement !== $('#alarmRange')) $('#alarmRange').value = obstacles.alarm_distance ?? 0.35;
  $('#alarmVal').textContent = Number(obstacles.alarm_distance ?? 0.35).toFixed(2);
  badge.hidden = protection.ready && !latched;
  badge.textContent = latched || blocked || obstacles.alarm ? 'Obstacle stop' : 'Protection warning';

  drawFan(obstacles);

  const rows = (obstacles.sensors || []).map((sensor) => {
    const near = sensor.range > 0
      ? Math.max(0, 1 - sensor.range / Math.max(0.01, obstacles.alarm_distance * 3)) : 0;
    const classes = sensor.triggered ? 'trip' : (sensor.enabled ? '' : 'off');
    const state = sensor.status || (sensor.enabled === null ? 'configuration unknown' : 'unverified');
    return `<tr class="${classes}">
      <td>${escapeText(sensor.id)}</td>
      <td class="num">${sensor.range > 0 ? fmt(sensor.range) + ' m' : '—'}</td>
      <td><div class="bar"><i class="${sensor.triggered ? 'near' : ''}"
          style="width:${(near * 100).toFixed(0)}%"></i></div></td>
      <td>${escapeText(state)}</td></tr>`;
  }).join('');
  $('#sensorRows').innerHTML = rows || '<tr><td colspan="4">No sensor data yet.</td></tr>';

  const bumper = snapshot.bumper || {};
  $('#bumperRow').innerHTML = ['front', 'rear', 'estop'].map((key) => {
    const on = !!bumper[key];
    const label = key === 'estop' ? 'chassis e-stop' : `${key} bumper`;
    return `<span class="chip ${on ? 'bad' : 'ok'}"><span class="dot"></span>${label}</span>`;
  }).join('');

  const devices = (snapshot.chassis || {}).devices || {};
  const chassisRows = Object.entries(devices).map(([name, entry]) => {
    const detail = entry.faulty_indices && entry.faulty_indices.length
      ? `sensors ${entry.faulty_indices.map((index) => index + 1).join(', ')} faulty`
      : (entry.healthy ? 'healthy' : 'fault');
    return `<tr class="${entry.healthy ? '' : 'trip'}">
      <td>${name}</td><td class="num">${entry.code}</td><td>${detail}</td></tr>`;
  }).join('');
  $('#chassisRows').innerHTML = chassisRows
    || '<tr><td colspan="3">No chassis data yet.</td></tr>';
}

function drawFan(obstacles) {
  const canvas = $('#fan');
  const ratio = window.devicePixelRatio || 1;
  const box = canvas.getBoundingClientRect();
  if (!box.width) return;
  canvas.width = box.width * ratio;
  canvas.height = box.height * ratio;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);

  const cx = box.width / 2, cy = box.height - 26;
  const maxRange = Math.max(1.2, (obstacles.alarm_distance || 0.35) * 3);
  const radius = Math.min(box.width / 2 - 40, box.height - 46);

  ctx.fillStyle = '#5d6b7c';
  ctx.font = '10px ui-monospace, monospace';
  for (let ring = 1; ring <= 3; ring++) {
    const distance = (maxRange / 3) * ring;
    const at = (distance / maxRange) * radius;
    ctx.strokeStyle = '#2a3341';
    ctx.beginPath();
    ctx.arc(cx, cy, at, Math.PI, 2 * Math.PI);
    ctx.stroke();
    // Labels sit outside the arc on the right, clear of the sensor rays.
    ctx.fillText(`${distance.toFixed(1)}m`, cx + at + 5, cy - 2);
  }
  // The alarm threshold, so "why did it trip" is answerable at a glance.
  const alarm = (obstacles.alarm_distance / maxRange) * radius;
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = '#d29922';
  ctx.beginPath();
  ctx.arc(cx, cy, alarm, Math.PI, 2 * Math.PI);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#d29922';
  ctx.fillText('alarm', cx - alarm - 34, cy - 2);
  ctx.fillStyle = '#5d6b7c';

  for (const sensor of obstacles.sensors || []) {
    if (!sensor.enabled) continue;
    // Sensor angle is measured in the robot frame, CCW positive, forward = +x.
    // On screen forward is up, so a sensor at angle a sits at -90deg + a.
    const screen = -Math.PI / 2 + (sensor.angle || 0);
    const distance = sensor.range > 0 ? Math.min(sensor.range, maxRange) : maxRange;
    const length = (distance / maxRange) * radius;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(screen) * length, cy + Math.sin(screen) * length);
    ctx.strokeStyle = sensor.triggered ? '#f85149'
      : (sensor.range > 0 ? '#4a9eff' : '#212a35');
    ctx.lineWidth = sensor.triggered ? 3 : 1.5;
    ctx.stroke();
    if (sensor.range > 0) {
      ctx.beginPath();
      ctx.arc(cx + Math.cos(screen) * length, cy + Math.sin(screen) * length, 3, 0, Math.PI * 2);
      ctx.fillStyle = sensor.triggered ? '#f85149' : '#4a9eff';
      ctx.fill();
    }
  }

  ctx.fillStyle = '#e6edf3';
  ctx.beginPath();
  ctx.moveTo(cx, cy - 9); ctx.lineTo(cx - 7, cy + 5); ctx.lineTo(cx + 7, cy + 5);
  ctx.closePath();
  ctx.fill();

  if (obstacles.alarm) {
    ctx.fillStyle = '#f85149';
    ctx.font = '700 12px ui-monospace, monospace';
    ctx.fillText('OBSTACLE', 8, 16);
    ctx.fillStyle = '#ffb3ad';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText((obstacles.reasons || []).slice(0, 2).join(' · '), 8, 31);
  }
}

/* ---------------------------------------------------------------- speed */

function renderSpeedLive() {
  const drive = snapshot.drive || {};
  const velocity = snapshot.velocity || {};
  $('#spCommanded').textContent = `${fmt(drive.linear)} m/s · ${fmt(drive.angular)} rad/s`;
  $('#spReported').textContent = `${fmt(velocity.linear)} m/s · ${fmt(velocity.angular)} rad/s`;
  $('#spCap').textContent = `${fmt(drive.max_linear)} m/s · ${fmt(drive.max_angular)} rad/s`;
}

/* ------------------------------------------------------------- navigate */

function renderNavigate() {
  const nav = snapshot.nav || {};
  const process = nav.process || {};
  const running = nav.nav_status_type && nav.nav_status_type !== 'idle';
  $('#navProgressCard').hidden = !running;
  if (running) {
    $('#navTarget').textContent = process.next_point || process.path_name || '—';
    $('#navStatus').textContent = nav.nav_status_type
      + ((nav.nav_status_data || {}).information ? ` · ${nav.nav_status_data.information}` : '');
    $('#navRemaining').textContent = `${fmt(process.remaining_milleage)} m`;
    $('#navEta').textContent = `${fmt(process.remaining_time, 0)} s`;
  }
  const localization = snapshot.localization || {};
  $('#locStatus').textContent = localization.status || '—';
  $('#locScore').textContent = fmt(localization.score, 3);
  $('#locAvg').textContent = fmt(localization.map_average, 3);
}

/* ------------------------------------------------------------- charging */

/* A field that is present but never filled is not a reading. This firmware
 * reports battery_current and battery_temperature as a constant 0.0 and an
 * empty chassis_state, so rendering those as "0.0 A" and "0.0 °C" invents
 * measurements that were never taken. Show them as unreported instead. */
const unreported = (value) =>
  value === undefined || value === null || Number(value) === 0;

function renderCharging() {
  const battery = snapshot.battery || {};
  const percent = Number(battery.battery_capacity_percentage);
  const fill = $('#battFill');
  fill.style.width = Number.isFinite(percent) ? `${Math.max(0, Math.min(100, percent))}%` : '0%';
  fill.className = percent < 15 ? 'crit' : (percent < 30 ? 'low' : '');

  // Tri-state: true / false / null-for-unknown. Null must not read as "no".
  const charging = snapshot.charging === undefined ? null : snapshot.charging;
  const suffix = charging === true ? ' — CHARGING' : '';
  $('#battText').textContent = Number.isFinite(percent)
    ? `${percent.toFixed(0)}%${suffix}` : '—';

  const current = Number(battery.battery_current);
  $('#battCurrent').textContent = unreported(battery.battery_current)
    ? 'not reported by this robot'
    : `${current > 0 ? '+' : ''}${current.toFixed(2)} A (${current > 0 ? 'charging' : 'not charging'})`;

  // The controller may report decivolts; the server normalises and passes volts.
  const volts = battery.volts;
  $('#battVoltage').textContent = volts ? `${fmt(volts)} V` : '—';
  $('#battTemp').textContent = unreported(battery.battery_temperature)
    ? 'not reported by this robot' : `${fmt(battery.battery_temperature, 0)} °C`;
  $('#battChassis').textContent = charging === null
    ? 'no charge signal available on this firmware'
    : (charging ? 'charging' : 'not charging');
}

/* ---------------------------------------------------------------- links */

function renderLinkHealth() {
  const link = snapshot.link || {};
  const notice = link.notice_feed || {}, scan = link.scan_feed || {};
  const errors = Object.entries(link.poll_errors || {});
  $('#linkHealth').innerHTML = [
    ['notice feed', notice.connected ? `live · ${notice.messages} msgs`
      : `down${notice.error ? ' · ' + notice.error : ''}`],
    ['laser feed', scan.connected ? `live · ${scan.messages} msgs`
      : `down${scan.error ? ' · ' + scan.error : ''}`],
    ...errors.map(([name, error]) => [`poll: ${name}`, error]),
  ].map(([key, value]) =>
    `<dt>${key}</dt><dd style="text-align:left;color:${
      String(value).startsWith('live') ? '#3fb950' : '#d29922'}">${value}</dd>`).join('');
}

function selectTab(name) {
  const button = $(`.tabs button[data-tab="${name}"]`);
  if (!button) return;
  $$('.tabs button').forEach((other) => other.setAttribute('aria-selected', other === button));
  $$('.panel').forEach((panel) =>
    panel.classList.toggle('active', panel.id === `panel-${name}`));
  if (location.hash.slice(1) !== name) history.replaceState(null, '', `#${name}`);
  if (name === 'navigate' || name === 'charging') loadPoints().catch(() => {});
  if (name === 'obstacles') renderObstacles();
}

function renderNotices() {
  const notices = snapshot.notices || [];
  const node = $('#notices');
  if (!notices.length) { node.innerHTML = '<div class="empty">No notices yet.</div>'; return; }
  node.innerHTML = notices.map((notice) => {
    const level = String(notice.notice_level || 'INFO').toUpperCase();
    const kind = level === 'ERROR' || level === 'FATAL' ? 'bad' : (level === 'WARN' ? 'warn' : '');
    return `<div class="row ${kind}"><b>${notice.target_name || '—'}</b>
      <span>${notice.message || ''}</span></div>`;
  }).join('');
}

/* ------------------------------------------------------------------ map */

function renderMapSummary() {
  const info = ((snapshot.pose || {}).mapinfo) || {};
  const world = (snapshot.pose || {}).world;
  const bits = [];
  if (info.mapname) bits.push(info.mapname);
  if (world) bits.push(`x ${world.x} · y ${world.y}`);
  const score = (snapshot.localization || {}).score;
  if (score !== undefined && score !== null) bits.push(`loc ${score}`);
  $('#mapSummary').textContent = bits.join('   ');
}

function renderMap() {
  if (!snapshot.connected) return;
  const pose = snapshot.pose || {};
  const info = pose.mapinfo || (snapshot.scan || {}).mapinfo || snapshot.map;
  if (info && info.gridWidth) MapView.setMapInfo(info);
  const name = (snapshot.map || {}).name || (info || {}).mapname;
  if (name && name !== mapLoadedFor) {
    mapLoadedFor = name;
    MapView.loadImage(`/api/map.png?name=${encodeURIComponent(name)}&t=${Date.now()}`);
    loadPoints().catch(() => {});
  }
  MapView.update({
    pose, points, path: pathPoints, scan: snapshot.scan,
    alarm: (snapshot.obstacles || {}).alarm,
  });
}

/* -------------------------------------------------------------- loaders */

async function loadPoints() {
  const data = await api('/api/points');
  points = data.points || [];
  $('#pointsHint').textContent = `${points.length} point(s)`;
  $('#pointRows').innerHTML = points.length ? points.map((point) => {
    const position = (point.world || {}).position || {};
    return `<tr class="click">
      <td>${point.name}</td>
      <td><span class="badge-type ${point.type_name}">${point.type_name}</span></td>
      <td class="num">${fmt(position.x)}</td>
      <td class="num">${fmt(position.y)}</td>
      <td><button class="btn primary" data-goto="${point.name}"
        data-needs-control ${holding ? '' : 'disabled'}>Go</button></td></tr>`;
  }).join('') : '<tr><td colspan="5">No points recorded on this map.</td></tr>';

  const charge = points.filter((point) => point.type_name === 'charge');
  fillSelect($('#dockPoint'), charge.map((point) => point.name), '— charge point —');
  fillSelect($('#locPoint'), points.map((point) => point.name), '— point to initialise at —');
  renderAutocharge(charge);
  MapView.update({ points });
}

function fillSelect(select, values, placeholder) {
  const previous = select.value;
  select.innerHTML = `<option value="">${placeholder}</option>`
    + values.map((value) => `<option>${value}</option>`).join('');
  if (values.includes(previous)) select.value = previous;
  // One candidate is not a choice — preselect it rather than making the
  // operator open a dropdown to pick the only dock on the map.
  else if (values.length === 1) select.value = values[0];
}

async function loadParams(refresh = false) {
  const data = await api(`/api/params${refresh ? '?refresh=1' : ''}`);
  params = data.params || [];
  if (data.error) toast(`parameters: ${data.error}`, 'bad');
  paintParams('#speedParams', 'speed');
  paintParams('#obstacleParams', 'obstacle');
  paintParams('#chargeParams', 'charge');
  paintParams('#batteryParams', 'battery');
}

async function renderAutocharge(chargePoints) {
  let data;
  try { data = await api('/api/charge'); } catch (_) { return; }
  const box = $('#autochargeBox');
  if (!data.autocharge_available) {
    box.innerHTML = `<div class="note">
        The controller's auto-charge endpoints are not answering
        (<code>${data.autocharge_error || 'unavailable'}</code>). The API reference marks them
        <b>未启用 — not enabled</b>, so this is the documented state, not a fault.
        <br><br>Use the trigger level in <b>Setup &rarr; Battery thresholds</b>
        (<code>auto_charge_percentage</code>) and enable auto charge in the vendor Robot APP,
        or dock manually below.</div>`;
    return;
  }
  const pile = $('#pileBox');
  if (pile) {
    pile.innerHTML = data.charging_pile
      ? `<pre style="margin:0;font:12px/1.5 var(--mono);white-space:pre-wrap">${
          JSON.stringify(data.charging_pile, null, 2)}</pre>`
      : `<span class="hint">not reported${data.charging_pile_error
          ? ` — <code>${data.charging_pile_error}</code>` : ''}</span>`;
  }
  const status = data.autocharge || {};
  box.innerHTML = `
    <dl class="kv">
      <dt>Switch</dt><dd>${status.enabled ? 'on' : 'off'}</dd>
      <dt>State</dt><dd>${status.state || '—'} — ${status.state_text || ''}</dd>
      <dt>Dock point</dt><dd>${status.point || '—'}</dd>
    </dl>
    <div class="row" style="margin-top:11px">
      <select id="acPoint"></select>
      <button class="btn ${status.enabled ? 'on' : 'primary'}" id="acOn">Enable</button>
      <button class="btn" id="acOff">Disable</button>
    </div>`;
  fillSelect($('#acPoint'), (chargePoints || []).map((point) => point.name), '— charge point —');
  if (status.point) $('#acPoint').value = status.point;
  $('#acOn').onclick = async () => {
    const point = $('#acPoint').value;
    if (!point) { toast('pick a charge point first', 'bad'); return; }
    await api('/api/charge/auto', { enabled: true, point });
    toast(`auto charge enabled at ${point}`, 'ok');
    renderAutocharge(chargePoints);
  };
  $('#acOff').onclick = async () => {
    await api('/api/charge/auto', { enabled: false, point: '' });
    toast('auto charge disabled', 'ok');
    renderAutocharge(chargePoints);
  };
}

/* ----------------------------------------------------------- parameters */

function paintParams(selector, group) {
  const node = $(selector);
  const entries = params.filter((param) => param.group === group);
  if (!entries.length) { node.innerHTML = '<p class="hint">No parameters reported.</p>'; return; }
  node.innerHTML = entries.map((param) => {
    const html = paramField(param);
    const safety = /switch_ultrasound|switch_fallprevention|avoid_obstacle|switch_camera|check_to_start/.test(param.namespace);
    return safety ? html.replaceAll('<input ', '<input disabled ') : html;
  }).join('')
    + `<div class="row"><button class="btn primary" data-apply="${group}">Apply changes</button>
       <span class="hint">Only edited fields are sent.</span></div>`;
  node.querySelectorAll('[data-apply]').forEach((button) => {
    button.onclick = () => applyParams(group, node);
  });
}

function paramField(param) {
  const id = `p_${param.namespace.replace(/[^\w]/g, '_')}`;
  const warn = param.warn ? `<div class="note bad" style="margin:6px 0 0">${param.warn}</div>` : '';
  const help = param.help ? `<span class="hint">${param.help}</span>` : '';
  let control;

  if (param.isArray) {
    const values = Array.isArray(param.value) ? param.value : [];
    control = `<div class="arr" data-array="${id}">` + values.map((value, index) =>
      param.type === 'bool'
        ? `<label class="inline"><input type="checkbox" data-i="${index}"
             ${String(value).toLowerCase() === 'true' ? 'checked' : ''}>${index + 1}</label>`
        : `<input type="number" step="0.01" data-i="${index}" value="${value}"
             title="sensor ${index + 1}">`).join('') + '</div>';
  } else if (param.type === 'bool') {
    control = `<label class="inline"><input type="checkbox" id="${id}"
      ${String(param.value).toLowerCase() === 'true' ? 'checked' : ''}> enabled</label>`;
  } else if (param.lower !== null && param.lower !== undefined) {
    // The controller publishes its own bounds; honour them instead of guessing.
    const options = [];
    for (let value = Number(param.lower); value <= Number(param.upper); value++) {
      const label = { 0: 'low', 1: 'medium', 2: 'high' }[value];
      options.push(`<option value="${value}" ${Number(param.value) === value ? 'selected' : ''}>
        ${value}${label ? ` — ${label}` : ''}</option>`);
    }
    control = `<select id="${id}">${options.join('')}</select>`;
  } else {
    control = `<input type="number" id="${id}" step="${param.type === 'int' ? '1' : '0.01'}"
      value="${param.value}">`;
  }

  return `<div class="row" data-ns="${param.namespace}" data-type="${param.type}"
              data-isarray="${param.isArray}" style="align-items:flex-start;margin-bottom:11px">
    <label class="field" style="flex:1">
      <span><b>${param.label}</b>${param.unit ? ` (${param.unit})` : ''}</span>
      ${control}${help}${warn}
    </label></div>`;
}

async function applyParams(group, node) {
  const updates = [];
  node.querySelectorAll('[data-ns]').forEach((row) => {
    const namespace = row.dataset.ns;
    const isArray = row.dataset.isarray === 'true';
    if (isArray) {
      const inputs = Array.from(row.querySelectorAll('[data-i]'));
      if (!inputs.length) return;
      updates.push({ namespace, value: inputs.map((input) =>
        input.type === 'checkbox' ? String(input.checked) : input.value) });
    } else {
      const input = row.querySelector('input, select');
      if (!input) return;
      updates.push({ namespace,
                     value: input.type === 'checkbox' ? String(input.checked) : input.value });
    }
  });
  if (!updates.length) return;
  const data = await api('/api/params', { updates });
  params = data.params || params;
  toast(`${group}: ${updates.length} parameter(s) written`, 'ok');
  paintParams(`#${group === 'battery' ? 'batteryParams'
    : group === 'charge' ? 'chargeParams'
    : group === 'obstacle' ? 'obstacleParams' : 'speedParams'}`, group);
}

/* ----------------------------------------------------------------- wire */

function wire() {
  // --- tabs
  $$('.tabs button').forEach((button) => {
    button.onclick = () => selectTab(button.dataset.tab);
  });
  // The tab lives in the URL hash, so a reload — or a link someone pastes into
  // chat — comes back to the panel it was on.
  window.addEventListener('hashchange', () => selectTab(location.hash.slice(1)));
  selectTab(location.hash.slice(1) || 'drive');

  // --- emergency stop
  $('#estopBtn').onclick = async () => {
    const braked = (snapshot.brake || {}).engaged;
    releaseAll();
    const result = await api('/api/estop', { engaged: !braked });
    toast(braked ? 'brake released' : 'BRAKE ENGAGED — robot locked', braked ? 'ok' : 'bad');
    if (result.errors && result.errors.length) toast(result.errors.join('; '), 'bad');
  };

  // --- control lease
  const claimControl = async () => {
    const info = snapshot.control || { held: false };
    if (holding) {
      releaseAll();
      await api('/api/control/release', { client: me.id });
      toast('control released', 'ok');
      return;
    }
    let name = me.name;
    if (!name) {
      name = (prompt('Your name, so others know who is driving:') || '').trim();
      if (!name) return;
      me.name = name;
    }
    // Stealing is explicit. Silently preempting whoever is driving is how two
    // people end up both believing they have the robot.
    if (info.held && !confirm(`${info.name} currently has control. Take it anyway?`)) {
      return;
    }
    try {
      await api('/api/control/take', { client: me.id, name, force: !!info.held });
      toast('you have control', 'ok');
    } catch (error) { /* toasted by api() */ }
  };
  $('#controlBtn').onclick = claimControl;

  // Heartbeat. The lease expires without this, which is what makes a closed
  // tab or a dead link release the robot instead of latching it forever.
  setInterval(async () => {
    if (!holding) return;
    try {
      const result = await api('/api/control/heartbeat', { client: me.id });
      if (!result.still_holding) toast('you no longer have control', 'bad');
    } catch (error) { /* transient; the next beat retries */ }
  }, 2500);

  // Give the robot back on the way out. Best-effort: if it does not land, the
  // lease still expires on its own a few seconds later.
  window.addEventListener('pagehide', () => {
    if (holding) {
      navigator.sendBeacon('/api/control/release',
        new Blob([JSON.stringify({ client: me.id })], { type: 'application/json' }));
    }
  });

  // --- map column folds away. Remembered, because whether you want the map
  // depends on the job you keep coming back to do.
  const mapToggle = $('#mapToggle');
  const setMap = (open) => {
    document.querySelector('main').classList.toggle('map-collapsed', !open);
    mapToggle.setAttribute('aria-expanded', String(open));
    try { localStorage.setItem('dt01.map', open ? 'open' : 'shut'); } catch (error) { /* private mode */ }
    if (open) requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
  };
  let mapOpen = true;
  try { mapOpen = localStorage.getItem('dt01.map') !== 'shut'; } catch (error) { /* private mode */ }
  setMap(mapOpen);
  mapToggle.onclick = () => setMap(mapToggle.getAttribute('aria-expanded') !== 'true');

  // --- lifting column. Millimetres only; the server owns the travel limits,
  // the speed and the closed loop.
  $('#liftGoto').oninput = (event) => {
    $('#liftGotoVal').textContent = event.target.value;
  };
  $('#liftGotoBtn').onclick = async () => {
    const target = Number($('#liftGoto').value);
    try {
      const data = await api('/api/lift/move', { to_mm: target });
      if (data.already_there) { toast(`already at ${data.height_mm} mm`, 'ok'); return; }
      toast(data.clamped ? `clamped to the ${data.target_mm} mm limit`
        : `going to ${data.target_mm} mm`, 'ok');
    } catch (error) { /* toasted by api() */ }
  };
  // Never gated on the lease: anyone watching the column must be able to stop it.
  $('#liftStop').onclick = async () => {
    try {
      await api('/api/lift/stop', {});
      toast('lift stopped', 'ok');
    } catch (error) { /* toasted by api() */ }
  };

  // --- drive pad
  $$('.pad button[data-drive]').forEach((button) => {
    const action = button.dataset.drive;
    const down = (event) => { event.preventDefault(); hold(action); };
    const up = (event) => { event.preventDefault(); release(action); };
    button.addEventListener('mousedown', down);
    button.addEventListener('touchstart', down, { passive: false });
    ['mouseup', 'mouseleave', 'touchend', 'touchcancel'].forEach((name) =>
      button.addEventListener(name, up));
  });

  window.addEventListener('keydown', (event) => {
    if (event.target.matches('input, select, textarea')) return;
    if (event.key === 'Escape') { $('#estopBtn').click(); return; }
    if (event.key === ' ') { event.preventDefault(); releaseAll(); return; }
    const action = KEY_MAP[event.key.toLowerCase()];
    if (action) { event.preventDefault(); hold(action); }
  });
  window.addEventListener('keyup', (event) => {
    const action = KEY_MAP[event.key.toLowerCase()];
    if (action) release(action);
  });
  // Losing focus must not leave a key stuck down.
  window.addEventListener('blur', releaseAll);
  document.addEventListener('visibilitychange', () => { if (document.hidden) releaseAll(); });
  window.addEventListener('pagehide', () => navigator.sendBeacon('/api/drive/stop', '{}'));

  // --- speed. Shown immediately, persisted server-side so the other viewers
  // and the next reload agree. Pushed on release rather than per-pixel: a
  // slider drag would otherwise be one request per frame over a hotspot.
  const showSpeed = () => {
    speed.linear = Number($('#linRange').value);
    speed.angular = Number($('#angRange').value);
    $('#linVal').textContent = speed.linear.toFixed(2);
    $('#angVal').textContent = speed.angular.toFixed(2);
  };
  const pushSpeed = async () => {
    dragging = false;
    if (!holding) return;
    try {
      await api('/api/settings', { linear: speed.linear, angular: speed.angular });
    } catch (error) { /* toasted by api(); the server value wins on next frame */ }
  };
  ['#linRange', '#angRange'].forEach((id) => {
    const node = $(id);
    node.oninput = () => { dragging = true; showSpeed(); };
    node.onchange = pushSpeed;
  });
  $$('[data-preset]').forEach((button) => {
    button.onclick = () => {
      const preset = (snapshot.speed_presets || {})[button.dataset.preset];
      if (!preset) return;
      $('#linRange').value = preset.linear;
      $('#angRange').value = preset.angular;
      showSpeed();
      pushSpeed();
    };
  });
  showSpeed();

  // --- mode
  $$('[data-state]').forEach((button) => {
    button.onclick = async () => {
      releaseAll();
      const data = await api('/api/mode', { state: button.dataset.state });
      toast(`state: ${data.steps.join(' → ') || data.state}`, 'ok');
    };
  });
  $('#autoMode').onchange = async (event) => {
    // mode 0 = autonomous driving, 1 = remote driving (API doc p.226)
    await api('/api/auto_mode', { mode: event.target.checked ? 0 : 1 });
    toast(`driving mode: ${event.target.checked ? 'autonomous' : 'remote'}`, 'ok');
  };

  $('#preflightBtn').onclick = async () => {
    const data = await api('/api/preflight');
    $('#preflightOut').innerHTML = '<table style="margin-top:10px">' + data.checks.map((check) =>
      `<tr class="${check.ok ? '' : 'trip'}"><td>${check.ok ? '✓' : '✕'}</td>
       <td>${check.name}</td><td>${check.detail}</td></tr>`).join('') + '</table>';
  };

  // --- obstacles
  $('#protectionEnable').onclick = async () => {
    releaseAll();
    const result = await api('/api/protection/enable', {});
    toast(result.obstacles.protection.ready ? 'Sensor settings verified; protection ready' : 'Settings enabled; sensor readiness still needs attention', result.obstacles.protection.ready ? 'ok' : 'bad');
    await loadParams(true);
  };
  $('#sensorStale').onchange = async (event) => {
    await api('/api/settings', { sensor_stale_s: Number(event.target.value) }).catch(() => {});
  };
  $('#alarmRange').oninput = (event) => {
    $('#alarmVal').textContent = Number(event.target.value).toFixed(2);
  };
  $('#alarmRange').onchange = async (event) => {
    // The threshold lives on the server so the display and the server's own
    // view of "something is close" cannot drift apart.
    await api('/api/settings', { alarm_distance: Number(event.target.value) }).catch(() => {});
  };

  // --- navigation
  $('#pointsReload').onclick = () => loadPoints().catch(() => {});
  $('#pointRows').onclick = async (event) => {
    const name = event.target.dataset && event.target.dataset.goto;
    if (!name) return;
    releaseAll();
    await api('/api/nav/goto', { name });
    toast(`navigating to ${name}`, 'ok');
    pollPath();
  };
  $('#navStopBtn').onclick = $('#dockStop').onclick = async () => {
    await api('/api/nav/stop', {});
    pathPoints = [];
    toast('task stopped', 'ok');
  };
  $('#pointSave').onclick = async () => {
    const name = $('#pointName').value.trim();
    if (!name) { toast('give the point a name', 'bad'); return; }
    await api('/api/points/save', { name, type: $('#pointType').value });
    $('#pointName').value = '';
    toast(`saved ${name}`, 'ok');
    loadPoints().catch(() => {});
  };
  $('#locTurning').onchange = (event) => { $('#locTurnWarn').hidden = !event.target.checked; };
  $('#locBtn').onclick = async () => {
    const point = $('#locPoint').value;
    if (!point) { toast('pick a point the robot is standing on', 'bad'); return; }
    await api('/api/localize', { point, turning: $('#locTurning').checked });
    toast(`initialising at ${point}`, 'ok');
  };

  // --- charging
  $('#dockBtn').onclick = async () => {
    const point = $('#dockPoint').value;
    releaseAll();
    const data = await api('/api/charge/dock', point ? { point } : {});
    toast(`docking: navigating to ${data.navigating_to}`, 'ok');
    pollPath();
  };

  // --- setup
  $('#connectBtn').onclick = async () => {
    const host = $('#hostInput').value.trim();
    if (!host) { toast('enter the controller IP', 'bad'); return; }
    await api('/api/connect', { host, port: Number($('#portInput').value) || 1819 });
    mapLoadedFor = null;
    toast(`connected to ${host}`, 'ok');
    setTimeout(() => { loadParams(true).catch(() => {}); loadPoints().catch(() => {}); }, 700);
  };
  $('#disconnectBtn').onclick = async () => {
    releaseAll();
    await api('/api/disconnect', {});
    toast('disconnected — robot stopped', 'ok');
  };
  $('#discoverBtn').onclick = async () => {
    $('#discoverOut').textContent = 'scanning…';
    const data = await api('/api/discover');
    const where = `scanned ${data.scanned.map((s) => `${s}.0/24`).join(', ')}`;
    $('#discoverOut').textContent = data.hosts.length
      ? `found ${data.hosts.join(', ')} (${where})`
      : `nothing answering — ${where}. Are you on the robot's network?`;
    if (data.hosts.length) $('#hostInput').value = data.hosts[0];
  };
  // Label the scan button with the LAN this machine is actually on, not a guess.
  api('/api/subnets').then((data) => {
    $('#discoverBtn').textContent = `Scan ${data.subnets.map((s) => `${s}.0/24`).join(' + ')}`;
  }).catch(() => {});
  $('#paramsReload').onclick = () => loadParams(true).catch(() => {});
  $('#mapsReload').onclick = async () => {
    const data = await api('/api/maps');
    $('#mapRow').innerHTML = (data.maps || []).map((map) =>
      `<span class="chip ${map.mapname === data.current ? 'ok' : ''}">${map.mapname}
        (${map.gridWidth}×${map.gridHeight})</span>`).join('') || '<span class="hint">no maps</span>';
  };

  // --- map tools
  $('#mapFit').onclick = () => MapView.fit();
  $('#mapReload').onclick = () => { mapLoadedFor = null; renderMap(); };
  ['mapScan:scan', 'mapPath:path'].forEach((pair) => {
    const [id, key] = pair.split(':');
    $(`#${id}`).onclick = () => {
      const on = !$(`#${id}`).classList.contains('on');
      $(`#${id}`).classList.toggle('on', on);
      MapView.toggle(key, on);
      if (key === 'scan') {
        // Turning the laser off stops it crossing the link, not just the canvas
        // — it is the single largest thing on the wire.
        wantScan = on;
        connectStream();
      }
    };
  });

  $('#streamHz').onchange = (event) => {
    localStorage.setItem('dt01.hz', event.target.value);
    connectStream();
    toast(`live updates at ${event.target.value} Hz`, 'ok');
  };
}

/* The planned route has no push channel, so it is pulled while a task runs. */
async function pollPath() {
  for (let attempt = 0; attempt < 600; attempt++) {
    const nav = snapshot.nav || {};
    if (!nav.nav_status_type || nav.nav_status_type === 'idle') { pathPoints = []; break; }
    try {
      const data = await api('/api/nav/path');
      pathPoints = data.grid_points || [];
      if (data.mapInfo) MapView.setMapInfo(data.mapInfo);
    } catch (_) { /* the navigator refuses this outside RunningTask; keep the last plan */ }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

/* ----------------------------------------------------------------- boot */

window.addEventListener('DOMContentLoaded', () => {
  MapView.init($('#map'), (grid) => {
    $('#mapReadout').textContent =
      `grid ${grid.gridX}, ${grid.gridY}\nworld ${grid.worldX.toFixed(2)}, ${grid.worldY.toFixed(2)} m`;
  });
  wire();
  const savedRate = localStorage.getItem('dt01.hz');
  if (savedRate) $('#streamHz').value = savedRate;
  connectStream();
  api('/api/state').then((state) => {
    if (state.connected) {
      $('#hostInput').value = state.link.host;
      $('#portInput').value = state.link.port;
      loadParams().catch(() => {});
      loadPoints().catch(() => {});
    }
  }).catch(() => {});
});
