// LiveKit XR operator dashboard.
//
// Flow: fetch a per-session token -> connect to LiveKit -> claim control ->
// enter WebXR -> publish controller pose at a fixed rate until the operator
// releases or the session ends.

import { LiveKitPublisher, ensureLiveKitSdk } from "./livekit_transport.js";
import {
  decodeControl,
  encodeButton,
  encodeControl,
  encodePing,
  encodeXrFrame,
  lastFrameSeq,
} from "./protocol.js";
import { WebXrInputLoop } from "./webxr_input.js";

// FLOOR, not a target. The only clock in this loop is the headset's rAF, so the
// achievable rates are exactly rAF/1, rAF/2, rAF/3 ... and 50 is not among them
// for any Quest. We therefore publish every k-th frame with the LARGEST k whose
// rate still clears this floor, which is uniform by construction: 72Hz->72,
// 90Hz->90, 120Hz->60. Uniform spacing is the point — jitter in the pose stream
// reaches the arm as jitter, and a 1 euro filter fed uneven dt estimates speed
// badly exactly when the operator is moving fast.
//
// The previous `if (now - last < 20ms) return; last = now;` gate looked like a
// 50 Hz throttle but was an integer divide of the refresh rate, landing on
// 36 Hz (72Hz headset) or 45 Hz (90Hz headset) and never on 50.
const PUBLISH_HZ = 50;
const PING_MS = 500; // round-trip probe; see encodePing
const HEARTBEAT_MS = 1000; // lease keepalive; follower expires a silent holder
const LOG_LINES = 200;
const REMOTE_LOG_MS = 1000; // batch interval for shipping the log to the server
// A claim is not believed until the robot echoes it back as the holder. The
// robot announces within one 250 ms room tick, so this only fires when the claim
// genuinely lost or the packet was dropped.
const CLAIM_TIMEOUT_MS = 700;
const MAX_CLAIM_ATTEMPTS = 4;

const el = (id) => document.getElementById(id);
const state = {
  config: null,
  publisher: null,
  xr: null,
  control: { cid: "", ep: 0 },
  holding: false,
  wantsControl: false,
  lastRobotStateAt: 0,
  maxEpochSeen: 0,
  // Authoritative, from the robot's `state` broadcast. `floorEpoch` is the
  // number a claim must strictly beat; the browser cannot derive it, because its
  // own counter restarts at 0 on every page load while the robot keeps the floor
  // across a release. Claiming without it is why Take Control used to need two
  // clicks after a clean disconnect.
  floorEpoch: -1,
  holderCid: "",
  holderName: "",
  robotStateSeen: false,
  claiming: false,
  claimAttempts: 0,
  claimDeadline: 0,
  operatorName: "",
  robot: "",
  // Publish pacing: measured rAF period -> uniform every-k-th-frame decimation.
  framePeriodMs: 0,
  lastFrameAt: 0,
  tick: 0,
  publishInFlight: false,
  publishSkipped: 0,
  hands: { left: false, right: false },
  published: 0,
  publishErrors: 0,
  // Round-trip measurement (see encodePing). pings maps id -> send time on THIS
  // page's monotonic clock; nothing here ever reads a robot timestamp.
  pings: new Map(),
  pingSeq: 0,
  rttMs: null,
  lastRateAt: 0,
  lastRateCount: 0,
};

// -- Logging -----------------------------------------------------------------

const logLines = [];
// Lines waiting to be shipped to the server terminal. Batched on a timer rather
// than one POST per line: the log is chatty during an XR session and a fetch per
// line would compete with pose publishing for the same connection.
const pendingRemote = [];

function log(message) {
  const stamp = new Date().toISOString().slice(11, 23);
  const line = `${stamp}  ${message}`;
  logLines.push(line);
  if (logLines.length > LOG_LINES) logLines.shift();
  const pane = el("log");
  if (pane) {
    pane.textContent = logLines.join("\n");
    pane.scrollTop = pane.scrollHeight;
  }
  // Mirrored to the devtools console so the Quest browser's remote inspector
  // shows the same trace the on-screen pane does.
  if (typeof console !== "undefined") console.log(`[xr] ${message}`);
  // ...and to the server's terminal. Once the operator is inside the headset
  // this pane is unreadable, so the person running the server is the only one
  // who can see why "Enter VR" failed. Capped so a runaway logger cannot grow
  // without bound while the network is down.
  if (pendingRemote.length < 500) pendingRemote.push(line);
}

// Best-effort: a dashboard that cannot reach its own log endpoint must still fly
// the arm, so failures here are swallowed rather than surfaced.
function flushRemoteLog() {
  if (!pendingRemote.length) return;
  const lines = pendingRemote.splice(0, pendingRemote.length);
  const body = JSON.stringify({
    identity: (state.config && state.config.identity) || "",
    name: state.operatorName,
    lines,
  });
  fetch("/api/log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    keepalive: true,
  }).catch(() => {});
}

function setStatus(id, text, tone = "") {
  const node = el(id);
  if (!node) return;
  node.textContent = text;
  node.dataset.tone = tone;
}

// -- Robot selection ---------------------------------------------------------

// One server serves every deployment, so the page has to say which arm it will
// move -- before Connect, not after something twitches.
function setRobotBadge(name) {
  state.robot = name || "";
  const badge = el("robot-badge");
  if (badge) badge.textContent = state.robot || "—";
  if (state.robot) document.title = `${state.robot} — PiPER XR`;
}

async function loadRobots() {
  const sel = el("robot");
  if (!sel) return;
  // ?robot=dt01-pro-02 preselects, so each deployment can be a bookmark.
  const wanted = new URLSearchParams(window.location.search).get("robot") || "";
  try {
    const res = await fetch("/api/robots");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { robots, default: fallback } = await res.json();
    if (!Array.isArray(robots) || !robots.length) throw new Error("empty list");
    sel.innerHTML = "";
    for (const name of robots) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
    if (wanted && !robots.includes(wanted)) {
      log(`this server does not serve '${wanted}' — using ${fallback || robots[0]}`);
    }
    sel.value = robots.includes(wanted) ? wanted : fallback || robots[0];
  } catch (exc) {
    // An older server has no /api/robots. Keep the page usable: one entry, and
    // connect() omits ?robot so the server falls back to its configured room.
    log(`robot list unavailable (${exc.message || exc}); using this server's default`);
    sel.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = wanted || "(server default)";
    sel.appendChild(opt);
  }
  setRobotBadge(sel.value || sel.options[0].textContent);
}

// -- Take-control ------------------------------------------------------------

// Paint the one line the operator actually reads. Plain sentences on purpose:
// the person wearing the headset is not debugging a lease protocol, they want to
// know whether the arms will move when they squeeze the trigger.
function renderControl() {
  const take = el("btn-take");
  const release = el("btn-release");
  if (state.publisher && (!state.lastRobotStateAt || performance.now()-state.lastRobotStateAt > 3000)) {
    setStatus("status-control", "Waiting for robot connection…", "warn");
  } else if (state.claiming) {
    setStatus("status-control", "Requesting control…", "warn");
  } else if (state.holding) {
    setStatus(
      "status-control",
      state.robot ? `Control of ${state.robot} — release, then squeeze to move` : "Release, then squeeze to move",
      "good"
    );
  } else if (state.holderCid) {
    setStatus("status-control", `${state.holderName} has control`, "warn");
  } else {
    setStatus("status-control", "Nobody has control", "");
  }
  if (take) {
    take.disabled = !state.publisher || state.claiming;
    take.textContent = state.claiming ? "Requesting…" : "Take control";
  }
  if (release) release.disabled = !state.wantsControl && !state.holding;
}

function sendClaim() {
  // Beat the robot's floor, not our own page-local counter. `floorEpoch` starts
  // at -1 and is replaced by the robot's first announce, so the very first click
  // of a fresh page already carries a winning number.
  const floor = Math.max(state.maxEpochSeen, state.floorEpoch);
  const ep = floor + 1;
  state.control = { cid: state.config.cid, ep };
  state.maxEpochSeen = ep;
  state.claimAttempts += 1;
  state.claimDeadline = performance.now() + CLAIM_TIMEOUT_MS;
  state.publisher.publishControl(
    encodeControl("claim", state.control, state.operatorName)
  );
  log(`asking for control (request #${ep}, attempt ${state.claimAttempts})`);
}

function claimControl() {
  if (!state.publisher || state.claiming) return;
  state.wantsControl = true;
  state.claiming = true;
  state.claimAttempts = 0;
  sendClaim();
  renderControl();
}

// Retry ONLY while the outcome is unknown. A claim that lost to a real person is
// not retried -- two browsers auto-escalating epochs at each other would take the
// arms back and forth with nobody touching a button.
function tickClaim() {
  if (state.lastRobotStateAt && performance.now()-state.lastRobotStateAt > 3000) {
    state.holding = false;
    renderControl();
  }
  if (!state.claiming) return;
  if (performance.now() < state.claimDeadline) return;
  if (state.claimAttempts >= MAX_CLAIM_ATTEMPTS) {
    state.claiming = false;
    log("could not take control after several tries — press Take control again");
    setStatus("status-control", "Could not take control — press it again", "bad");
    flushRemoteLog();
    return;
  }
  sendClaim();
}

function releaseControl() {
  state.wantsControl = false;
  state.claiming = false;
  if (!state.holding) {
    renderControl();
    return;
  }
  state.publisher.publishControl(
    encodeControl("release", state.control, state.operatorName)
  );
  state.holding = false;
  log("gave up control");
  renderControl();
}

// The robot's authoritative view: who holds the arms, and the number a claim
// must beat. This is the only message that may set `holding` true.
function onRobotState(msg) {
  if (msg.ep < state.floorEpoch) return; // delayed state cannot undo a newer lease
  state.lastRobotStateAt = performance.now();
  const first = !state.robotStateSeen;
  state.robotStateSeen = true;
  state.floorEpoch = msg.ep;
  if (msg.ep > state.maxEpochSeen) state.maxEpochSeen = msg.ep;
  state.holderCid = msg.cid || "";
  state.holderName = msg.name || msg.cid || "";
  if (first) log(`robot is reachable; next request will be #${msg.ep + 1}`);

  const weHold = state.wantsControl && state.holderCid === state.config.cid && msg.ep === state.control.ep;
  if (weHold) {
    if (state.claiming || !state.holding) log("robot confirmed: you have control");
    state.claiming = false;
    state.holding = true;
  } else {
    if (state.holding) {
      log(
        state.holderCid
          ? `${state.holderName} took control from you`
          : "you no longer have control"
      );
    }
    state.holding = false;
    // Someone else genuinely holds it: stop retrying and let the operator decide
    // whether to take it from them with another press.
    if (state.claiming && state.holderCid && msg.ep >= state.control.ep) {
      state.claiming = false;
      state.wantsControl = false;
      log(`${state.holderName} has control — press Take control again to take over`);
    }
  }
  if (state.holderCid && state.holderCid !== state.config.cid &&
      (!state.claiming || msg.ep >= state.control.ep)) {
    state.wantsControl = false;
  }
  if (!state.holderCid && (msg.resume_cid === state.config.cid || first) && state.wantsControl && !state.claiming) {
    claimControl(); // only restore the same operator's still-requested session
  }
  renderControl();
}

// Round trip closed. Both readings come off performance.now() on this page, so
// the robot's clock never enters the arithmetic and the result cannot be
// negative — the failure that made the old stamp-based latency unusable.
function onPong(pid) {
  const sent = state.pings.get(pid);
  if (sent === undefined) return; // already pruned, or not ours
  state.pings.delete(pid);
  const rtt = performance.now() - sent;
  // Light smoothing: one retransmit-free lossy link still varies frame to frame,
  // and the operator wants the trend, not the instantaneous sample.
  state.rttMs = state.rttMs == null ? rtt : state.rttMs * 0.7 + rtt * 0.3;
}

function sendPing() {
  if (!state.publisher) return;
  const pid = String(++state.pingSeq);
  const now = performance.now();
  state.pings.set(pid, now);
  // Drop probes that were never answered, so a link outage cannot grow this map
  // without bound and a late pong cannot report an absurd RTT.
  for (const [id, t] of state.pings) {
    if (now - t > 10000) state.pings.delete(id);
  }
  state.publisher.publishControl(encodePing(pid, state.control, state.operatorName));
}

function onControlMessage(text) {
  const msg = decodeControl(text);
  if (!msg) return;
  if (msg.kind === "pong") {
    onPong(msg.pid);
    return;
  }
  if (msg.kind === "ping") return; // another operator's probe; the robot answers it
  if (msg.kind === "state") {
    onRobotState(msg); // authoritative; must be handled before the echo check
    return;
  }
  if (msg.ep > state.maxEpochSeen) state.maxEpochSeen = msg.ep;
  if (msg.cid === state.config.cid) return; // our own echo

  // Peer-to-peer preemption notice. Kept as a fast path so the outgoing operator
  // sees the handover immediately rather than waiting for the robot's announce,
  // but the robot's `state` message remains the authority.
  const preempted =
    msg.kind !== "release" &&
    state.holding &&
    (msg.ep > state.control.ep ||
      (msg.ep === state.control.ep && msg.cid > state.control.cid));
  if (preempted) {
    state.wantsControl = false;
    state.claiming = false;
    state.holding = false;
    state.holderCid = msg.cid;
    state.holderName = msg.name || msg.cid;
    log(`${state.holderName} took control from you`);
    renderControl();
  }
}

// -- XR frame publishing -----------------------------------------------------

function onXrFrame(perHand) {
  const now = performance.now();

  // Track the headset's real frame period. Smoothed, and outliers (a compositor
  // hiccup, the first frame) are ignored so one bad frame cannot change k.
  if (state.lastFrameAt) {
    const p = now - state.lastFrameAt;
    if (p > 2 && p < 60) {
      state.framePeriodMs = state.framePeriodMs
        ? state.framePeriodMs * 0.9 + p * 0.1
        : p;
    }
  }
  state.lastFrameAt = now;

  // Publish every k-th frame: uniform spacing, and never slower than PUBLISH_HZ.
  const rafHz = state.framePeriodMs ? 1000 / state.framePeriodMs : PUBLISH_HZ;
  const k = Math.max(1, Math.floor(rafHz / PUBLISH_HZ));
  state.tick += 1;
  if (state.tick < k) return;
  state.tick = 0;

  // Recorded regardless of whether we hold control, so the tracking readout is
  // honest before Take Control. Painted by tickStats at 1 Hz rather than here:
  // a DOM write inside the XR frame callback costs frames, and dropped frames
  // are dropped poses.
  state.hands.left = Boolean(perHand.left && perHand.left.valid);
  state.hands.right = Boolean(perHand.right && perHand.right.valid);

  if (!state.holding || !state.publisher) return;

  // NEWEST WINS. publishData resolves when the packet is handed to the
  // transport, so on a congested long-haul link an unguarded loop queues poses
  // the operator has already moved past — head-of-line delay that arrives at the
  // arm as lag and then as a catch-up lurch. Skipping while one is outstanding
  // means the robot always gets the freshest pose we had, and the lossy topic
  // never retransmits a stale one.
  if (state.publishInFlight) {
    state.publishSkipped += 1;
    return;
  }
  state.publishInFlight = true;

  const text = encodeXrFrame(perHand, state.control);
  state.publisher
    .publishXr(text)
    .then(() => {
      state.published += 1;
    })
    .catch((exc) => {
      state.publishErrors += 1;
      if (state.publishErrors % 50 === 1) {
        log(`publish failed (${state.publishErrors} total): ${exc.message || exc}`);
      }
    })
    .finally(() => {
      state.publishInFlight = false;
    });
}

// Health this page reports to the robot on every heartbeat. The heartbeat is
// reliable and the pose topic is lossy, so this is the one signal guaranteed to
// arrive when pose does not — it turns the follower's bare "xr=0" into "the
// operator is not in VR" or "the operator IS publishing, so the link is losing
// it", which are opposite problems with opposite fixes.
function leaderDiag() {
  const xr = state.xr ? state.xr.status() : { active: false, frames: 0 };
  return {
    vr: xr.active,
    xr: state.published,
    err: state.publishErrors,
    // null until the first pong, so the robot reports "not measured" rather
    // than a confident zero.
    rtt: state.rttMs == null ? null : Math.round(state.rttMs),
  };
}

function onButton(name) {
  if (!state.holding || !state.publisher) return;
  state.publisher.publishButton(encodeButton(name, state.control));
  log(`button ${name}`);
}

// -- Stats -------------------------------------------------------------------

function tickStats() {
  const now = performance.now();
  if (state.lastRateAt === 0) {
    state.lastRateAt = now;
    state.lastRateCount = state.published;
    return;
  }
  const dt = (now - state.lastRateAt) / 1000;
  if (dt < 1) return;
  const rate = (state.published - state.lastRateCount) / dt;
  state.lastRateAt = now;
  state.lastRateCount = state.published;
  const rtt = state.rttMs == null ? "—" : `${Math.round(state.rttMs)}ms`;
  setStatus(
    "status-rate",
    `${rate.toFixed(1)} fps · rtt ${rtt} · seq ${lastFrameSeq()} · ` +
      `err ${state.publishErrors} · skipped ${state.publishSkipped}`,
    // Judged against the floor the publisher actually guarantees. The old
    // threshold was 0.6x, wide enough to paint the 36 Hz bug green.
    rate >= PUBLISH_HZ * 0.95 ? "good" : state.holding ? "warn" : ""
  );
  setStatus(
    "status-hands",
    `L ${state.hands.left ? "tracked" : "—"} · R ${state.hands.right ? "tracked" : "—"}`,
    state.hands.left || state.hands.right ? "good" : "warn"
  );
}

// -- Wiring ------------------------------------------------------------------

async function connect() {
  el("btn-connect").disabled = true;
  try {
    const picked = (el("robot") && el("robot").value) || "";
    log(picked ? `connecting to ${picked}…` : "fetching session config…");
    const res = await fetch(
      "/api/config" + (picked ? `?robot=${encodeURIComponent(picked)}` : "")
    );
    const cfg = await res.json();
    if (cfg.error) throw new Error(cfg.error);
    state.config = cfg;
    state.operatorName = el("operator-name").value.trim() || cfg.identity;
    // The server is the authority on which room the token is actually for.
    setRobotBadge(cfg.robot || cfg.room);
    // Switching robots mid-session would leave the old room holding our lease,
    // so the picker locks until the page is reloaded.
    if (el("robot")) el("robot").disabled = true;
    log(`robot=${cfg.room} identity=${cfg.identity} cid=${cfg.cid}`);

    await ensureLiveKitSdk(log);
    const publisher = new LiveKitPublisher(cfg, log);
    publisher.onControl = onControlMessage;
    publisher.onStateChange = (s) =>
      setStatus("status-link", `Browser: ${s}`, s === "connected" ? "good" : "warn");
    state.publisher = publisher;
    await publisher.connect();

    el("btn-xr").disabled = false;
    renderControl(); // enables Take control; Release stays off until we hold it
  } catch (exc) {
    log(`connect failed: ${exc.message || exc}`);
    setStatus("status-link", "failed", "bad");
    el("btn-connect").disabled = false;
    if (el("robot")) el("robot").disabled = false;
  }
}

// NOTE: nothing is awaited before xr.start(). requestSession() needs transient
// user activation from this click, and the old support probe sat in front of it
// burning that activation — the failure mode being a rejected session, logged to
// a pane the operator cannot read once the headset is on, while the page happily
// went on heartbeating and holding the lease. Support is probed at page load
// instead (see preflightXr).
async function enterXr() {
  try {
    const xr = new WebXrInputLoop({
      canvas: el("xr-canvas"),
      log,
      onFrame: onXrFrame,
      onButton,
      onInputSummary: (s) => setStatus("status-inputs", s),
      getAxisMap: () => el("axis-map").value,
      onSessionChange: (active) =>
        setStatus(
          "status-xr",
          active ? "in VR · publishing" : "ended — press Enter VR again",
          active ? "good" : "bad"
        ),
    });
    state.xr = xr;
    if (!state.holding) {
      log("note: you do not have control yet — press Take control or the arms will not move");
    }
    setStatus("status-xr", "starting…", "warn");
    await xr.start();
  } catch (exc) {
    // Loud on purpose. This is the single most common reason the robot sees a
    // holder that never sends pose, and until now it was one grey log line.
    const detail = exc.message || exc;
    setStatus("status-xr", `FAILED: ${detail}`, "bad");
    log(`XR start FAILED: ${detail}`);
    log("the robot will see you holding control but sending no pose — press Enter VR again");
    flushRemoteLog();
  }
}

async function preflightXr() {
  try {
    if (!navigator.xr) {
      setStatus("status-xr", "no WebXR (needs HTTPS + Quest browser)", "bad");
      log("navigator.xr missing — this page must be served over HTTPS and opened in the Quest browser");
      return;
    }
    const ok = await navigator.xr.isSessionSupported("immersive-vr");
    setStatus("status-xr", ok ? "ready" : "immersive-vr unsupported", ok ? "" : "bad");
    if (!ok) log("immersive-vr is not supported by this browser");
  } catch (exc) {
    setStatus("status-xr", `probe failed: ${exc.message || exc}`, "bad");
  }
}

window.addEventListener("DOMContentLoaded", () => {
  el("btn-connect").addEventListener("click", connect);
  el("btn-take").addEventListener("click", claimControl);
  el("btn-release").addEventListener("click", releaseControl);
  el("btn-xr").addEventListener("click", enterXr);

  window.setInterval(() => {
    if (state.holding && state.publisher) {
      state.publisher.publishControl(
        encodeControl("heartbeat", state.control, state.operatorName, leaderDiag())
      );
    }
  }, HEARTBEAT_MS);
  window.setInterval(sendPing, PING_MS);
  window.setInterval(tickStats, 1000);
  window.setInterval(tickClaim, 150);
  window.setInterval(flushRemoteLog, REMOTE_LOG_MS);

  // Releasing on unload means closing the tab frees control immediately instead
  // of making the next operator wait out the follower's liveness timeout.
  window.addEventListener("beforeunload", () => {
    releaseControl();
    flushRemoteLog();
  });

  if (el("robot")) {
    el("robot").addEventListener("change", (e) => setRobotBadge(e.target.value));
  }
  loadRobots();
  preflightXr();
  renderControl();
  log("ready — press Connect, then Take control, then Enter VR");
});
