// Wire format for the LiveKit XR control path.
//
// Byte-identical to the Agora leader's piper_protocol.js on purpose: the robot
// decodes both transports with the SAME Python functions
// (xr_control.decode_xr_frame / decode_button / decode_control), so an
// A/B between Agora and LiveKit measures the transports and not two encoders
// that happen to disagree. Do not "improve" this format on one side only.

const HANDS = ["left", "right"];

// Monotonic per-tick frame counter. Both hands in one tick share the same seq,
// so the follower can spot frames lost in transit as gaps — operator-side
// telemetry the robot otherwise has no way to observe.
let frameSeq = 0;
let lastSeq = -1;

export function lastFrameSeq() {
  return lastSeq;
}

export function emptyHand() {
  return {
    transform: [],
    grip: 0.0,
    trigger: 0.0,
    joystick_y: 0.0,
    stamp: Date.now() / 1000,
    valid: false,
  };
}

// `control` = { cid, ep }: the operator's take-control identity. The follower
// only lets frames from the current lease holder reach the robot, so every frame
// carries who sent it (cid) and the epoch they believe they hold.
export function encodeXrFrame(perHand, control = {}) {
  const now = Date.now() / 1000;
  const seq = frameSeq++;
  lastSeq = seq;
  const payload = {};
  for (const hand of HANDS) {
    const src = perHand[hand] || emptyHand();
    const transform =
      Array.isArray(src.transform) && src.transform.length === 16
        ? src.transform.map((v) => Number(v))
        : [];
    payload[hand] = {
      transform,
      grip: Number(src.grip || 0),
      trigger: Number(src.trigger || 0),
      joystick_y: Number(src.joystick_y || 0),
      stamp: now,
      seq,
      valid: transform.length === 16,
    };
  }
  payload.cid = String(control.cid || "");
  payload.ep = Number(control.ep || 0);
  return JSON.stringify(payload);
}

// Round-trip probe. The id is opaque to the robot, which echoes it untouched —
// that is what makes the measurement clock-free: the browser subtracts its OWN
// performance.now() readings, so skew against the robot's clock cancels instead
// of being mistaken for latency. The old `stamp`-minus-receive-time approach
// measured skew and reported it as latency, which is why it was usually
// NEGATIVE.
export function encodePing(pid, control = {}, name = "") {
  return JSON.stringify({
    kind: "ping",
    cid: String(control.cid || ""),
    ep: Number(control.ep || 0),
    name: String(name || ""),
    pid: String(pid),
  });
}

export function encodeButton(name, control = {}) {
  return JSON.stringify({
    name: String(name),
    stamp: Date.now() / 1000,
    cid: String(control.cid || ""),
    ep: Number(control.ep || 0),
  });
}

// `diag` is an OPTIONAL, additive extension carried only on the ctl topic:
//   vr  1 when an immersive session is live, 0 otherwise
//   xr  frames this page has published
//   err publish failures
// Both followers decode ctl with xr_control.decode_control, which
// keeps only {kind, cid, ep, name} and ignores anything else — so this stays
// wire-compatible with the Agora leader in both directions. It exists because
// the pose topic is lossy and one-way: without it a follower seeing xr=0 cannot
// tell "the headset never published" from "the link ate every packet".
export function encodeControl(kind, control = {}, name = "", diag = null) {
  const msg = {
    kind: String(kind),
    cid: String(control.cid || ""),
    ep: Number(control.ep || 0),
    name: String(name || ""),
    stamp: Date.now() / 1000,
  };
  if (diag) {
    msg.vr = diag.vr ? 1 : 0;
    msg.xr = Number(diag.xr || 0);
    msg.err = Number(diag.err || 0);
    // Measured by THIS page against its own monotonic clock; the robot only
    // carries it to Prometheus. Omitted until a pong has actually come back, so
    // a missing value reads as "not measured yet" rather than as zero latency.
    if (diag.rtt != null) msg.rtt = Number(diag.rtt);
  }
  return JSON.stringify(msg);
}

export function decodeControl(text) {
  try {
    const obj = JSON.parse(text);
    if (!obj || typeof obj !== "object") return null;
    return {
      kind: String(obj.kind || ""),
      cid: String(obj.cid || ""),
      ep: Number(obj.ep || 0),
      name: String(obj.name || ""),
      pid: String(obj.pid || ""),
      resume_cid: String(obj.resume_cid || ""),
    };
  } catch (_exc) {
    return null;
  }
}

export function buttonNameForGamepad(handedness, buttonIndex) {
  if (handedness === "right") {
    if (buttonIndex === 4) return "a";
    if (buttonIndex === 5) return "b";
    if (buttonIndex === 3) return "rj";
  }
  if (handedness === "left") {
    if (buttonIndex === 4) return "x";
    if (buttonIndex === 5) return "y";
  }
  return "";
}

export { HANDS };
