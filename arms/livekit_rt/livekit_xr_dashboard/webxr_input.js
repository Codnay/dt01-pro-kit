import { buttonNameForGamepad, emptyHand } from "./protocol.js";

const HANDS = ["left", "right"];
const AXIS_MAPS = {
  // WebXR reference space: +X right, +Y up, -Z forward.
  // ROS-ish teleop frame: +X forward, +Y left, +Z up.
  "ros-default": [
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
  ],
  // Alternate lateral sign if left/right motion feels mirrored.
  "ros-alt-y": [
    [0, 0, -1],
    [1, 0, 0],
    [0, 1, 0],
  ],
  identity: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
};

function mat3Mul(a, b) {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
    }
  }
  return out;
}

function mat3Transpose(a) {
  return [
    [a[0][0], a[1][0], a[2][0]],
    [a[0][1], a[1][1], a[2][1]],
    [a[0][2], a[1][2], a[2][2]],
  ];
}

function matrixToRowMajorArray(matrix, axisMapName) {
  const m = Array.from(matrix);
  const webxrRotation = [
    [m[0], m[4], m[8]],
    [m[1], m[5], m[9]],
    [m[2], m[6], m[10]],
  ];
  const p = [m[12], m[13], m[14]];
  const c = AXIS_MAPS[axisMapName] || AXIS_MAPS["ros-default"];
  const ct = mat3Transpose(c);
  const r = mat3Mul(mat3Mul(c, webxrRotation), ct);
  const t = [
    c[0][0] * p[0] + c[0][1] * p[1] + c[0][2] * p[2],
    c[1][0] * p[0] + c[1][1] * p[1] + c[1][2] * p[2],
    c[2][0] * p[0] + c[2][1] * p[1] + c[2][2] * p[2],
  ];
  return [
    r[0][0], r[0][1], r[0][2], t[0],
    r[1][0], r[1][1], r[1][2], t[1],
    r[2][0], r[2][1], r[2][2], t[2],
    m[3], m[7], m[11], m[15],
  ];
}

function buttonValue(gamepad, index) {
  const button = gamepad && gamepad.buttons ? gamepad.buttons[index] : null;
  return button ? Number(button.value || 0) : 0;
}

// True when an input source is hand tracking rather than a physical controller.
// When the controllers are set down, the Quest falls back to hand tracking,
// whose WebXR gamepad maps the PINCH gesture onto buttons[0] — the same index
// we read as the teleop "trigger". A resting/occluded hand throws spurious
// pinches (b0=1.0), which would phantom-activate teleop. Teleop is controller-
// only, so these sources are ignored.
function isHandInput(source) {
  if (source.hand) {
    return true;
  }
  const profiles = source.profiles || [];
  return profiles.some((p) => typeof p === "string" && p.includes("hand"));
}

function joystickY(gamepad) {
  if (!gamepad || !gamepad.axes || gamepad.axes.length < 4) {
    return 0;
  }
  return Number(gamepad.axes[3] || 0);
}

export class WebXrInputLoop {
  constructor({ canvas, log, onFrame, onButton, onInputSummary, getAxisMap, onSessionChange }) {
    this.canvas = canvas;
    this.log = log;
    this.onFrame = onFrame;
    this.onButton = onButton;
    this.onInputSummary = onInputSummary || (() => {});
    this.getAxisMap = getAxisMap || (() => "ros-default");
    this.onSessionChange = onSessionChange || (() => {});
    this.session = null;
    this.referenceSpace = null;
    this.gl = null;
    this.lastButtons = new Map();
    this.lastInputLog = 0;
    this.clearHue = 0;
    // Reported to the robot on the lease heartbeat so "xr=0" on the follower can
    // be attributed to the headset rather than to the transport.
    this.active = false;
    this.frames = 0;
    this.frameErrors = 0;
  }

  async isSupported() {
    return Boolean(navigator.xr && await navigator.xr.isSessionSupported("immersive-vr"));
  }

  // MUST be called synchronously from the operator's click. requestSession
  // needs transient user activation, and every `await` before it spends part of
  // that budget -- a support probe or makeXRCompatible() ahead of it is exactly
  // how "I pressed Enter VR and nothing happened" is produced. So the session is
  // requested FIRST and the GL context is prepared afterwards, which is also the
  // order the WebXR samples use.
  async start() {
    if (!navigator.xr) {
      throw new Error("navigator.xr is not available; open this page over HTTPS in the Meta Quest Browser");
    }

    this.session = await navigator.xr.requestSession("immersive-vr", {
      optionalFeatures: ["local-floor", "bounded-floor", "hand-tracking"],
    });

    this.gl = this.canvas.getContext("webgl", { xrCompatible: true });
    if (!this.gl) {
      await this.session.end();
      this.session = null;
      throw new Error("WebGL is required for immersive WebXR");
    }
    if (typeof this.gl.makeXRCompatible === "function") {
      await this.gl.makeXRCompatible();
    }
    this.session.updateRenderState({
      baseLayer: new XRWebGLLayer(this.session, this.gl),
    });

    try {
      this.referenceSpace = await this.session.requestReferenceSpace("local-floor");
    } catch {
      this.referenceSpace = await this.session.requestReferenceSpace("local");
    }

    this.session.addEventListener("end", () => {
      this.session = null;
      this.referenceSpace = null;
      this.active = false;
      this.log("XR session ended — pose publishing has stopped");
      this.onSessionChange(false);
    });

    this.active = true;
    this.onSessionChange(true);
    this.session.requestAnimationFrame((time, frame) => this._onXrFrame(time, frame));
    this.log("XR session started");
  }

  // Snapshot for the lease heartbeat: whether the immersive session is live and
  // how many frames it has handed to the publisher.
  status() {
    return { active: this.active, frames: this.frames, errors: this.frameErrors };
  }

  async stop() {
    if (this.session) {
      await this.session.end();
    }
  }

  // The rAF chain is self-rescheduling, so ANY throw in here used to end teleop
  // permanently and silently: the lease heartbeat is a separate window timer, so
  // the operator kept holding control while sending nothing, and the robot saw
  // an operator standing perfectly still forever. One bad frame must cost one
  // frame, so the reschedule lives in a finally.
  _onXrFrame(time, frame) {
    const session = frame.session;
    try {
      this._readAndPublish(frame, session);
    } catch (exc) {
      this.frameErrors += 1;
      if (this.frameErrors % 100 === 1) {
        this.log(`XR frame error (${this.frameErrors} total): ${exc.message || exc}`);
      }
    } finally {
      if (this.session) {
        session.requestAnimationFrame((nextTime, nextFrame) => this._onXrFrame(nextTime, nextFrame));
      }
    }
  }

  _readAndPublish(frame, session) {
    this._renderXrFrame(frame);
    const perHand = {
      left: emptyHand(),
      right: emptyHand(),
    };

    for (const source of session.inputSources) {
      const hand = source.handedness;
      if (!HANDS.includes(hand)) {
        continue;
      }

      // Skip hand-tracking sources so a pinch gesture can't be read as the
      // teleop trigger (see isHandInput). Leaves this hand as emptyHand() →
      // valid:false → no teleop, which is correct when controllers are down.
      if (isHandInput(source)) {
        continue;
      }

      const space = source.gripSpace || source.targetRaySpace;
      const pose = space ? frame.getPose(space, this.referenceSpace) : null;
      const gamepad = source.gamepad;

      perHand[hand] = {
        transform: pose ? matrixToRowMajorArray(pose.transform.matrix, this.getAxisMap()) : [],
        grip: buttonValue(gamepad, 1),
        trigger: buttonValue(gamepad, 0),
        joystick_y: joystickY(gamepad),
        valid: Boolean(pose),
      };

      this._detectButtonPresses(hand, gamepad);
    }

    this._reportInputs(session.inputSources);
    this.frames += 1;
    this.onFrame(perHand);
  }

  _renderXrFrame(frame) {
    if (!this.gl || !this.session || !this.referenceSpace) {
      return;
    }
    const baseLayer = this.session.renderState.baseLayer;
    if (!baseLayer) {
      return;
    }
    const pose = frame.getViewerPose(this.referenceSpace);
    if (!pose) {
      return;
    }

    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, baseLayer.framebuffer);
    this.clearHue = (this.clearHue + 0.002) % 1;
    const pulse = 0.18 + 0.06 * Math.sin(this.clearHue * Math.PI * 2);
    gl.clearColor(0.03, pulse, 0.16, 1.0);
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);

    for (const view of pose.views) {
      const viewport = baseLayer.getViewport(view);
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    }
  }

  // ALL of this is diagnostics, and it used to run on EVERY XR frame: a
  // per-button toFixed() string for both controllers, a profiles join, and a DOM
  // write, at 72-90 Hz, inside the frame callback. Dropped XR frames are dropped
  // poses, so it is now gated on the same 2 s window the logging was already
  // throttled to -- the strings are simply never built in between.
  _reportInputs(inputSources) {
    const now = performance.now();
    if (now - this.lastInputLog < 2000) {
      return;
    }
    this.lastInputLog = now;

    const parts = [];
    const rawParts = [];
    for (const source of inputSources) {
      const gamepad = source.gamepad;
      const hand = source.handedness || "none";
      parts.push(`${hand}:${source.gripSpace ? "grip" : "nogrip"}:${gamepad ? `${gamepad.buttons.length}b/${gamepad.axes.length}a` : "nogp"}`);
      if (gamepad && gamepad.buttons) {
        // Raw per-button dump so a stuck/miscalibrated trigger (or a wrong
        // button index for one controller) is directly visible. trigger is
        // read from b0, grip from b1 (see _onXrFrame). At rest every value
        // should be ~0.00; a value pinned near 1.00 is the culprit.
        const btns = Array.from(gamepad.buttons)
          .map((b, i) => `b${i}=${Number((b && b.value) || 0).toFixed(2)}`)
          .join(" ");
        const axes = gamepad.axes
          ? Array.from(gamepad.axes).map((a) => Number(a || 0).toFixed(2)).join(",")
          : "";
        const profiles = Array.isArray(source.profiles) ? source.profiles.join("/") : "";
        rawParts.push(`${hand} [${profiles}]: ${btns} | ax=[${axes}]`);
      }
    }

    const summary = parts.length ? parts.join(" ") : "0";
    this.onInputSummary(summary);
    this.log(`XR inputs: ${summary}`);
    for (const rp of rawParts) {
      this.log(`raw ${rp}`);
      if (typeof console !== "undefined" && console.log) {
        console.log(`[webxr-raw] ${rp}`);
      }
    }
  }

  _detectButtonPresses(hand, gamepad) {
    if (!gamepad || !gamepad.buttons) {
      return;
    }

    for (let i = 0; i < gamepad.buttons.length; i += 1) {
      const name = buttonNameForGamepad(hand, i);
      if (!name) {
        continue;
      }
      const key = `${hand}:${i}`;
      const pressed = Boolean(gamepad.buttons[i].pressed || gamepad.buttons[i].value > 0.7);
      const wasPressed = Boolean(this.lastButtons.get(key));
      this.lastButtons.set(key, pressed);
      if (pressed && !wasPressed) {
        this.onButton(name);
      }
    }
  }
}
