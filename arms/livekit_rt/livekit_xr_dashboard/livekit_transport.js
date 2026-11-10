// LiveKit transport for the XR leader page.
//
// Replaces agora_client.js. The whole file is ~150 lines against Agora's ~250
// because there is one connection instead of three channels, and LiveKit's own
// reconnect handling replaces the manual watchdog.
//
// The SDK is loaded from a vendored copy when present, else from a CDN. Vendor
// it for anything you actually rely on — a CDN fetch is one more thing between
// the operator and the arm, and it fails at exactly the wrong moment.

const CDN_URL = "https://cdn.jsdelivr.net/npm/livekit-client@2/dist/livekit-client.umd.min.js";
const VENDOR_URL = "/static/vendor/livekit-client.umd.min.js";

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = src;
    el.onload = () => resolve(src);
    el.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(el);
  });
}

export async function ensureLiveKitSdk(log) {
  if (window.LivekitClient) return "already-loaded";
  try {
    await loadScript(VENDOR_URL);
    log("LiveKit SDK loaded from vendored copy");
    return "vendor";
  } catch (_exc) {
    log("no vendored SDK; falling back to CDN");
  }
  await loadScript(CDN_URL);
  log("LiveKit SDK loaded from CDN");
  return "cdn";
}

export class LiveKitPublisher {
  constructor(config, log) {
    this.config = config; // { url, token, room, identity, cid }
    this.log = log;
    this.room = null;
    this.connectionState = "idle";
    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder();
    // Set by app.js to receive take-control lease messages from OTHER operators.
    this.onControl = null;
    this.onStateChange = null;
  }

  _setState(state) {
    this.connectionState = state;
    if (typeof this.onStateChange === "function") {
      try {
        this.onStateChange(state);
      } catch (exc) {
        this.log(`state handler error: ${exc.message || exc}`);
      }
    }
  }

  async connect() {
    const LK = window.LivekitClient;
    if (!LK) throw new Error("LiveKit SDK not loaded");

    // No media is published or subscribed: this page sends control data only,
    // and camera video reaches the operator over Agora RTC as before.
    const room = new LK.Room({ adaptiveStream: false, dynacast: false });
    this.room = room;

    room.on(LK.RoomEvent.DataReceived, (payload, _participant, _kind, topic) => {
      if (topic !== "ctl") return;
      let text = "";
      try {
        text = this.decoder.decode(payload);
      } catch (_exc) {
        return;
      }
      if (typeof this.onControl === "function") {
        try {
          this.onControl(text);
        } catch (exc) {
          this.log(`onControl handler error: ${exc.message || exc}`);
        }
      }
    });

    room.on(LK.RoomEvent.Connected, () => this._setState("connected"));
    room.on(LK.RoomEvent.Reconnecting, () => this._setState("reconnecting"));
    room.on(LK.RoomEvent.Reconnected, () => this._setState("connected"));
    room.on(LK.RoomEvent.Disconnected, () => this._setState("disconnected"));
    room.on(LK.RoomEvent.ParticipantConnected, (p) =>
      this.log(`participant joined: ${p.identity}`)
    );
    room.on(LK.RoomEvent.ParticipantDisconnected, (p) =>
      this.log(`participant left: ${p.identity}`)
    );

    this._setState("connecting");
    await room.connect(this.config.url, this.config.token);
    this._setState("connected");
    this.log(`connected to room '${this.config.room}' as '${this.config.identity}'`);
  }

  // reliable=false for pose: a retransmitted stale pose is worse than a dropped
  // one, because the arm would act on intent the operator has already moved
  // past. Buttons and lease traffic go reliable — losing a gripper toggle or a
  // take-control claim is not recoverable by the next packet.
  async _publish(topic, text, reliable) {
    const room = this.room;
    if (!room || !room.localParticipant) return;
    const data = this.encoder.encode(text);
    const lp = room.localParticipant;
    try {
      await lp.publishData(data, { reliable, topic });
    } catch (_exc) {
      // Older livekit-client: publishData(data, kind, options)
      const LK = window.LivekitClient;
      const kind = reliable
        ? LK.DataPacket_Kind.RELIABLE
        : LK.DataPacket_Kind.LOSSY;
      await lp.publishData(data, kind, { topic });
    }
  }

  publishXr(text) {
    return this._publish("xr", text, false);
  }

  publishButton(text) {
    return this._publish("btn", text, true);
  }

  publishControl(text) {
    return this._publish("ctl", text, true);
  }

  async disconnect() {
    if (this.room) {
      try {
        await this.room.disconnect();
      } catch (_exc) {
        /* shutdown must not throw */
      }
      this.room = null;
    }
    this._setState("idle");
  }
}
