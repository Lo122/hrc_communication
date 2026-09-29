/* State-driven HRC watch. The backend decides WHAT is on screen
 * (GET /api/watch -> snapshot.screen); this file only renders it and sends
 * the chosen command back (POST /api/watch/command).
 *
 * Two skins, each with a light and a dark mode (see UX_GUIDE.md):
 *   clay  — the pressed pad is the subject, drafting details underneath
 *   plate — one big instrument: the pad becomes a gauge face with a tick ring
 *   sheet — a drawing block: left-aligned stamp and title, the signal plotted
 *           on one pressed plate, the readings on dimension rails
 *
 * The signal inside the pad is stepped and square while the robot speaks and
 * soft concentric rings while it listens to you; while the robot moves it
 * becomes the progress arc with the speed scale inside it. */

(() => {
  const POLL_MS = 300;
  const HOLD_MS = 5000;      // press-and-hold before a guarded command fires
  const CONFIRM_MS = 2000;   // how long the pill stays red afterwards
  const SPEED_SEGS = 8;
  const TALK_HOLD_MS = 900;  // keep a voice signal alive across a gap in the data
  const GAUGE_SCREENS = ["running", "paused", "starting", "restarting", "homing", "stopping"];
  const DEEP_TONES = ["run", "wait", "alert"];
  const SKINS = ["clay", "plate", "sheet"];
  const ARC_LEN = 2 * Math.PI * 45;   // gauge circle: r = 45 in a 100x100 viewBox

  const ICONS = {
    check: '<path d="M4 12.5l5 5L20 6.5"/>',
    x: '<path d="M6 6l12 12M18 6L6 18"/>',
    clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    pause: '<path d="M9 5v14M15 5v14"/>',
    play: '<path d="M8 5l11 7-11 7z" fill="currentColor"/>',
    redo: '<path d="M19 12a7 7 0 1 1-2.05-4.95M19 4v4h-4"/>',
    stop: '<rect x="6.5" y="6.5" width="11" height="11" rx="2.5" fill="currentColor"/>',
    hand: '<path d="M8 12V6.5a1.5 1.5 0 0 1 3 0V11m0-5.5V5a1.5 1.5 0 0 1 3 0v6m0-4.5a1.5 1.5 0 0 1 3 0V14a6 6 0 0 1-6 6h-.5a6 6 0 0 1-4.9-2.6L4.3 14a1.5 1.5 0 0 1 2.4-1.8L8 13.5"/>',
    home: '<path d="M4 11l8-7 8 7M6.5 9.5V20h11V9.5"/>',
  };
  const HAPTICS = { tap: [40], ask: [70, 60, 70], alert: [200, 80, 200, 80, 200], success: [30, 40, 30] };

  // 12 tick marks around the rim — the plate skin shows them, clay hides them.
  const TICKS = Array.from({ length: 12 }, (_, i) => {
    const a = (i / 12) * Math.PI * 2 - Math.PI / 2;
    const inner = i % 3 === 0 ? 33 : 36.5;
    const x = (r) => (50 + r * Math.cos(a)).toFixed(2);
    const y = (r) => (50 + r * Math.sin(a)).toFixed(2);
    return `<line class="tick" x1="${x(inner)}" y1="${y(inner)}" x2="${x(40)}" y2="${y(40)}"/>`;
  }).join("");

  const TEMPLATE = `
    <div class="screen-viewport">
      <div class="screen" data-theme="light" data-tone="idle" data-sig="steps">
        <div class="paper" aria-hidden="true"><span class="cross tl"></span><span class="cross tr"></span><span class="cross bl"></span><span class="cross br"></span></div>

        <header class="status-bar">
          <span class="link" data-link="off"><span class="link-dot"></span><span class="link-text">HRC</span></span>
          <span class="badge" data-ref="badge" hidden></span>
          <time data-ref="clock">--:--</time>
        </header>

        <p class="stamp-row"><span class="stamp" data-ref="stamp" hidden></span><span class="status-word" data-ref="statusWord" hidden></span></p>

        <div class="stage">
          <div class="pad" data-ref="pad">
            <svg class="gauge" viewBox="0 0 100 100" aria-hidden="true">
              <g class="ticks">${TICKS}</g>
              <circle class="gauge-track" cx="50" cy="50" r="45"></circle>
              <circle class="gauge-arc is-hidden" data-ref="gaugeArc" cx="50" cy="50" r="45"></circle>
            </svg>
            <span class="pad-label" aria-hidden="true">SIGNAL</span>
            <div class="sig">
              <div class="sig-bars" data-ref="bars" aria-hidden="true"></div>
              <svg class="sig-wave" viewBox="0 0 120 40" preserveAspectRatio="none" aria-hidden="true">
                <path class="wave-back" d="M0 20 C 12 6, 24 34, 36 20 S 60 4, 72 20 S 96 34, 120 18"/>
                <path class="wave-front" d="M0 22 C 12 8, 26 32, 38 18 S 62 6, 74 22 S 98 32, 120 16"/>
              </svg>
              <div class="sig-steps" aria-hidden="true"><i></i><i></i><i></i></div>
              <div class="sig-rings" aria-hidden="true"><i></i><i></i><i></i><b></b></div>
              <div class="sig-speed" data-ref="sigSpeed" hidden>
                <span class="sig-label">SPEED</span>
                <div class="segs" data-ref="segs"></div>
              </div>
            </div>
          </div>
        </div>

        <h1 class="title" data-ref="title">Connecting…</h1>
        <p class="detail" data-ref="detail"></p>

        <div class="rail" data-ref="rail" hidden>
          <button class="speed-btn" type="button" data-command="H_SLOWDOWN" data-ref="slower" aria-label="Slower" hidden>−</button>
          <span class="rail-label" data-ref="railLabel"></span>
          <span class="rail-track" data-ref="railTrack"><i data-ref="railFill"></i></span>
          <span class="rail-value" data-ref="railValue"></span>
          <div class="segs segs-rail" data-ref="segsRail" hidden></div>
          <button class="speed-btn" type="button" data-command="H_SPEEDUP" data-ref="faster" aria-label="Faster" hidden>+</button>
        </div>

        <div class="actions" data-ref="actions"></div>
        <div class="toast" data-ref="toast" role="status" aria-live="polite"></div>
      </div>
    </div>`;

  function icon(name) {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ""}</svg>`;
  }

  function mount(root, options = {}) {
    root.classList.add("watch");
    root.innerHTML = TEMPLATE;
    const $ = (ref) => root.querySelector(`[data-ref="${ref}"]`);
    const written = new Map();
    const setText = (ref, value) => {          // write only when it changed
      if (written.get(ref) === value) return;
      written.set(ref, value);
      $(ref).textContent = value;
    };
    const setHidden = (ref, hidden) => {
      const el = $(ref);
      if (el.hidden !== hidden) el.hidden = hidden;
    };
    const setData = (el, key, value) => {
      if (el.dataset[key] !== value) el.dataset[key] = value;
    };
    const screenEl = root.querySelector(".screen");
    const linkEl = root.querySelector(".link");
    const listeners = [];
    const segs = $("segs");
    for (let i = 0; i < SPEED_SEGS; i += 1) segs.appendChild(document.createElement("i"));
    for (let i = 0; i < SPEED_SEGS; i += 1) $("segsRail").appendChild(document.createElement("i"));
    // The sheet skin plots the robot's voice as a bar trace.
    for (let i = 0; i < 15; i += 1) $("bars").appendChild(document.createElement("i"));
    $("gaugeArc").style.strokeDasharray = `${ARC_LEN}`;

    const st = {
      snap: null,
      clockOffset: 0,
      screenKey: null,
      actionsKey: null,
      lastMessageTs: Infinity,
      sending: false,
      toastTimer: null,
      talk: "none",
      talkAt: 0,
      sound: false,
      shape: "rect",
      skin: "clay",
      mode: "light",
    };

    /* ---------- feedback channels ---------- */
    function toast(text) {
      const el = $("toast");
      el.textContent = text;
      el.classList.add("is-visible");
      clearTimeout(st.toastTimer);
      st.toastTimer = setTimeout(() => el.classList.remove("is-visible"), 2600);
    }

    function beep(kind) {
      if (!st.sound) return;
      try {
        const ctx = (st.audio ||= new AudioContext());
        const tones = kind === "alert" ? [880, 660, 880] : kind === "ask" ? [660, 880] : [740];
        tones.forEach((f, i) => {
          const o = ctx.createOscillator();
          const g = ctx.createGain();
          o.frequency.value = f;
          g.gain.setValueAtTime(0.08, ctx.currentTime + i * 0.12);
          g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.12 + 0.1);
          o.connect(g).connect(ctx.destination);
          o.start(ctx.currentTime + i * 0.12);
          o.stop(ctx.currentTime + i * 0.12 + 0.11);
        });
      } catch (_) { /* audio optional */ }
    }

    function haptic(kind) {
      if (!kind || kind === "none") return;
      if (navigator.vibrate) navigator.vibrate(HAPTICS[kind] || HAPTICS.tap);
      root.classList.remove("is-buzzing");
      void root.offsetWidth;
      root.classList.add("is-buzzing");
      beep(kind);
    }

    /* ---------- networking ---------- */
    async function send(command, extra = {}) {
      if (st.sending) return;
      st.sending = true;
      try {
        const res = await fetch("/api/watch/command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ command, task_instance_id: st.snap?.task?.task_instance_id ?? null, ...extra }),
        });
        const body = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(body.detail || res.status);
        listeners.forEach((fn) => fn({ type: "command", command, ok: true }));
        render(body);
      } catch (err) {
        toast(`Not sent: ${err.message}`);
        listeners.forEach((fn) => fn({ type: "command", command, ok: false, error: String(err.message) }));
      } finally {
        st.sending = false;
      }
    }

    async function poll() {
      try {
        const res = await fetch("/api/watch", { cache: "no-store" });
        if (!res.ok) throw new Error(res.status);
        render(await res.json());
      } catch (_) {
        linkEl.dataset.link = "off";
        if (!st.snap) $("title").textContent = "Connecting…";
      } finally {
        setTimeout(poll, POLL_MS);
      }
    }

    /* ---------- rendering ---------- */
    function makePill(a) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "btn";
      b.dataset.role = a.role;
      b.dataset.command = a.command;
      if (a.hold) {
        b.dataset.hold = "1";
        b.setAttribute("aria-label", `${a.label} (press and hold)`);
      }
      b.innerHTML = `${a.hold ? '<span class="fill"></span>' : ""}${icon(a.icon)}<span class="label">${a.label}</span>`;
      return b;
    }

    function renderActions(screen) {
      const key = st.skin + "|" + screen.screen + "|" + screen.actions.map((a) => a.command).join(",");
      if (key === st.actionsKey) return;
      st.actionsKey = key;
      const box = $("actions");
      box.innerHTML = "";
      const primary = screen.actions.filter((a) => a.role === "primary");
      const rest = screen.actions.filter((a) => a.role !== "primary");

      const rowOf = (items) => {
        const row = document.createElement("div");
        row.className = "row";
        items.forEach((a) => row.appendChild(makePill(a)));
        box.appendChild(row);
      };
      if (screen.actions.length <= 3) {
        rowOf([...primary, ...rest]);
      } else {
        if (primary.length) rowOf(primary);
        for (let i = 0; i < rest.length; i += 2) rowOf(rest.slice(i, i + 2));
      }
      if (screen.actions.some((a) => a.hold)) {
        const hint = document.createElement("div");
        hint.className = "hold-hint";
        hint.textContent = `HOLD ${Math.round(HOLD_MS / 1000)}s TO STOP`;
        box.appendChild(hint);
      }
    }

    function setArc(fraction) {              // 0 = empty, 1 = the full circle
      const arc = $("gaugeArc");
      if (fraction === null) { arc.classList.add("is-hidden"); return; }
      arc.classList.remove("is-hidden");
      arc.style.strokeDashoffset = `${ARC_LEN * (1 - Math.min(1, Math.max(0, fraction)))}`;
    }

    /* The rail under the title carries whatever is counting right now: the
     * seconds left on a question, or how far the robot has come. */
    function renderRail(snap, now) {
      const s = snap.screen;
      const rail = $("rail");
      const fill = $("railFill");
      const track = $("railTrack");

      if (st.skin === "clay") setArc(null);

      if (st.skin === "plate") {
        // The gauge face already shows the countdown and the progress; the row
        // under it carries the speed alone.
        if (s.countdown_deadline) {
          const left = Math.max(0, s.countdown_deadline - now);
          setArc(s.countdown_total ? left / s.countdown_total : 0);
        } else if (GAUGE_SCREENS.includes(s.screen) && typeof snap.progress === "number") {
          setArc(snap.progress);
        } else {
          setArc(null);
        }
        rail.hidden = !s.show_speed;
        $("segsRail").hidden = !s.show_speed;
        return;
      }
      $("segsRail").hidden = true;

      if (s.countdown_deadline) {
        const left = Math.max(0, s.countdown_deadline - now);
        const fraction = s.countdown_total ? left / s.countdown_total : 0;
        rail.hidden = false;
        setText("railLabel", st.skin === "sheet" ? "ANS" : "ANSWER");
        setText("railValue", `${Math.ceil(left)}s`);
        track.classList.remove("is-indeterminate");
        fill.style.width = `${Math.round(fraction * 100)}%`;
        setArc(fraction);
        return;
      }
      if (GAUGE_SCREENS.includes(s.screen)) {
        const known = typeof snap.progress === "number";
        rail.hidden = false;
        setText("railLabel", st.skin === "sheet" ? "PRG" : "PROGRESS");
        // Only the drawing block carries the number; the other skins read the
        // filling line alone.
        setText("railValue", st.skin === "sheet" && known
          ? `${Math.round(snap.progress * 100)}%` : "");
        track.classList.toggle("is-indeterminate", !known);
        fill.style.width = known ? `${Math.round(snap.progress * 100)}%` : "";
        setArc(known ? snap.progress : null);
        return;
      }
      rail.hidden = true;
      setArc(null);
    }

    function render(snap) {
      st.snap = snap;
      st.clockOffset = snap.server_time - Date.now() / 1000;
      if (st.lastMessageTs === Infinity) st.lastMessageTs = snap.server_time;
      const s = snap.screen;
      linkEl.dataset.link = snap.mode;
      root.querySelector(".link-text").textContent = snap.mode === "sim" ? "SIM" : "HRC";

      setData(screenEl, "tone", s.tone);
      setData(screenEl, "screen", s.screen);
      setData(screenEl, "theme", DEEP_TONES.includes(s.tone) ? "deep" : "light");

      // The stamp is the line the backend already writes (LIFT PANEL,
      // PANEL LIFTED, MANUAL RECOVERY…) — no second vocabulary on the watch.
      setText("stamp", s.eyebrow);
      setHidden("stamp", !s.eyebrow);
      setText("title", s.title);
      setText("detail", s.detail);
      setHidden("detail", !s.detail);

      // Square and stepped while the robot talks, soft rings while it listens
      // to you, the progress arc while it moves.
      // Who holds the channel. The runtime can go quiet for a moment (the
      // voice layer waits for the microphone thread when a question is
      // announced); without this the signal would stop mid-sentence and pick
      // up again, which reads as a stutter. So a voice signal survives a short
      // gap in the data and only then falls back to silence.
      const voice = snap.voice || {};
      const now = Date.now();
      const reported = voice.speaking ? "speaking" : voice.listening ? "listening" : "none";
      if (reported !== "none") { st.talk = reported; st.talkAt = now; }
      const talk = reported !== "none"
        ? reported
        : (now - st.talkAt < TALK_HOLD_MS ? st.talk : "none");

      const gaugeOn = GAUGE_SCREENS.includes(s.screen);
      setData(screenEl, "sig", gaugeOn ? "gauge" : talk === "listening" ? "rings" : "steps");
      setData(screenEl, "talk", talk);

      // Beside the stamp, who holds the channel right now (sheet skin only).
      const word = gaugeOn ? "UR10E"
        : talk === "speaking" ? "ROBOT SPEAKING"
        : talk === "listening" ? "MIC OPEN" : "";
      setText("statusWord", word);
      setHidden("statusWord", !word);

      setHidden("sigSpeed", !(gaugeOn && s.show_speed));
      // On the sheet skin the plate is labelled by what it is plotting.
      const padLabel = gaugeOn && s.show_speed ? "SPEED" : "SIGNAL";
      const labelEl = root.querySelector(".pad-label");
      if (labelEl.textContent !== padLabel) labelEl.textContent = padLabel;
      setHidden("slower", !s.show_speed);
      setHidden("faster", !s.show_speed);
      if (s.show_speed) {
        // Speed is a scale, not a number: the segments fill as the robot
        // speeds up and the whole row brightens with it.
        const { value, min, max } = snap.speed;
        const lit = Math.min(1, Math.max(0, (value - min) / (max - min)));
        const on = Math.max(1, Math.round(lit * SPEED_SEGS));
        [segs, $("segsRail")].forEach((row) => {
          row.style.setProperty("--lit", lit.toFixed(3));
          [...row.children].forEach((el, i) => el.classList.toggle("on", i < on));
        });
      }

      setHidden("badge", !(snap.pending.length && snap.task));
      setText("badge", `${snap.pending.length} waiting`);

      renderActions(s);
      renderRail(snap, Date.now() / 1000 + st.clockOffset);

      const screenKey = `${snap.task?.task_instance_id}|${s.screen}`;
      const screenChanged = screenKey !== st.screenKey;
      if (screenChanged) {
        if (st.screenKey !== null) haptic(s.haptic);
        st.screenKey = screenKey;
      }

      // Spoken prompts repeat what a new screen already shows, so only toast
      // messages that arrive without a screen change (speed, invalid command).
      const fresh = snap.messages.filter((m) => m.timestamp > st.lastMessageTs);
      if (fresh.length) {
        st.lastMessageTs = fresh[fresh.length - 1].timestamp;
        const text = fresh[fresh.length - 1].text || "";
        // Speed acknowledgements are already visible in the scale.
        if (!screenChanged && !/speed/i.test(text)) toast(text);
      }
      listeners.forEach((fn) => fn({ type: "snapshot", snap }));
    }

    /* ---------- input: tap and press-and-hold ---------- */
    let holdTimer = null;
    let holdBtn = null;
    const cancelHold = () => {
      clearTimeout(holdTimer);
      holdBtn?.classList.remove("is-holding");
      holdBtn = null;
    };

    root.addEventListener("pointerdown", (e) => {
      const b = e.target.closest("button[data-hold]");
      if (!b || b.disabled) return;
      holdBtn = b;
      b.classList.add("is-holding");
      holdTimer = setTimeout(() => {
        const cmd = b.dataset.command;
        cancelHold();
        b.classList.add("is-confirmed");
        setTimeout(() => b.classList.remove("is-confirmed"), CONFIRM_MS);
        send(cmd);
      }, HOLD_MS);
    });
    ["pointerup", "pointerleave", "pointercancel"].forEach((t) => root.addEventListener(t, cancelHold));

    root.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-command]");
      if (!b || b.disabled) return;
      if (b.dataset.hold) {
        if (e.detail === 0) send(b.dataset.command);   // keyboard: Enter/Space
        else toast("Press and hold to stop");
        return;
      }
      send(b.dataset.command);
    });

    function tickClock() {
      const now = new Date(Date.now() + st.clockOffset * 1000);
      $("clock").textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
      if (st.snap) renderRail(st.snap, Date.now() / 1000 + st.clockOffset);
    }
    setInterval(tickClock, 250);

    root.dataset.skin = st.skin;
    root.dataset.mode = st.mode;
    root.style.setProperty("--hold-ms", `${HOLD_MS}ms`);
    tickClock();
    poll();

    return {
      onChange: (fn) => listeners.push(fn),
      setShape: (shape) => {
        // Logical points: Apple Watch 45 mm = 198x242, Galaxy Watch (454 px @2x) = 227x227.
        st.shape = shape;
        root.dataset.shape = shape;
        const [w, h] = shape === "round" ? [227, 227] : [198, 242];
        root.style.setProperty("--screen-w", `${w}px`);
        root.style.setProperty("--screen-h", `${h}px`);
        document.documentElement.style.setProperty("--screen-w", `${w}px`);
        document.documentElement.style.setProperty("--screen-h", `${h}px`);
      },
      setSkin: (skin) => {
        st.skin = SKINS.includes(skin) ? skin : "clay";
        root.dataset.skin = st.skin;
        st.actionsKey = null;
        written.clear();
        if (st.snap) render(st.snap);
      },
      setMode: (mode) => {
        st.mode = mode === "dark" ? "dark" : "light";
        root.dataset.mode = st.mode;
      },
      setSound: (on) => { st.sound = on; },
      send,
    };
  }

  window.HRCWatch = { mount };
})();
