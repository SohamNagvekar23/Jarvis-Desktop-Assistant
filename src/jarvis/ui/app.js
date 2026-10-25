// Jarvis UI: renders events pushed from Python (window.jarvis.onEvent) and calls
// back through window.pywebview.api. Shared by index.html and mini.html.
// Text is always set with textContent, never innerHTML. All animation is CSS;
// the only timers here are the clock and the minimum time WAKE DETECTED shows.
(() => {
  "use strict";

  const page = document.body.dataset.page;  // "main" or "mini"
  const $ = (selector) => document.querySelector(selector);
  const WAKE_MIN_MS = 700;  // the wake state is brief (one beep); keep it visible a moment

  function call(name, ...args) {
    const api = window.pywebview && window.pywebview.api;
    if (api && typeof api[name] === "function") {
      return api[name](...args);
    }
    return Promise.resolve(null);
  }

  // ---------- orb state ----------

  let shownOrb = "idle";
  let wakeShownAt = 0;
  let pendingOrb = null;
  let pendingTimer = 0;

  function showOrb(orb, label) {
    shownOrb = orb;
    if (orb === "wake") {
      wakeShownAt = performance.now();
    }
    for (const id of ["orb", "stage", "mini-orb", "pill"]) {
      const element = document.getElementById(id);
      if (element) {
        element.dataset.state = orb;
      }
    }
    $("#status-label").textContent = label;
  }

  function setOrb(value) {
    clearTimeout(pendingTimer);
    const elapsed = performance.now() - wakeShownAt;
    if (shownOrb === "wake" && value.orb !== "wake" && elapsed < WAKE_MIN_MS) {
      pendingOrb = value;  // the latest state wins once the wake moment has shown
      pendingTimer = setTimeout(() => showOrb(pendingOrb.orb, pendingOrb.label), WAKE_MIN_MS - elapsed);
      return;
    }
    showOrb(value.orb, value.label);
  }

  // ---------- event handlers ----------

  const MODULE_ORDER = ["microphone", "stt", "brain", "voice", "youtube"];
  const loggedIds = new Set();

  const handlers = {
    state: setOrb,

    phrase(text) {
      $("#phrase").textContent = text || "Say “Hey Jarvis”";
    },

    muted(isMuted) {
      document.body.classList.toggle("muted", Boolean(isMuted));
      for (const button of document.querySelectorAll('[data-action="mute"]')) {
        button.title = isMuted ? "Unmute the microphone" : "Mute the microphone";
      }
    },

    module(value) {
      if (page !== "main" || !MODULE_ORDER.includes(value.name)) return;
      const row = document.querySelector(`[data-module="${value.name}"]`);
      row.dataset.level = value.level;
      row.querySelector(".module-status").textContent = value.text;
    },

    meters(values) {
      if (page !== "main") return;
      for (const [name, meter] of Object.entries(values)) {
        const element = document.querySelector(`[data-meter="${name}"]`);
        if (!element) continue;
        element.classList.toggle("unavailable", meter === null);
        element.querySelector(".meter-value").textContent = meter === null ? "N/A" : meter.text;
        element.querySelector(".meter-fill").style.width = meter === null ? "0" : `${meter.percent}%`;
      }
    },

    confirm(question) {
      if (page !== "main") return;
      const box = $("#confirm");
      const input = $("#command");
      box.hidden = !question;
      $("#confirm-question").textContent = question || "";
      input.placeholder = question ? "Type yes or no" : "Type a command, or say ‘Hey Jarvis’";
      if (question) input.focus();
    },

    error(message) {
      if (page === "mini") {
        if (message) {
          $("#status-label").textContent = "ERROR";
          $("#phrase").textContent = message;
        }
        return;
      }
      $("#error").hidden = !message;
      $("#error-text").textContent = message || "";
    },

    log(entry) {
      if (page !== "main" || loggedIds.has(entry.id)) return;
      loggedIds.add(entry.id);
      const list = $("#log");
      const empty = $("#log-empty");
      if (empty) empty.remove();

      const item = document.createElement("div");
      item.className = "entry";
      const said = document.createElement("div");
      said.className = "entry-said";
      const time = document.createElement("span");
      time.className = "entry-time";
      time.textContent = entry.time;
      const saidText = document.createElement("span");
      saidText.className = "entry-said-text";
      saidText.textContent = entry.said || "(nothing heard)";
      said.append(time, saidText);

      const reply = document.createElement("div");
      reply.className = "entry-reply";
      const replyText = document.createElement("span");
      replyText.className = "entry-reply-text";
      replyText.textContent = entry.reply || "(done)";
      const tag = document.createElement("span");
      tag.className = `tag ${entry.tag}`;
      tag.textContent = entry.tag.toUpperCase();
      reply.append(replyText, tag);

      item.append(said, reply);
      const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 40;
      list.append(item);
      if (atBottom) list.scrollTop = list.scrollHeight;
      $("#log-count").textContent = `TODAY · ${loggedIds.size}`;
    },
  };

  window.jarvis = {
    onEvent(event) {
      const handler = handlers[event.type];
      if (handler) handler(event.value);
    },
  };

  // ---------- controls ----------

  function bindActions() {
    document.addEventListener("click", (event) => {
      const target = event.target.closest("[data-action], [data-command], [data-answer]");
      if (!target) return;
      if (target.dataset.command) {
        call("submit", target.dataset.command);  // quick actions go through the same pipeline
      } else if (target.dataset.answer) {
        call("submit", target.dataset.answer);
      } else {
        const action = target.dataset.action;
        if (action === "mute") call("toggle_mute");
        else if (action === "minimize") call("minimize");
        else if (action === "maximize") call("maximize");
        else if (action === "close") call("close");
        else if (action === "restore") call("restore");
        else if (action === "settings") openSettings();
        else if (action === "settings-close") $("#settings").hidden = true;
      }
    });
    if (page !== "main") return;

    $("#startup-toggle").addEventListener("change", async (event) => {
      const toggle = event.target;
      toggle.disabled = true;
      await showStartup(await call("set_startup", toggle.checked));
    });

    const input = $("#command");
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && input.value.trim()) {
        call("submit", input.value);
        input.value = "";
      } else if (event.key === "Escape") {
        input.value = "";
      }
    });

    const hold = $("#hold");
    const release = () => {
      if (!hold.classList.contains("held")) return;
      hold.classList.remove("held");
      call("hold_end");
    };
    hold.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      hold.setPointerCapture(event.pointerId);
      hold.classList.add("held");
      call("hold_start");
    });
    hold.addEventListener("pointerup", release);
    hold.addEventListener("pointercancel", release);
    hold.addEventListener("lostpointercapture", release);
  }

  async function openSettings() {
    const settings = (await call("settings")) || {};
    const labels = {
      wakeWord: "Wake word", wakeThreshold: "Wake threshold", speechModel: "Speech model",
      brainModel: "Brain model", brainUrl: "Brain server", volumeStep: "Volume step",
      screenshots: "Screenshots",
    };
    const list = $("#settings-list");
    list.replaceChildren();
    for (const [key, label] of Object.entries(labels)) {
      const term = document.createElement("dt");
      term.textContent = label;
      const value = document.createElement("dd");
      value.textContent = key === "volumeStep" ? `${settings[key]}%` : String(settings[key] ?? "--");
      list.append(term, value);
    }
    await showStartup(await call("startup_status"));
    $("#settings").hidden = false;
  }

  const STARTUP_NOTES = {
    on: "Opens as the mini pill at login",
    off: "Opens as the mini pill at login",
    outdated: "The entry points to an old location; switch on to fix",
    unavailable: "Not available (pythonw.exe not found)",
  };

  async function showStartup(status) {
    const toggle = $("#startup-toggle");
    const note = $("#startup-note");
    toggle.checked = status === "on";
    toggle.disabled = status === "unavailable" || !status;
    note.textContent = STARTUP_NOTES[status] || "";
    note.classList.toggle("outdated", status === "outdated");
  }

  // ---------- static decoration, built once ----------

  function buildDial() {
    const dial = document.getElementById("dial");
    if (!dial) return;
    const ns = "http://www.w3.org/2000/svg";
    for (let i = 0; i < 120; i += 1) {
      const major = i % 10 === 0;
      const tick = document.createElementNS(ns, "line");
      tick.setAttribute("x1", "190");
      tick.setAttribute("x2", "190");
      tick.setAttribute("y1", major ? "2" : "5");
      tick.setAttribute("y2", major ? "16" : "11");
      tick.setAttribute("stroke", "currentColor");
      tick.setAttribute("stroke-width", major ? "1.5" : "1");
      tick.setAttribute("opacity", major ? "0.9" : "0.45");
      tick.setAttribute("transform", `rotate(${i * 3} 190 190)`);
      dial.append(tick);
    }
  }

  function startClock() {
    const clock = document.getElementById("clock");
    if (!clock) return;
    const tick = () => {
      clock.textContent = new Date().toLocaleTimeString("en-GB", { hour12: false });
    };
    tick();
    setInterval(tick, 1000);
  }

  // ---------- start ----------

  buildDial();
  startClock();
  bindActions();

  window.addEventListener("pywebviewready", async () => {
    const snapshot = await call("snapshot");
    if (snapshot) {
      for (const event of snapshot.events) window.jarvis.onEvent(event);
      for (const entry of snapshot.log) handlers.log(entry);
    }
    if (page === "main") {
      const settings = (await call("settings")) || {};
      if (settings.wakeWord) $("#wake-phrase").textContent = settings.wakeWord;
      if (settings.wakeThreshold !== undefined) {
        $("#wake-threshold").textContent = Number(settings.wakeThreshold).toFixed(2);
      }
    }
  });
})();
