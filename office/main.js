"use strict";

const WORLD = { width: 1200, height: 720, cell: 32 };
const SYSTEM_TARGETS = new Set(["finish", "approval", "FINISH"]);
const ROOMS = {
  leadership: { x: 28, y: 34, w: 260, h: 330, title: "LEADERSHIP", color: 0x253d4b },
  planning: { x: 306, y: 34, w: 322, h: 330, title: "PLANNING", color: 0x293c50 },
  build: { x: 646, y: 34, w: 526, h: 330, title: "BUILD STUDIO", color: 0x30394d },
  quality: { x: 95, y: 384, w: 523, h: 300, title: "QUALITY LAB", color: 0x3c3547 },
  ops: { x: 636, y: 384, w: 536, h: 300, title: "OPS + DOCS", color: 0x3d3a31 },
};

let events = [];
let scene = null;
let cursor = -1;
let playbackToken = 0;
let speed = 1;

function roomFor(agent) {
  if (["tech_lead", "product_manager", "project_manager"].includes(agent)) return "leadership";
  if (["architect", "api_designer", "ui_ux_designer"].includes(agent) || agent.includes("advisor")) return "planning";
  if (["test_engineer", "qa_runner", "debugger"].includes(agent) || agent.includes("reviewer")) return "quality";
  if (["devops", "docs_writer"].includes(agent)) return "ops";
  return "build";
}

function agentsIn(eventsToRead) {
  const names = new Set();
  for (const event of eventsToRead) {
    if (event.agent) names.add(event.agent);
    if (event.from) names.add(event.from);
    if (event.to && !SYSTEM_TARGETS.has(event.to)) names.add(event.to);
  }
  return [...names].filter((name) => !SYSTEM_TARGETS.has(name)).sort();
}

function parseEventFile(text) {
  const source = text.trim();
  if (!source) throw new Error("The selected file is empty.");
  try {
    const value = JSON.parse(source);
    if (Array.isArray(value)) return value;
    if (Array.isArray(value.events)) return value.events;
  } catch (_) {
    // JSONL is expected for normal event logs, so parse one line at a time below.
  }
  return source.split(/\r?\n/).filter(Boolean).map((line, index) => {
    try {
      return JSON.parse(line);
    } catch (error) {
      throw new Error(`Line ${index + 1} is not valid JSON: ${error.message}`);
    }
  });
}

function describe(event) {
  switch (event.type) {
    case "activate": return `${event.agent} started: ${event.objective || "work assignment"}`;
    case "handoff": return `${event.from} → ${event.to}: ${event.objective || "assignment"}`;
    case "tool_call": return `${event.agent} used ${event.tool}${event.arg ? ` — ${event.arg}` : ""}`;
    case "artifact_written": return `${event.agent} saved ${event.name}`;
    case "tests": return `${event.agent}: ${event.passed} passed, ${event.failed} failed`;
    case "blocked": return `${event.agent} is blocked: ${(event.questions || []).join("; ")}`;
    case "result": return `${event.agent}: ${event.status} — ${event.summary}`;
    case "approval": return `Plan approval ${event.status}: ${event.reason || `${event.task_count} tasks`}`;
    case "finish": return `Run ${event.status}: ${event.reason || "complete"}`;
    default: return `Event: ${event.type || "unknown"}`;
  }
}

class OfficeScene extends Phaser.Scene {
  constructor() {
    super("OfficeScene");
    this.actors = new Map();
    this.blockedCells = new Set();
    this.motions = new Map();
  }

  create() {
    this.cameras.main.setBackgroundColor("#18222e");
    this.drawBackground();
    this.drawRooms();
    this.createActors(agentsIn(events));
    scene = this;
  }

  drawBackground() {
    const grid = this.add.graphics();
    grid.lineStyle(1, 0x253243, 0.36);
    for (let x = 0; x <= WORLD.width; x += WORLD.cell) grid.lineBetween(x, 0, x, WORLD.height);
    for (let y = 0; y <= WORLD.height; y += WORLD.cell) grid.lineBetween(0, y, WORLD.width, y);
  }

  drawRooms() {
    for (const room of Object.values(ROOMS)) {
      this.add.rectangle(room.x + room.w / 2, room.y + room.h / 2, room.w, room.h, room.color, 0.96)
        .setStrokeStyle(2, 0x52647a, 0.9);
      this.add.rectangle(room.x + room.w / 2, room.y + 25, room.w - 2, 48, 0x17212c, 0.8);
      this.add.text(room.x + 15, room.y + 11, room.title, {
        fontFamily: "Segoe UI, sans-serif", fontSize: "13px", color: "#b8c7d8", fontStyle: "bold", letterSpacing: 2,
      });
    }
  }

  createActors(names) {
    const teams = new Map(Object.keys(ROOMS).map((key) => [key, []]));
    for (const name of names) teams.get(roomFor(name)).push(name);
    for (const [roomKey, members] of teams) {
      const room = ROOMS[roomKey];
      const columns = Math.max(1, Math.floor((room.w - 40) / 88));
      members.forEach((name, index) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        const x = room.x + 22 + column * 88;
        const y = room.y + 82 + row * 78;
        const tint = this.colorFor(name);
        this.add.rectangle(x + 28, y + 14, 58, 26, 0x111a24, 0.9).setStrokeStyle(1, 0x65758a, 0.7);
        this.add.rectangle(x + 28, y + 27, 7, 16, 0x5b6878, 0.85);
        this.blockedCells.add(this.cellKey(x + 28, y + 14));
        const avatar = this.add.container(x + 28, y + 51, [
          this.add.ellipse(0, 20, 26, 7, 0x0e151d, 0.55),
          this.add.rectangle(0, 7, 20, 21, tint).setStrokeStyle(1, 0x17212c, 0.8),
          this.add.circle(0, -8, 8, 0xf0c7a6).setStrokeStyle(1, 0x543d35, 0.8),
          this.add.text(0, 27, name.replaceAll("_", " "), {
            fontFamily: "Segoe UI, sans-serif", fontSize: "10px", color: "#edf3fa", align: "center",
            backgroundColor: "#17212c", padding: { x: 3, y: 2 },
          }).setOrigin(0.5, 0),
        ]);
        const actor = { name, home: { x: x + 28, y: y + 51 }, avatar, bubble: null };
        this.actors.set(name, actor);
      });
    }
  }

  colorFor(name) {
    let hash = 0;
    for (const char of name) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
    return [0x4ea995, 0x667fd0, 0xd18158, 0x9d70bd, 0x4e91ad, 0xb17e44][hash % 6];
  }

  cellKey(x, y) {
    return `${Math.floor(x / WORLD.cell)},${Math.floor(y / WORLD.cell)}`;
  }

  findPath(start, end) {
    const cols = Math.ceil(WORLD.width / WORLD.cell);
    const rows = Math.ceil(WORLD.height / WORLD.cell);
    const toCell = (point) => ({ x: Math.floor(point.x / WORLD.cell), y: Math.floor(point.y / WORLD.cell) });
    const origin = toCell(start);
    const destination = toCell(end);
    const keyOf = (point) => `${point.x},${point.y}`;
    const queue = [origin];
    const previous = new Map([[keyOf(origin), null]]);
    const steps = [{ x: 1, y: 0 }, { x: -1, y: 0 }, { x: 0, y: 1 }, { x: 0, y: -1 }];
    for (let cursorIndex = 0; cursorIndex < queue.length; cursorIndex += 1) {
      const current = queue[cursorIndex];
      if (current.x === destination.x && current.y === destination.y) break;
      for (const step of steps) {
        const next = { x: current.x + step.x, y: current.y + step.y };
        const nextKey = keyOf(next);
        if (next.x < 0 || next.y < 0 || next.x >= cols || next.y >= rows || previous.has(nextKey)) continue;
        const centerX = next.x * WORLD.cell + WORLD.cell / 2;
        const centerY = next.y * WORLD.cell + WORLD.cell / 2;
        if (this.blockedCells.has(this.cellKey(centerX, centerY)) && nextKey !== keyOf(destination)) continue;
        previous.set(nextKey, keyOf(current));
        queue.push(next);
      }
    }
    const endKey = keyOf(destination);
    if (!previous.has(endKey)) return [start, end];
    const cells = [];
    let key = endKey;
    while (key) {
      const [x, y] = key.split(",").map(Number);
      cells.push({ x: x * WORLD.cell + WORLD.cell / 2, y: y * WORLD.cell + WORLD.cell / 2 });
      key = previous.get(key);
    }
    cells.reverse();
    return [start, ...cells.slice(1, -1), end];
  }

  async move(actor, destination, animate) {
    if (!actor) return;
    if (!animate) {
      this.cancelMotion(actor);
      actor.avatar.setPosition(destination.x, destination.y);
      return;
    }
    this.cancelMotion(actor);
    const motion = { cancelled: false, resolve: null, tween: null };
    this.motions.set(actor.name, motion);
    const path = this.findPath({ x: actor.avatar.x, y: actor.avatar.y }, destination);
    for (const point of path.slice(1)) {
      if (motion.cancelled) break;
      await new Promise((resolve) => {
        motion.resolve = resolve;
        motion.tween = this.tweens.add({
          targets: actor.avatar,
          x: point.x,
          y: point.y,
          duration: Math.max(45, 115 / speed),
          ease: "Sine.easeInOut",
          onComplete: () => {
            motion.resolve = null;
            motion.tween = null;
            resolve();
          },
        });
      });
    }
    if (this.motions.get(actor.name) === motion) this.motions.delete(actor.name);
  }

  cancelMotion(actor) {
    const motion = this.motions.get(actor.name);
    if (!motion) return;
    motion.cancelled = true;
    if (motion.tween) motion.tween.stop();
    if (motion.resolve) motion.resolve();
    this.motions.delete(actor.name);
  }

  clearBubbles() {
    for (const actor of this.actors.values()) {
      if (actor.bubble) actor.bubble.destroy();
      actor.bubble = null;
    }
  }

  showBubble(agentName, text, thought = false) {
    const actor = this.actors.get(agentName) || this.actors.get("tech_lead");
    if (!actor) return;
    if (actor.bubble) actor.bubble.destroy();
    const content = String(text || "").slice(0, 100);
    const label = this.add.text(0, 0, content, {
      fontFamily: "Segoe UI, sans-serif", fontSize: "11px", color: "#17212c", align: "center",
      wordWrap: { width: 198 },
    }).setOrigin(0.5);
    const width = Math.max(104, Math.min(220, label.width + 22));
    const height = Math.max(34, label.height + 16);
    const bg = this.add.rectangle(0, 0, width, height, thought ? 0xe4d6b6 : 0xf1f4e9, 0.97)
      .setStrokeStyle(2, thought ? 0x9e8254 : 0x778975, 1);
    const tail = thought
      ? this.add.circle(0, height / 2 + 7, 4, 0xe4d6b6)
      : this.add.triangle(0, height / 2 + 5, -6, -5, 6, -5, 0, 6, 0xf1f4e9);
    actor.bubble = this.add.container(actor.avatar.x, actor.avatar.y - 54, [bg, tail, label]).setDepth(20);
  }

  async applyEvent(event, animate) {
    this.clearBubbles();
    if (event.type === "handoff") {
      const lead = this.actors.get(event.from || "tech_lead");
      const target = this.actors.get(event.to);
      if (lead && target) await this.move(lead, target.home, animate);
      this.showBubble(event.from || "tech_lead", event.objective || `Handoff to ${event.to}`);
      return;
    }
    if (event.type === "activate" && event.agent !== "tech_lead") {
      const lead = this.actors.get("tech_lead");
      if (lead) await this.move(lead, lead.home, animate);
    }
    const agent = event.agent || (event.type === "approval" ? "tech_lead" : "tech_lead");
    const thought = ["tool_call", "blocked"].includes(event.type);
    this.showBubble(agent, describe(event), thought);
  }

  async playEvent(index) {
    const event = events[index];
    if (event) await this.applyEvent(event, true);
  }

  seek(index) {
    this.clearBubbles();
    for (const actor of this.actors.values()) {
      this.cancelMotion(actor);
      actor.avatar.setPosition(actor.home.x, actor.home.y);
    }
    for (let item = 0; item <= index; item += 1) {
      const event = events[item];
      if (event.type === "handoff") {
        const lead = this.actors.get(event.from || "tech_lead");
        const target = this.actors.get(event.to);
        if (lead && target) lead.avatar.setPosition(target.home.x, target.home.y);
      } else if (event.type === "activate" && event.agent !== "tech_lead") {
        const lead = this.actors.get("tech_lead");
        if (lead) lead.avatar.setPosition(lead.home.x, lead.home.y);
      }
    }
    if (events[index]) {
      const event = events[index];
      this.showBubble(event.agent || event.from || "tech_lead", describe(event), ["tool_call", "blocked"].includes(event.type));
    }
  }

  loadEvents() {
    for (const actor of this.actors.values()) this.cancelMotion(actor);
    this.children.removeAll(true);
    this.actors.clear();
    this.blockedCells.clear();
    this.cameras.main.setBackgroundColor("#18222e");
    this.drawBackground();
    this.drawRooms();
    this.createActors(agentsIn(events));
  }
}

const game = new Phaser.Game({
  type: Phaser.AUTO,
  parent: "office-canvas",
  width: WORLD.width,
  height: WORLD.height,
  backgroundColor: "#18222e",
  scale: { mode: Phaser.Scale.FIT, autoCenter: Phaser.Scale.CENTER_BOTH },
  scene: [OfficeScene],
});

const fileInput = document.querySelector("#event-file");
const sampleButton = document.querySelector("#sample-button");
const playButton = document.querySelector("#play-button");
const pauseButton = document.querySelector("#pause-button");
const speedSelect = document.querySelector("#speed-select");
const timeline = document.querySelector("#timeline");
const eventCount = document.querySelector("#event-count");
const eventKind = document.querySelector("#event-kind");
const eventCaption = document.querySelector("#event-caption");

function setCaption(event, index) {
  eventKind.textContent = event ? event.type.replaceAll("_", " ").toUpperCase() : "READY";
  eventCaption.textContent = event ? describe(event) : "Load a run log or the sample to begin.";
  eventCount.textContent = events.length ? `${Math.max(index + 1, 0)} / ${events.length}` : "No run loaded";
}

function updateControls() {
  const ready = events.length > 0;
  playButton.disabled = !ready;
  pauseButton.disabled = !ready;
  speedSelect.disabled = !ready;
  timeline.disabled = !ready;
  timeline.max = String(Math.max(events.length - 1, 0));
  timeline.value = String(Math.max(cursor, 0));
}

function loadEvents(nextEvents, label) {
  if (!Array.isArray(nextEvents) || !nextEvents.length) throw new Error("No events were found in this file.");
  for (let index = 0; index < nextEvents.length; index += 1) {
    const event = nextEvents[index];
    if (!event || typeof event.type !== "string") throw new Error(`Event ${index + 1} has no type.`);
    if (typeof event.t !== "number") throw new Error(`Event ${index + 1} has no numeric timeline field 't'.`);
  }
  playbackToken += 1;
  events = nextEvents;
  cursor = -1;
  setCaption(null, cursor);
  eventCaption.textContent = `${label}: ${events.length} events ready. Press Play.`;
  updateControls();
  if (scene) scene.loadEvents();
}

async function pause(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function play() {
  if (!events.length) return;
  playbackToken += 1;
  const token = playbackToken;
  playButton.disabled = true;
  let index = cursor + 1;
  if (index >= events.length) index = 0;
  while (index < events.length && token === playbackToken) {
    const prior = events[Math.max(index - 1, 0)];
    const event = events[index];
    if (scene) await scene.playEvent(index);
    if (token !== playbackToken) break;
    cursor = index;
    timeline.value = String(cursor);
    setCaption(event, cursor);
    const elapsed = Math.max(0, (event.t - (prior?.t || 0)) * 1000 / speed);
    await pause(Math.max(220, Math.min(2200, elapsed || 450)));
    index += 1;
  }
  if (token === playbackToken) {
    playbackToken += 1;
    playButton.disabled = false;
  }
}

function seek(index) {
  playbackToken += 1;
  cursor = Math.max(0, Math.min(Number(index), events.length - 1));
  if (scene) scene.seek(cursor);
  timeline.value = String(cursor);
  setCaption(events[cursor], cursor);
  playButton.disabled = false;
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  try {
    loadEvents(parseEventFile(await file.text()), file.name);
  } catch (error) {
    eventKind.textContent = "LOAD ERROR";
    eventCaption.textContent = error.message;
  }
});

sampleButton.addEventListener("click", async () => {
  try {
    const response = await fetch("sample-events.jsonl");
    if (!response.ok) throw new Error(`Could not load sample (${response.status}).`);
    loadEvents(parseEventFile(await response.text()), "Sample run");
  } catch (error) {
    eventKind.textContent = "LOAD ERROR";
    eventCaption.textContent = `${error.message} Open sample-events.jsonl with the file picker, or serve this folder over localhost.`;
  }
});

playButton.addEventListener("click", play);
pauseButton.addEventListener("click", () => {
  playbackToken += 1;
  playButton.disabled = false;
});
timeline.addEventListener("input", (event) => seek(event.target.value));
speedSelect.addEventListener("change", (event) => { speed = Number(event.target.value) || 1; });

window.addEventListener("load", () => {
  if (!window.Phaser) {
    eventKind.textContent = "PHASER MISSING";
    eventCaption.textContent = "The pinned Phaser script did not load. Check the network connection and refresh.";
  }
});
