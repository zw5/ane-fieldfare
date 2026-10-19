/// The chat page the server serves at `/`: one self-contained HTML file (no
/// external scripts, styles or fonts) that talks to this server's own
/// `/v1/chat/completions` with streaming. Chats and settings live in the
/// browser's `localStorage`.
enum ChatPage {
    static let html = #"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fieldfare Chat</title>
<style>
:root {
  --bg: #f7f7f5; --panel: #ffffff; --side: #efefec; --text: #1d1d1f; --muted: #6b6b70;
  --line: #e2e2de; --accent: #2f6fde; --accent-text: #ffffff; --user: #e7eefc; --code: #f1f1ee;
  --danger: #c53b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161618; --panel: #1f1f22; --side: #1a1a1c; --text: #ececee; --muted: #9a9aa1;
    --line: #2e2e33; --accent: #5b8ff0; --accent-text: #0e0e10; --user: #263247; --code: #141416;
    --danger: #e06a6a;
  }
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body {
  background: var(--bg); color: var(--text);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  display: flex; overflow: hidden;
}
button { font: inherit; color: inherit; cursor: pointer; }
#side {
  width: 250px; flex: none; background: var(--side); border-right: 1px solid var(--line);
  display: flex; flex-direction: column;
}
#side header { padding: 14px; display: flex; gap: 8px; }
#new {
  flex: 1; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line);
  background: var(--panel); text-align: left;
}
#chats { flex: 1; overflow-y: auto; padding: 0 8px 8px; }
.chat-item {
  display: flex; align-items: center; gap: 6px; padding: 7px 9px; border-radius: 7px;
  color: var(--muted); cursor: pointer; white-space: nowrap; overflow: hidden;
}
.chat-item span { flex: 1; overflow: hidden; text-overflow: ellipsis; }
.chat-item.active, .chat-item:hover { background: var(--panel); color: var(--text); }
.chat-item button { border: 0; background: none; color: var(--muted); padding: 0 2px; visibility: hidden; }
.chat-item:hover button { visibility: visible; }
#side footer { padding: 12px 14px; border-top: 1px solid var(--line); font-size: 13px; color: var(--muted); }
#status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--muted); margin-right: 6px; }
#status-dot.ok { background: #3aa76d; } #status-dot.busy { background: #d99a2b; } #status-dot.down { background: var(--danger); }
main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
#top {
  height: 50px; flex: none; display: flex; align-items: center; justify-content: space-between;
  padding: 0 16px; border-bottom: 1px solid var(--line);
}
#top .title { font-weight: 600; }
#top .model { color: var(--muted); font-size: 13px; margin-left: 8px; }
#settings-btn, #menu-btn { border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 5px 10px; }
#menu-btn { display: none; margin-right: 8px; }
#log { flex: 1; overflow-y: auto; }
#log .inner { max-width: 780px; margin: 0 auto; padding: 24px 16px 8px; }
.empty { color: var(--muted); text-align: center; margin-top: 18vh; }
.empty h1 { color: var(--text); font-size: 26px; margin-bottom: 4px; }
.msg { margin: 0 0 20px; }
.msg.user { display: flex; justify-content: flex-end; }
.msg.user .bubble {
  background: var(--user); padding: 9px 14px; border-radius: 16px; max-width: 85%;
  white-space: pre-wrap; overflow-wrap: anywhere;
}
.msg.assistant .body { overflow-wrap: anywhere; }
.body p { margin: 0 0 0.8em; } .body p:last-child { margin-bottom: 0; }
.body h1, .body h2, .body h3, .body h4 { margin: 1em 0 0.5em; line-height: 1.3; }
.body h1 { font-size: 1.35em; } .body h2 { font-size: 1.2em; } .body h3, .body h4 { font-size: 1.05em; }
.body ul, .body ol { margin: 0 0 0.8em; padding-left: 1.4em; }
.body code { background: var(--code); border-radius: 4px; padding: 1px 5px; font: 0.88em ui-monospace, SFMono-Regular, Menlo, monospace; }
.body pre { background: var(--code); border: 1px solid var(--line); border-radius: 9px; margin: 0 0 0.9em; overflow: hidden; }
.body pre .bar { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); padding: 5px 10px; border-bottom: 1px solid var(--line); }
.body pre .bar button { border: 0; background: none; color: var(--muted); font-size: 12px; }
.body pre code { display: block; background: none; padding: 10px 12px; overflow-x: auto; white-space: pre; font-size: 13px; }
.body a { color: var(--accent); }
.body blockquote { margin: 0 0 0.8em; padding-left: 12px; border-left: 3px solid var(--line); color: var(--muted); }
.meta { color: var(--muted); font-size: 12px; margin-top: 6px; }
.meta.error { color: var(--danger); }
.cursor::after { content: "▍"; animation: blink 1s steps(1) infinite; color: var(--muted); }
@keyframes blink { 50% { opacity: 0; } }
#composer { flex: none; padding: 10px 16px 18px; }
#composer .box {
  max-width: 780px; margin: 0 auto; display: flex; align-items: flex-end; gap: 8px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 8px 8px 8px 14px;
}
#input {
  flex: 1; border: 0; outline: 0; background: none; color: var(--text); resize: none;
  font: inherit; max-height: 220px; padding: 6px 0;
}
#send {
  border: 0; border-radius: 10px; background: var(--accent); color: var(--accent-text);
  padding: 7px 14px; font-weight: 600;
}
#send.stop { background: var(--danger); }
#send:disabled { opacity: 0.4; cursor: default; }
#composer .hint { max-width: 780px; margin: 6px auto 0; color: var(--muted); font-size: 12px; text-align: center; }
#panel {
  position: fixed; top: 0; right: 0; height: 100%; width: 320px; max-width: 100%; background: var(--panel);
  border-left: 1px solid var(--line); padding: 18px; overflow-y: auto; transform: translateX(100%);
  transition: transform 0.15s ease; z-index: 10;
}
#panel.open { transform: none; }
#panel h2 { font-size: 16px; margin: 0 0 14px; display: flex; justify-content: space-between; }
#panel h2 button { border: 0; background: none; font-size: 18px; color: var(--muted); }
#panel label { display: block; font-size: 13px; color: var(--muted); margin: 12px 0 4px; }
#panel input, #panel textarea {
  width: 100%; padding: 7px 9px; border-radius: 8px; border: 1px solid var(--line);
  background: var(--bg); color: var(--text); font: inherit; font-size: 14px;
}
#panel textarea { min-height: 90px; resize: vertical; }
#panel .row { display: flex; gap: 8px; } #panel .row > div { flex: 1; }
#panel .note { font-size: 12px; color: var(--muted); margin-top: 6px; }
@media (max-width: 720px) {
  #side { position: fixed; left: 0; top: 0; height: 100%; z-index: 9; transform: translateX(-100%); transition: transform 0.15s; }
  #side.open { transform: none; }
  #menu-btn { display: inline-block; }
}
</style>
</head>
<body>
<aside id="side">
  <header><button id="new">＋ New chat</button></header>
  <div id="chats"></div>
  <footer><span id="status-dot"></span><span id="status">connecting…</span></footer>
</aside>
<main>
  <div id="top">
    <div><button id="menu-btn" aria-label="Chats">☰</button><span class="title">Fieldfare</span><span class="model" id="model"></span></div>
    <button id="settings-btn">Settings</button>
  </div>
  <div id="log"><div class="inner" id="messages"></div></div>
  <div id="composer">
    <div class="box">
      <textarea id="input" rows="1" placeholder="Message Gemma…"></textarea>
      <button id="send">Send</button>
    </div>
    <div class="hint">Enter to send · Shift+Enter for a new line · runs locally on this Mac</div>
  </div>
</main>
<div id="panel">
  <h2>Settings <button id="close-panel" aria-label="Close">✕</button></h2>
  <label for="system">System prompt</label>
  <textarea id="system" placeholder="Optional"></textarea>
  <div class="row">
    <div><label for="temperature">Temperature</label><input id="temperature" type="number" min="0" max="2" step="0.1"></div>
    <div><label for="max_tokens">Max tokens</label><input id="max_tokens" type="number" min="1" step="64"></div>
  </div>
  <div class="row">
    <div><label for="top_p">Top-p</label><input id="top_p" type="number" min="0.01" max="1" step="0.05"></div>
    <div><label for="top_k">Top-k</label><input id="top_k" type="number" min="1" max="256" step="1"></div>
  </div>
  <label for="seed">Seed</label>
  <input id="seed" type="number" placeholder="random">
  <div class="note">Temperature 0 is greedy. Any temperature uses the drafter when the server has one; a seed gives the same text with or without it.</div>
  <label for="api_key">API key</label>
  <input id="api_key" type="password" placeholder="only if the server sets --api-key">
</div>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const store = {
  get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} },
};
const defaults = { system: "", temperature: 0.7, max_tokens: 1024, top_p: 0.95, top_k: 64, seed: "", api_key: "" };
const settings = Object.assign({}, defaults, store.get("ff.settings", {}));
let chats = store.get("ff.chats", []);
let current = store.get("ff.current", null);
let modelID = "gemma";
let controller = null;

for (const key of Object.keys(defaults)) {
  const el = $(key);
  el.value = settings[key];
  el.addEventListener("input", () => { settings[key] = el.value; store.set("ff.settings", settings); });
}

function save() { store.set("ff.chats", chats); store.set("ff.current", current); }
function chat() { return chats.find((c) => c.id === current); }
function newChat() {
  const c = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), title: "New chat", messages: [] };
  chats.unshift(c); current = c.id; save(); render();
  $("input").focus();
}

// ---- Markdown (escaped first; only a small, safe subset) ----
function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return "\u0000" + (codes.length - 1) + "\u0000"; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => "<code>" + esc(codes[+i]) + "</code>");
}
function blocks(text) {
  const out = []; const lines = text.split("\n"); let para = []; let list = null;
  const flushPara = () => { if (para.length) { out.push("<p>" + para.map(inline).join("<br>") + "</p>"); para = []; } };
  const flushList = () => { if (list) { out.push("<" + list.tag + ">" + list.items.map((i) => "<li>" + inline(i) + "</li>").join("") + "</" + list.tag + ">"); list = null; } };
  for (const line of lines) {
    let m;
    if (!line.trim()) { flushPara(); flushList(); continue; }
    if ((m = line.match(/^(#{1,4})\s+(.*)$/))) { flushPara(); flushList(); out.push("<h" + m[1].length + ">" + inline(m[2]) + "</h" + m[1].length + ">"); continue; }
    if ((m = line.match(/^\s*[-*+]\s+(.*)$/)) || (m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      const tag = /^\s*\d/.test(line) ? "ol" : "ul";
      flushPara(); if (list && list.tag !== tag) flushList();
      if (!list) list = { tag, items: [] };
      list.items.push(m[1]); continue;
    }
    if ((m = line.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push("<blockquote>" + inline(m[1]) + "</blockquote>"); continue; }
    if (/^(-{3,}|\*{3,})$/.test(line.trim())) { flushPara(); flushList(); out.push("<hr>"); continue; }
    flushList(); para.push(line);
  }
  flushPara(); flushList();
  return out.join("");
}
function markdown(text) {
  const parts = text.split(/```/);
  let html = "";
  parts.forEach((part, i) => {
    if (i % 2 === 0) { html += blocks(part); return; }
    const nl = part.indexOf("\n");
    const lang = nl >= 0 ? part.slice(0, nl).trim() : "";
    const code = nl >= 0 ? part.slice(nl + 1) : part;
    html += '<pre><div class="bar"><span>' + esc(lang || "code") + '</span><button data-copy>Copy</button></div><code>' +
            esc(code.replace(/\n$/, "")) + "</code></pre>";
  });
  return html;
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]"); if (!b) return;
  navigator.clipboard?.writeText(b.closest("pre").querySelector("code").textContent);
  b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1200);
});

// ---- Rendering ----
function render() {
  const list = $("chats"); list.innerHTML = "";
  for (const c of chats) {
    const item = document.createElement("div");
    item.className = "chat-item" + (c.id === current ? " active" : "");
    item.innerHTML = "<span></span><button title='Delete'>✕</button>";
    item.firstChild.textContent = c.title;
    item.onclick = () => { if (controller) return; current = c.id; save(); render(); $("side").classList.remove("open"); };
    item.lastChild.onclick = (e) => {
      e.stopPropagation(); if (controller && c.id === current) return;
      chats = chats.filter((x) => x.id !== c.id); if (current === c.id) current = chats[0]?.id ?? null; save(); render();
    };
    list.appendChild(item);
  }
  const box = $("messages"); box.innerHTML = "";
  const c = chat();
  if (!c || !c.messages.length) {
    box.innerHTML = '<div class="empty"><h1>What can I help with?</h1><div>Gemma 4 26B-A4B, running on this Mac.</div></div>';
    return;
  }
  for (const m of c.messages) box.appendChild(messageEl(m));
  scrollDown(true);
}
function messageEl(m) {
  const el = document.createElement("div");
  el.className = "msg " + m.role;
  if (m.role === "user") {
    const b = document.createElement("div"); b.className = "bubble"; b.textContent = m.content; el.appendChild(b);
  } else {
    el.innerHTML = '<div class="body"></div><div class="meta"></div>';
    fill(el, m);
  }
  return el;
}
function fill(el, m, streaming) {
  const body = el.querySelector(".body");
  body.innerHTML = markdown(m.content);
  body.classList.toggle("cursor", !!streaming);
  const meta = el.querySelector(".meta");
  meta.className = "meta" + (m.error ? " error" : "");
  meta.textContent = m.error ? m.error : (m.stats || "");
}
function scrollDown(force) {
  const log = $("log");
  if (force || log.scrollHeight - log.scrollTop - log.clientHeight < 120) log.scrollTop = log.scrollHeight;
}

// ---- Sending ----
function headers() {
  const h = { "Content-Type": "application/json" };
  if (settings.api_key) h.Authorization = "Bearer " + settings.api_key;
  return h;
}
async function send() {
  if (controller) { controller.abort(); return; }
  const text = $("input").value.trim(); if (!text) return;
  if (!chat()) newChat();
  const c = chat();
  c.messages.push({ role: "user", content: text });
  if (c.title === "New chat") c.title = text.slice(0, 48);
  $("input").value = ""; autosize();
  const reply = { role: "assistant", content: "" };
  c.messages.push(reply); save(); render();
  const el = $("messages").lastChild;
  fill(el, reply, true);

  const messages = [];
  if (settings.system.trim()) messages.push({ role: "system", content: settings.system.trim() });
  for (const m of c.messages.slice(0, -1)) if (!m.error || m.content) messages.push({ role: m.role, content: m.content });
  const body = {
    model: modelID, messages, stream: true, stream_options: { include_usage: true },
    temperature: Number(settings.temperature), top_p: Number(settings.top_p), top_k: Number(settings.top_k),
    max_tokens: Number(settings.max_tokens),
  };
  if (String(settings.seed).trim() !== "") body.seed = Number(settings.seed);

  controller = new AbortController(); setBusy(true);
  const started = performance.now(); let first = 0; let usage = null;
  try {
    const res = await fetch("/v1/chat/completions", { method: "POST", headers: headers(), body: JSON.stringify(body), signal: controller.signal });
    if (!res.ok) {
      let msg = res.status + " " + res.statusText;
      try { msg = (await res.json()).error.message || msg; } catch {}
      throw new Error(msg);
    }
    const reader = res.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const event = buffer.slice(0, cut); buffer = buffer.slice(cut + 2);
        for (const line of event.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim(); if (data === "[DONE]") continue;
          const chunk = JSON.parse(data);
          if (chunk.error) throw new Error(chunk.error.message || "error");
          if (chunk.usage) usage = chunk.usage;
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta) { if (!first) first = performance.now(); reply.content += delta; fill(el, reply, true); scrollDown(); }
        }
      }
    }
    const seconds = (performance.now() - (first || started)) / 1000;
    const tokens = usage?.completion_tokens;
    reply.stats = tokens ? `${tokens} tokens · ${(tokens / Math.max(seconds, 1e-3)).toFixed(1)} tok/s · first token ${((first - started) / 1000).toFixed(2)} s` : "";
  } catch (e) {
    if (e.name === "AbortError") reply.stats = "stopped";
    else reply.error = String(e.message || e);
  } finally {
    controller = null; setBusy(false);
    fill(el, reply, false); save(); poll();
  }
}
function setBusy(busy) {
  const b = $("send"); b.textContent = busy ? "Stop" : "Send"; b.classList.toggle("stop", busy);
}
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 220) + "px"; }

$("input").addEventListener("input", autosize);
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!controller) send(); }
});
$("send").onclick = send;
$("new").onclick = () => { if (!controller) newChat(); };
$("settings-btn").onclick = () => $("panel").classList.toggle("open");
$("close-panel").onclick = () => $("panel").classList.remove("open");
$("menu-btn").onclick = () => $("side").classList.toggle("open");

// ---- Server status ----
async function poll() {
  try {
    const h = await (await fetch("/health")).json();
    $("status-dot").className = h.busy ? "busy" : "ok";
    $("status").textContent = h.busy ? `generating${h.queued ? `, ${h.queued} queued` : ""}` : "ready";
    if (h.model) { modelID = h.model; $("model").textContent = h.model; }
  } catch {
    $("status-dot").className = "down"; $("status").textContent = "server unreachable";
  }
}
poll(); setInterval(poll, 5000);
if (!chat()) current = chats[0]?.id ?? null;
render();
$("input").focus();
</script>
</body>
</html>
"""#
}
