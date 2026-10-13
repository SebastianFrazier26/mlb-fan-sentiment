"""Local labeling page for sentiment-eval CSVs: `fansent label --file <csv>`."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LABELS = ("pos", "neu", "neg", "na")
REQUIRED = ("id", "player", "text")
OUTPUT_COLUMNS = ("sarcasm", "uncertain", "final_label", "reviewed", "reviewed_at")
HOST = "127.0.0.1"
MAX_BODY = 4096


def load_rows(path: Path) -> tuple[list[str], list[dict]]:
    """Read the CSV; missing output columns are appended so a draft file can be labeled as-is."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in REQUIRED if c not in fields]
    if missing:
        raise ValueError(f"{path}: missing required column(s) {', '.join(missing)}")
    for col in OUTPUT_COLUMNS:
        if col not in fields:
            fields.append(col)
    for row in rows:
        for col in fields:
            if row.get(col) is None:
                row[col] = ""
    return fields, rows


def save_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Write to a temp file in the same directory, fsync, then rename over the original.

    Same directory so os.replace is an atomic rename on one filesystem; a crash mid-write
    leaves the previous CSV intact rather than a truncated one.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _flag(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else ""
    if value in (0, 1, "0", "1", ""):
        return "1" if str(value) == "1" else ""
    raise ValueError(f"flag must be boolean, got {value!r}")


class LabelStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fields, self.rows = load_rows(path)
        self._lock = threading.Lock()

    def apply(
        self,
        index: int,
        label: str,
        sarcasm: object = False,
        uncertain: object = False,
        now: dt.datetime | None = None,
    ) -> dict:
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {label!r}")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if not 0 <= index < len(self.rows):
            raise ValueError(f"index {index} out of range")
        stamp = (now or dt.datetime.now(dt.UTC)).isoformat(timespec="seconds")
        update = {
            "final_label": label,
            "sarcasm": _flag(sarcasm) or "0",
            "uncertain": _flag(uncertain) or "0",
            "reviewed": "1",
            "reviewed_at": stamp,
        }
        with self._lock:
            row = self.rows[index]
            previous = {k: row[k] for k in update}
            row.update(update)
            try:
                save_rows(self.path, self.fields, self.rows)
            except BaseException:
                row.update(previous)
                raise
            return dict(row)


def make_handler(store: LabelStore, port: int) -> type[BaseHTTPRequestHandler]:
    allowed_hosts = {f"{HOST}:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _host_ok(self) -> bool:
            # Rejecting foreign Host headers blocks DNS-rebinding pages from reading the rows.
            return self.headers.get("Host", "") in allowed_hosts

        def _send(self, status: HTTPStatus, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, obj: object) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "bad host"})
            if self.path == "/":
                return self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/api/rows":
                return self._json(
                    HTTPStatus.OK,
                    {"file": store.path.name, "labels": list(LABELS), "rows": store.rows},
                )
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "bad host"})
            if self.path != "/api/label":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            # Requiring application/json forces a CORS preflight, so a cross-site form post
            # cannot write labels.
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                return self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "json only"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 < length <= MAX_BODY:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "bad length"})
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise TypeError("body must be an object")
                row = store.apply(
                    body.get("index"),
                    body.get("label"),
                    body.get("sarcasm", False),
                    body.get("uncertain", False),
                )
            except (ValueError, TypeError) as e:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            self._json(HTTPStatus.OK, {"row": row})

    return Handler


def serve(path: Path, port: int = 8765) -> ThreadingHTTPServer:
    store = LabelStore(path)
    server = ThreadingHTTPServer((HOST, port), BaseHTTPRequestHandler)
    server.RequestHandlerClass = make_handler(store, server.server_address[1])
    return server


def run(path: str, port: int, open_browser: bool) -> None:
    server = serve(Path(path), port)
    url = f"http://{HOST}:{server.server_address[1]}/"
    print(f"Labeling {path} at {url}  (Ctrl+C to stop; every decision is saved immediately)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fansent label</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mute:#6b6b66;--card:#fff;--line:#deded8;--hl:#ffe58a;
--pos:#1f8a4c;--neu:#6b6b66;--neg:#c0392b;--na:#7d5ba6;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecece8;--mute:#9a9a94;--card:#1f1f1d;
--line:#34342f;--hl:#6b5a12;--acc:#60a5fa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:820px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;justify-content:space-between}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin:8px 0 16px}
.bar>div{height:100%;background:var(--acc);width:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px}
.meta{color:var(--mute);font-size:14px;display:flex;flex-wrap:wrap;gap:4px 14px}
.player{font-size:20px;font-weight:600;margin:4px 0 8px}
.text{font-size:18px;white-space:pre-wrap;overflow-wrap:anywhere;margin:12px 0}
mark{background:var(--hl);color:inherit;border-radius:3px;padding:0 2px}
.parent{border-left:3px solid var(--line);padding:4px 12px;color:var(--mute);font-size:15px;
white-space:pre-wrap;overflow-wrap:anywhere}
.parent b{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.labels{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:16px 0 8px}
.labels button,.flags button{font:inherit;padding:10px 6px;border-radius:8px;cursor:pointer;
border:2px solid var(--line);background:transparent;color:var(--fg)}
.labels button.on{color:#fff}
.labels button[data-l=pos].on{background:var(--pos);border-color:var(--pos)}
.labels button[data-l=neu].on{background:var(--neu);border-color:var(--neu)}
.labels button[data-l=neg].on{background:var(--neg);border-color:var(--neg)}
.labels button[data-l=na].on{background:var(--na);border-color:var(--na)}
.flags{display:flex;gap:8px;flex-wrap:wrap}.flags button.on{border-color:var(--acc);color:var(--acc)}
.note{color:var(--mute);font-size:14px;margin-top:8px}
.status{font-size:14px;color:var(--mute);min-height:1.5em;margin-top:8px}
.keys{color:var(--mute);font-size:13px;margin-top:16px}
kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:12px}
.err{color:var(--neg)}
</style></head><body><main>
<header><strong id="file">loading…</strong>
<label><input type="checkbox" id="only"> only unreviewed</label></header>
<div class="bar"><div id="prog"></div></div>
<div class="meta"><span id="count"></span><span id="pos"></span></div>
<div class="card" id="card">
 <div class="meta" id="info"></div>
 <div class="player" id="player"></div>
 <div class="parent" id="parent" hidden><b>replying to</b><span id="ptext"></span></div>
 <div class="text" id="text"></div>
 <div class="labels" id="labels">
  <button data-l="pos">1 · pos</button><button data-l="neu">2 · neu</button>
  <button data-l="neg">3 · neg</button><button data-l="na">0 · na</button></div>
 <div class="flags"><button id="fs">s · sarcasm</button><button id="fu">u · uncertain</button></div>
 <div class="note" id="note"></div>
 <div class="status" id="status"></div>
</div>
<div class="keys"><kbd>1</kbd> pos <kbd>2</kbd> neu <kbd>3</kbd> neg <kbd>0</kbd> na ·
<kbd>s</kbd> sarcasm <kbd>u</kbd> uncertain · <kbd>Enter</kbd>/<kbd>→</kbd> accept &amp; next ·
<kbd>←</kbd> back. Sentiment toward the named player; injury concern/sympathy = neu;
na = not about the player.</div>
</main><script>
"use strict";
let rows=[],order=[],pos=0,cur=null;
const $=id=>document.getElementById(id);
const KEY={"1":"pos","2":"neu","3":"neg","0":"na"};
const isOn=v=>String(v)==="1";
const fold=s=>Array.from(s,c=>c.normalize("NFD")[0]||c).join("").toLowerCase();
function esc(s){return s.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function highlight(text,name){
  // Fold accents per code point so match offsets in the folded copy map back to the original.
  const chars=Array.from(text),f=chars.map(c=>(c.normalize("NFD")[0]||c).toLowerCase());
  const parts=[name,...name.split(/\s+/)].map(fold).filter(p=>p.length>=3);
  const hit=new Array(chars.length).fill(false),hay=f.join("");
  const idx=[];let o=0;for(const c of f){idx.push(o);o+=c.length}
  for(const p of parts){let i=hay.indexOf(p);
    while(i>=0){const b=i===0||!/\p{L}/u.test(hay[i-1]),e=i+p.length>=hay.length||!/\p{L}/u.test(hay[i+p.length]);
      if(b&&e)for(let k=0;k<chars.length;k++)if(idx[k]>=i&&idx[k]<i+p.length)hit[k]=true;
      i=hay.indexOf(p,i+1)}}
  let out="",open=false;
  chars.forEach((c,k)=>{if(hit[k]&&!open){out+="<mark>";open=true}
    if(!hit[k]&&open){out+="</mark>";open=false}out+=esc(c)});
  return out+(open?"</mark>":"");
}
function buildOrder(){
  const only=$("only").checked,here=order[pos];
  order=rows.map((r,i)=>i).filter(i=>!only||!isOn(rows[i].reviewed));
  if(!order.length)order=only?[]:[0];
  const keep=order.indexOf(here);pos=keep>=0?keep:0;
}
function render(){
  const done=rows.filter(r=>isOn(r.reviewed)).length;
  $("prog").style.width=(rows.length?100*done/rows.length:0)+"%";
  $("count").textContent=`${done} / ${rows.length} reviewed`;
  if(!order.length){$("card").hidden=true;$("pos").textContent="nothing left to review";return}
  $("card").hidden=false;
  const i=order[pos],r=rows[i];
  cur={i,label:r.final_label||r.draft_label||"",s:isOn(r.sarcasm),u:isOn(r.uncertain)};
  $("pos").textContent=`item ${pos+1} of ${order.length}${$("only").checked?" unreviewed":""} · row ${i+1}`;
  $("info").textContent=[r.subreddit&&"r/"+r.subreddit,r.date,r.subset,r.attribution,
    r.draft_label&&"draft: "+r.draft_label,isOn(r.reviewed)&&"reviewed "+r.reviewed_at].filter(Boolean).join(" · ");
  $("player").textContent=r.player||"";
  $("text").innerHTML=highlight(r.text||"",r.player||"");
  const pt=r.parent_text||"";$("parent").hidden=!pt;$("ptext").innerHTML=pt?highlight(pt,r.player||""):"";
  $("note").textContent=r.note?"note: "+r.note:"";
  $("status").textContent="";$("status").className="status";
  paint();
}
function paint(){
  document.querySelectorAll("#labels button").forEach(b=>b.classList.toggle("on",b.dataset.l===cur.label));
  $("fs").classList.toggle("on",cur.s);$("fu").classList.toggle("on",cur.u);
}
async function accept(){
  if(!cur)return;
  if(!cur.label){fail("pick a label first");return}
  try{
    const res=await fetch("/api/label",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({index:cur.i,label:cur.label,sarcasm:cur.s,uncertain:cur.u})});
    const j=await res.json();if(!res.ok)throw new Error(j.error||res.status);
    rows[cur.i]=j.row;
  }catch(e){fail("save failed: "+e.message);return}
  if(pos<order.length-1)pos++;
  render();
  if(pos===order.length-1&&isOn(rows[order[pos]].reviewed))$("status").textContent="last item";
}
function fail(m){$("status").textContent=m;$("status").className="status err"}
function back(){if(pos>0){pos--;render()}}
document.addEventListener("keydown",e=>{
  if(e.metaKey||e.ctrlKey||e.altKey||e.target.tagName==="INPUT"||!cur)return;
  const k=e.key.toLowerCase();
  if(k in KEY){cur.label=KEY[k];paint()}
  else if(k==="s"){cur.s=!cur.s;paint()}
  else if(k==="u"){cur.u=!cur.u;paint()}
  else if(k==="enter"||k==="arrowright"){e.preventDefault();accept()}
  else if(k==="arrowleft"){e.preventDefault();back()}
});
document.querySelectorAll("#labels button").forEach(b=>b.onclick=()=>{cur.label=b.dataset.l;paint()});
$("fs").onclick=()=>{cur.s=!cur.s;paint()};$("fu").onclick=()=>{cur.u=!cur.u;paint()};
$("only").onchange=()=>{buildOrder();render();$("only").blur()};
fetch("/api/rows").then(r=>r.json()).then(j=>{
  rows=j.rows;$("file").textContent=j.file;document.title="label · "+j.file;
  $("only").checked=true;order=[];buildOrder();render();
}).catch(e=>fail("load failed: "+e.message));
</script></body></html>
"""
