"""LOCAL DEVELOPMENT TOOL — not the deployed service.

Serves a phone-facing HTML page for eyeballing the pipeline over LAN. The
deployed engine is API-only (`app/api.py`) and this file is deliberately not
built into either container image; see DEPLOY.md.

Kept because it is genuinely useful for a quick manual check on the GPU box.
For anything programmatic use `POST /v1/extract` instead.

Demo server: phone photo -> Surya OCR -> fine-tuned Qwen3.5-2B -> structured JSON.

Serves a mobile web page on the LAN. Open it on a phone, take a photo of a receipt, and get
back the extracted merchant / items / prices / total.

Shape of the thing:
    phone browser  --image bytes-->  THIS (GPU, extraction model)
                                       |  --image bytes-->  ocr_service.py (CPU, Surya)
                                       |  <--OCR text-----
                                       v
                                     JSON back to the phone

Stdlib only, on purpose: no FastAPI/uvicorn install into a venv that took real effort to get
working. The browser POSTs the raw image bytes as the request body, so there is no multipart
form to parse.

The page uses <input type="file" capture="environment"> rather than getUserMedia, because
browsers only grant camera access over HTTPS or localhost -- and this is served over plain HTTP
on a LAN address. The file input still opens the camera directly on both iOS and Android.

Run (from this project, using ITS venv), AFTER ocr_service.py is up:

    .venv\\Scripts\\python.exe demo_server.py

Then open http://<this-pc-lan-ip>:8000 on a phone on the same Wi-Fi.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from unsloth import FastLanguageModel  # noqa: I001  (must precede transformers imports)

import torch

import postprocess
from prompts import build_messages

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT / "checkpoints" / "qwen3.5-2b-qlora" / "checkpoint-550"
OCR_URL = os.environ.get("OCR_URL", "http://127.0.0.1:8001/ocr")
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 768
MAX_BYTES = 25 * 1024 * 1024

_model = None
_tokenizer = None
_lock = Lock()  # generation is not safe to run concurrently on one model instance


def load_model(checkpoint: str):
    global _model, _tokenizer
    print(f"Loading {checkpoint} in 4-bit on GPU ...", flush=True)
    t0 = time.time()
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint).replace("\\", "/"),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(_model)
    print(f"Model ready in {time.time() - t0:.1f}s "
          f"({torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB VRAM)", flush=True)


def _text_tokenizer():
    """Unsloth hands back a multimodal PROCESSOR for Qwen3.5 (it is a VL model); calling it
    positionally routes the string into image preprocessing and dies with a confusing base64
    error. Tokenize through the inner text tokenizer, use the processor only for templates."""
    return getattr(_tokenizer, "tokenizer", _tokenizer)


def extract(ocr_text: str) -> tuple[dict | None, str, float]:
    """OCR text -> (parsed prediction or None, raw completion, seconds)."""
    messages = build_messages(ocr_text)  # identical prompt to training and evaluation
    try:
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inner = _text_tokenizer()
    inputs = inner(text=prompt, return_tensors="pt").to(_model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy, same as every number in the eval reports
            pad_token_id=inner.pad_token_id or inner.eos_token_id,
        )
    elapsed = time.time() - t0

    raw = inner.decode(out[0][input_len:], skip_special_tokens=True)
    pred, _err = postprocess.extract_json(raw)
    if pred is not None:
        # Code-layer job per task.md Section 4: reformat prices to NN.DD. Never changes a value.
        pred = postprocess.normalize_prediction(pred)
    return pred, raw, elapsed


def result_payload(ocr_text: str, apply_discount: bool = False) -> tuple[int, dict]:
    """Run the model on OCR text and build the response body.

    Shared by /extract (which gets its text from Surya) and /extract-text (which gets it
    pasted in), so both paths run byte-identical prompting, decoding and post-processing --
    a result you see on the paste page is exactly what the photo path would produce.
    """
    if not ocr_text.strip():
        return 400, {"error": "No OCR text provided."}
    try:
        with _lock:
            pred, raw, model_s = extract(ocr_text)
    except Exception as exc:  # noqa: BLE001
        return 500, {"error": f"Extraction failed: {type(exc).__name__}: {exc}"}

    # Reconcile BEFORE the discount is spread: afterwards items sum to the total by
    # construction, so a post-spread check would always pass and tell you nothing.
    recon = postprocess.reconcile(pred, ocr_text) if pred else None

    final = pred
    discount_applied = False
    if apply_discount and pred is not None:
        final = postprocess.apply_basket_discount(pred)
        discount_applied = final is not pred

    return 200, {
        "prediction": final,
        "model_prediction": pred if discount_applied else None,  # pre-spread view, for debugging
        "discount_applied": discount_applied,
        "raw_completion": raw,
        "model_seconds": round(model_s, 2),
        "reconciles": bool(recon.passes) if recon else False,
        "reconcile_status": recon.status if recon else "unparseable",
    }


def run_ocr(image_bytes: bytes, content_type: str) -> dict:
    req = urllib.request.Request(
        OCR_URL, data=image_bytes, method="POST",
        headers={"Content-Type": content_type or "image/jpeg"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Receipt Scanner</title>
<style>
  :root{
    --bg:#f5f6f8; --card:#fff; --ink:#12151a; --soft:#5d6672; --mute:#8b94a1;
    --rule:#e3e6ec; --accent:#2a78d6; --good:#1baf7a; --warn:#eb6834;
  }
  @media (prefers-color-scheme:dark){:root{
    --bg:#0e1116; --card:#171b22; --ink:#e9ecf1; --soft:#a3acba; --mute:#79828f;
    --rule:#262c35; --accent:#3987e5; --good:#199e70; --warn:#d95926;
  }}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    padding:max(16px,env(safe-area-inset-top)) 16px calc(24px + env(safe-area-inset-bottom));
    -webkit-font-smoothing:antialiased}
  .wrap{max-width:520px;margin:0 auto;display:flex;flex-direction:column;gap:16px}
  h1{font-size:21px;margin:0;letter-spacing:-.01em}
  .sub{color:var(--soft);font-size:13.5px;margin:0}
  .card{background:var(--card);border:1px solid var(--rule);border-radius:14px;padding:16px}
  label.shoot{display:block;background:var(--accent);color:#fff;text-align:center;
    padding:17px;border-radius:14px;font-weight:600;font-size:17px;cursor:pointer;
    user-select:none;-webkit-tap-highlight-color:transparent}
  label.shoot:active{filter:brightness(.92)}
  input[type=file]{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}
  #preview{width:100%;border-radius:10px;display:none;margin-bottom:12px}
  .status{display:flex;align-items:center;gap:10px;color:var(--soft);font-size:14px}
  .spin{width:15px;height:15px;border:2px solid var(--rule);border-top-color:var(--accent);
    border-radius:50%;animation:sp .8s linear infinite;flex:none}
  @keyframes sp{to{transform:rotate(360deg)}}
  @media (prefers-reduced-motion:reduce){.spin{animation-duration:2.5s}}
  .shop{font-size:19px;font-weight:700;letter-spacing:-.01em;margin-bottom:2px}
  table{width:100%;border-collapse:collapse;font-size:15px;margin-top:10px}
  td{padding:8px 0;border-bottom:1px solid var(--rule);vertical-align:top}
  td.p{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;padding-left:12px;
    font-family:ui-monospace,Menlo,Consolas,monospace;font-size:14px}
  tr.tot td{border-bottom:none;font-weight:700;padding-top:12px;font-size:16.5px}
  tr.disc td{color:var(--warn)}
  .pill{display:inline-flex;align-items:center;gap:5px;font-size:11.5px;font-weight:600;
    padding:3px 9px;border-radius:99px;border:1px solid var(--rule);color:var(--soft)}
  .pill.ok{color:var(--good);border-color:color-mix(in srgb,var(--good) 40%,transparent)}
  .pill.no{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 40%,transparent)}
  .meta{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
  details{margin-top:12px}
  summary{font-size:13px;color:var(--mute);cursor:pointer}
  pre{white-space:pre-wrap;word-break:break-word;font-size:11.5px;color:var(--soft);
    background:var(--bg);padding:11px;border-radius:8px;max-height:250px;overflow:auto;
    border:1px solid var(--rule);margin:8px 0 0}
  .err{color:var(--warn);font-size:14px}
</style></head><body>
<div class="wrap">
  <div>
    <h1>Receipt Scanner</h1>
    <p class="sub">Qwen3.5-2B fine-tune &middot; Surya OCR</p>
  </div>

  <label class="shoot" for="f">Take a photo</label>
  <input id="f" type="file" accept="image/*" capture="environment">

  <div id="out"></div>
</div>
<script>
const f = document.getElementById('f'), out = document.getElementById('out');
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

f.addEventListener('change', async () => {
  const file = f.files[0]; if (!file) return;
  const url = URL.createObjectURL(file);
  out.innerHTML = `<div class="card"><img id="preview" src="${url}" style="display:block">
    <div class="status"><div class="spin"></div><span id="st">Reading the receipt&hellip;</span></div></div>`;
  const st = document.getElementById('st');
  const t0 = performance.now();
  const tick = setInterval(() => {
    st.textContent = `Reading the receipt… ${((performance.now()-t0)/1000).toFixed(0)}s`;
  }, 500);

  try {
    const r = await fetch('/extract', {
      method: 'POST', body: file,
      headers: {'Content-Type': file.type || 'image/jpeg'}
    });
    clearInterval(tick);
    const d = await r.json();
    if (!r.ok || d.error) { render_err(url, d.error || ('HTTP ' + r.status)); return; }
    render(url, d);
  } catch (e) {
    clearInterval(tick);
    render_err(url, e.message);
  } finally {
    f.value = '';   // let the same photo be retaken
  }
});

function render_err(url, msg) {
  out.innerHTML = `<div class="card"><img src="${url}" style="width:100%;border-radius:10px">
    <p class="err" style="margin-top:12px">${esc(msg)}</p></div>`;
}

function render(url, d) {
  const p = d.prediction;
  if (!p) { render_err(url, 'The model did not return valid JSON.'); return; }
  const rows = (p.items || []).map(i =>
    `<tr><td>${esc(i.name)}</td><td class="p">${esc(i.price)}</td></tr>`).join('');
  const disc = p['basket-wide_discount']
    ? `<tr class="disc"><td>Basket discount</td><td class="p">&minus;${esc(p['basket-wide_discount'])}</td></tr>` : '';
  const rec = d.reconciles
    ? '<span class="pill ok">&#10003; Reconciles</span>'
    : `<span class="pill no">&#9888; ${esc(d.reconcile_status)}</span>`;

  out.innerHTML = `<div class="card">
    <img src="${url}" style="width:100%;border-radius:10px;margin-bottom:14px">
    <div class="shop">${esc(p.shop_name || '(no merchant)')}</div>
    <table>${rows}${disc}
      <tr class="tot"><td>Total</td><td class="p">${esc(p.total_price || '')}</td></tr>
    </table>
    <div class="meta">
      ${rec}
      <span class="pill">${(p.items || []).length} items</span>
      <span class="pill">OCR ${d.ocr_seconds}s</span>
      <span class="pill">model ${d.model_seconds}s</span>
      <span class="pill">total ${d.total_seconds}s</span>
    </div>
    <details><summary>OCR text &amp; raw JSON</summary>
      <pre>${esc(d.ocr_text)}</pre>
      <pre>${esc(JSON.stringify(p, null, 2))}</pre>
    </details>
  </div>`;
}
</script></body></html>"""


TEXT_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paste OCR Text</title>
<style>
  :root{
    --bg:#f5f6f8; --card:#fff; --ink:#12151a; --soft:#5d6672; --mute:#8b94a1;
    --rule:#e3e6ec; --accent:#2a78d6; --good:#1baf7a; --warn:#eb6834;
    --mono:ui-monospace,"Cascadia Mono",Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme:dark){:root{
    --bg:#0e1116; --card:#171b22; --ink:#e9ecf1; --soft:#a3acba; --mute:#79828f;
    --rule:#262c35; --accent:#3987e5; --good:#199e70; --warn:#d95926;
  }}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:24px 18px 64px}
  .wrap{max-width:1080px;margin:0 auto;display:flex;flex-direction:column;gap:18px}
  header{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap}
  h1{font-size:21px;margin:0;letter-spacing:-.01em}
  .sub{color:var(--soft);font-size:13.5px;margin:3px 0 0}
  a{color:var(--accent);font-size:13.5px}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:860px){.cols{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:16px;
    display:flex;flex-direction:column;gap:12px;min-width:0}
  .lbl{font-size:11.5px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--mute);
    font-family:var(--mono)}
  textarea{width:100%;min-height:340px;resize:vertical;background:var(--bg);color:var(--ink);
    border:1px solid var(--rule);border-radius:9px;padding:12px;font-family:var(--mono);
    font-size:12.5px;line-height:1.5}
  textarea:focus{outline:2px solid var(--accent);outline-offset:1px}
  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  button{background:var(--accent);color:#fff;border:0;border-radius:9px;padding:11px 20px;
    font-size:14.5px;font-weight:600;cursor:pointer}
  button:disabled{opacity:.55;cursor:default}
  button.ghost{background:transparent;color:var(--soft);border:1px solid var(--rule);padding:9px 14px;
    font-weight:500;font-size:13px}
  .hint{color:var(--mute);font-size:12.5px}
  .shop{font-size:18px;font-weight:700;letter-spacing:-.01em}
  table{width:100%;border-collapse:collapse;font-size:14px}
  td{padding:7px 0;border-bottom:1px solid var(--rule);vertical-align:top}
  td.p{text-align:right;font-family:var(--mono);font-size:13px;font-variant-numeric:tabular-nums;
    white-space:nowrap;padding-left:14px}
  tr.tot td{border-bottom:none;font-weight:700;padding-top:11px;font-size:15.5px}
  tr.disc td{color:var(--warn)}
  .pill{display:inline-flex;gap:5px;font-size:11.5px;font-weight:600;padding:3px 9px;
    border-radius:99px;border:1px solid var(--rule);color:var(--soft);font-family:var(--mono)}
  .pill.ok{color:var(--good);border-color:color-mix(in srgb,var(--good) 40%,transparent)}
  .pill.no{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 40%,transparent)}
  pre{white-space:pre-wrap;word-break:break-word;font-family:var(--mono);font-size:12px;
    background:var(--bg);border:1px solid var(--rule);border-radius:9px;padding:12px;margin:0;
    max-height:300px;overflow:auto;color:var(--soft)}
  .err{color:var(--warn)}
  .placeholder{color:var(--mute);font-size:13.5px}
</style></head><body>
<div class="wrap">
  <header>
    <div>
      <h1>Paste OCR Text</h1>
      <p class="sub">Runs the identical prompt and decoding as the photo pipeline &mdash; the model is already loaded.</p>
    </div>
    <a href="/">&larr; photo capture page</a>
  </header>

  <div class="cols">
    <div class="card">
      <div class="lbl">Surya output</div>
      <textarea id="t" spellcheck="false" placeholder="Paste the contents of text.txt here&#10;(one receipt), then press Extract."></textarea>
      <div class="row">
        <button id="go">Extract</button>
        <button id="clr" class="ghost">Clear</button>
        <label class="hint" style="display:flex;gap:6px;align-items:center;cursor:pointer">
          <input type="checkbox" id="ad"> deduct basket discount into items
        </label>
      </div>
      <span class="hint" id="hint">Ctrl+Enter to run</span>
    </div>

    <div class="card">
      <div class="lbl">Result</div>
      <div id="out"><p class="placeholder">Nothing yet.</p></div>
    </div>
  </div>
</div>
<script>
const t=document.getElementById('t'), out=document.getElementById('out'),
      go=document.getElementById('go'), hint=document.getElementById('hint');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

document.getElementById('clr').onclick=()=>{t.value='';out.innerHTML='<p class="placeholder">Nothing yet.</p>';t.focus();};
go.onclick=run;
t.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')run();});

async function run(){
  const text=t.value;
  if(!text.trim()){out.innerHTML='<p class="err">Paste some OCR text first.</p>';return;}
  go.disabled=true; hint.textContent='running…';
  out.innerHTML='<p class="placeholder">Extracting…</p>';
  try{
    const r=await fetch('/extract-text',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text, apply_discount:document.getElementById('ad').checked})});
    const d=await r.json();
    if(!r.ok||d.error){out.innerHTML='<p class="err">'+esc(d.error||('HTTP '+r.status))+'</p>';return;}
    render(d);
  }catch(e){out.innerHTML='<p class="err">'+esc(e.message)+'</p>';}
  finally{go.disabled=false; hint.textContent='Ctrl+Enter to run';}
}

function render(d){
  const p=d.prediction;
  if(!p){out.innerHTML='<p class="err">The model did not return valid JSON.</p>'
      +'<pre>'+esc(d.raw_completion||'')+'</pre>';return;}
  const rows=(p.items||[]).map(i=>'<tr><td>'+esc(i.name)+'</td><td class="p">'+esc(i.price)+'</td></tr>').join('');
  const disc=p['basket-wide_discount']
    ?'<tr class="disc"><td>Basket discount</td><td class="p">&minus;'+esc(p['basket-wide_discount'])+'</td></tr>':'';
  const rec=d.reconciles?'<span class="pill ok">&#10003; Reconciles</span>'
    :'<span class="pill no">&#9888; '+esc(d.reconcile_status)+'</span>';
  const spread=d.discount_applied?'<span class="pill ok">discount deducted into items</span>':'';
  out.innerHTML='<div class="shop">'+esc(p.shop_name||'(no merchant)')+'</div>'
    +'<table>'+rows+disc+'<tr class="tot"><td>Total</td><td class="p">'+esc(p.total_price||'')+'</td></tr></table>'
    +'<div class="row" style="margin-top:12px">'+rec+spread
      +'<span class="pill">'+(p.items||[]).length+' items</span>'
      +'<span class="pill">'+d.model_seconds+'s</span>'
      +'<button class="ghost" id="cp">Copy JSON</button></div>'
    +'<pre style="margin-top:12px">'+esc(JSON.stringify(p,null,2))+'</pre>';
  document.getElementById('cp').onclick=async e=>{
    try{await navigator.clipboard.writeText(JSON.stringify(p,null,2));
        e.target.textContent='Copied';setTimeout(()=>e.target.textContent='Copy JSON',1200);}
    catch{e.target.textContent='Copy failed';}
  };
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path in ("/text", "/text/"):
            self._send(200, TEXT_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path, _, query = self.path.partition("?")
        self._apply_discount = "apply_discount=1" in query
        if path == "/extract-text":
            self._handle_extract_text()
            return
        if path != "/extract":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json(400, {"error": "No image received."})
            return
        if length > MAX_BYTES:
            self._json(413, {"error": "That photo is too large. Try again."})
            return
        image = self.rfile.read(length)
        ctype = (self.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip()

        t_start = time.time()
        try:
            ocr = run_ocr(image, ctype)
        except urllib.error.URLError as exc:
            self._json(503, {"error": f"Cannot reach the OCR service at {OCR_URL}. "
                                      f"Is ocr_service.py running? ({exc.reason})"})
            return
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": f"OCR failed: {exc}"})
            return

        ocr_text = ocr.get("text", "")
        if not ocr_text.strip():
            self._json(200, {"prediction": None, "error": "No text found in that photo.",
                             "ocr_text": "", "ocr_seconds": ocr.get("ocr_seconds", 0)})
            return

        code, payload = result_payload(ocr_text, apply_discount=self._apply_discount)
        if code == 200:
            payload.update({
                "ocr_text": ocr_text,
                "ocr_lines": ocr.get("num_lines"),
                "ocr_confidence": ocr.get("avg_confidence"),
                "ocr_seconds": ocr.get("ocr_seconds", 0),
                "total_seconds": round(time.time() - t_start, 2),
            })
        self._json(code, payload)

    def _handle_extract_text(self):
        """Same model path as /extract, but the OCR text is supplied by the caller.

        Useful for testing an integration without paying ~20s of OCR on every iteration, and
        for replaying a saved text.txt against the model.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 2 * 1024 * 1024:
            self._json(400, {"error": "Body must be JSON: {\"text\": \"...\"} (max 2MB)."})
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": f"Body is not valid JSON: {exc}"})
            return

        text = body.get("text")
        if not isinstance(text, str):
            self._json(400, {"error": 'Expected a JSON object with a string "text" field.'})
            return

        # Accept the flag from either the body or the query string, so both endpoints behave
        # the same way and a caller cannot silently get un-discounted output by using the form
        # that happens not to be wired up.
        t_start = time.time()
        code, payload = result_payload(
            text, apply_discount=bool(body.get("apply_discount")) or self._apply_discount
        )
        if code == 200:
            payload["total_seconds"] = round(time.time() - t_start, 2)
        self._json(code, payload)

    def log_message(self, fmt, *args):
        sys.stderr.write("  [app] " + (fmt % args) + "\n")

    def log_error(self, fmt, *args):
        """Keep real errors visible, drop one benign line.

        Browsers open speculative keep-alive connections and drop them without sending a
        request. That raises TimeoutError inside the stdlib's handle_one_request, which logs
        it through log_error -- and since Python 3.10 `socket.timeout IS TimeoutError`, it
        prints as "Request timed out". No request is affected (they all still return 200), but
        a screenful of it looks exactly like a server that is failing. Suppress that one
        message only; everything else still gets through, marked as an error.
        """
        msg = fmt % args
        if "Request timed out" in msg or "Connection reset" in msg:
            return
        sys.stderr.write("  [app] ERROR " + msg + "\n")


def lan_ip() -> str:
    """The address a phone on the same Wi-Fi should open. Uses a UDP socket to pick the
    interface that actually routes outward -- hostname lookup often returns 127.0.0.1."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 so the phone can reach it")
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"FATAL: checkpoint not found: {args.checkpoint}")
        return 1

    load_model(args.checkpoint)

    try:
        health = urllib.request.urlopen(OCR_URL.replace("/ocr", "/health"), timeout=5)
        print(f"OCR service: {json.loads(health.read().decode())}", flush=True)
    except Exception:  # noqa: BLE001
        print(f"WARNING: OCR service not reachable at {OCR_URL}. Start ocr_service.py first.",
              flush=True)

    ip = lan_ip()
    print("\n" + "=" * 58)
    print(f"  Open this on your phone:   http://{ip}:{args.port}")
    print(f"  (same Wi-Fi; allow Python through the Windows firewall)")
    print("=" * 58 + "\n", flush=True)

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
