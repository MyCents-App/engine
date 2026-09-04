"""Eyes-on review of the pipeline, one photo at a time.

For looking at real output and judging it yourself -- not scoring. (For scored
evaluation against gold labels, that is eval_metrics.py.)

It walks a folder of receipt photos in order, sends ONE to the running engine,
prints the OCR text beside the extracted fields, and waits for you before
moving to the next. Every response is also written to <output>/<stem>.json so a
run can be re-read later without re-running the GPU.

Needs the engine up (ocr_service.py on :8001, app.api on :8000) -- see RUN.md.
The tunnel is not needed; this talks to localhost.

    .venv\\Scripts\\python.exe review_photos.py                 # all 20, one by one
    .venv\\Scripts\\python.exe review_photos.py --photo 7       # just photo-7
    .venv\\Scripts\\python.exe review_photos.py --start 5       # resume at the 5th
    .venv\\Scripts\\python.exe review_photos.py --no-pause      # run straight through

Stdlib only, so it runs under either venv or a bare python.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT.parent / "SuryaOCR" / "data" / "input"
DEFAULT_OUTPUT = ROOT.parent / "SuryaOCR" / "data" / "review"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                 ".webp": "image/webp", ".heic": "image/heic", ".heif": "image/heif",
                 ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff"}

RULE = "=" * 72
THIN = "-" * 72


def api_key(env_path: Path) -> str:
    """ENGINE_API_KEY, from the environment or train/.env.

    The same file run_demo.bat reads, so the key is defined in one place only.
    """
    if os.environ.get("ENGINE_API_KEY"):
        return os.environ["ENGINE_API_KEY"]
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == "ENGINE_API_KEY":
                return value.strip()
    return ""


def natural_key(path: Path):
    """photo-2 before photo-10 -- plain sorting puts photo-10 second."""
    digits = "".join(c for c in path.stem if c.isdigit())
    return (int(digits) if digits else 0, path.stem)


def post(url: str, body: bytes, headers: dict, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"detail": raw[:500]}


def check_ready(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/ready", timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            body = {}
        print(f"Engine not ready ({exc.code}): {body}")
        print("The model is probably still loading -- window 2 logs "
              '"warm-up generation done" when it is up.')
        return False
    except OSError as exc:
        print(f"Cannot reach the engine at {base}: {exc}")
        print("Start it first -- see RUN.md.")
        return False
    print(f"Engine ready  model={body.get('model_loaded')}  "
          f"ocr={body.get('ocr_reachable')}  auth={body.get('auth_enabled')}")
    return True


def money(value) -> str:
    return "-" if value in (None, "") else str(value)


def show(result: dict, ocr_lines: int, show_ocr: bool) -> None:
    if show_ocr:
        texts = result.get("ocrTexts") or [""]
        print(f"{THIN}\nOCR text ({ocr_lines} lines)\n{THIN}")
        print(texts[0].rstrip() or "(empty)")

    print(f"{THIN}\nExtraction\n{THIN}")
    if not result.get("ok", False):
        print(f"  FAILED: {result.get('error', 'unknown error')}")
    else:
        print(f"  shop       {result.get('shopName') or '(none)'}")
        print(f"  date       {result.get('receiptDate') or '(none -- app defaults to today)'}")
        items = result.get("items") or []
        print(f"  items      {len(items)}")
        items_sum = 0.0
        for item in items:
            price = item.get("price")
            try:
                items_sum += float(price)
            except (TypeError, ValueError):
                pass
            print(f"      {str(item.get('name') or '?'):<44} {money(price):>10}")
        print(f"      {'items sum':>44} {items_sum:>10.2f}")
        print(f"  tax        {money(result.get('taxAmount'))}")
        print(f"  discount   {money(result.get('basketDiscount'))}   (already spread across items)")
        print(f"  TOTAL      {money(result.get('totalAmount'))}")
        flag = "OK" if result.get("reconciles") else "!!"
        print(f"  {flag} reconcile {result.get('reconcileStatus')}  "
              f"(reconciles={result.get('reconciles')})")

    eng = result.get("engine") or {}
    print(f"  timing     ocr {eng.get('ocr_seconds', '?')}s   model "
          f"{eng.get('model_seconds', '?')}s   total {eng.get('total_seconds', '?')}s"
          f"   prompt {eng.get('prompt_tokens', '?')}/{eng.get('token_budget', '?')} tok")


def run_one(base: str, key: str, path: Path, out_dir: Path, timeout: float,
            show_ocr: bool) -> dict:
    payload = path.read_bytes()
    ctype = CONTENT_TYPES.get(path.suffix.lower(), "image/jpeg")
    headers = {"Content-Type": ctype, "Content-Length": str(len(payload))}
    if key:
        headers["X-API-Key"] = key

    print(f"  sending {len(payload) / 1024:.0f} KB ... (4-8s)", flush=True)
    code, result = post(f"{base}/v1/extract", payload, headers, timeout)

    if code != 200:
        print(f"  HTTP {code}: {result.get('detail', result)}")
        return result

    lines = len((result.get("ocrTexts") or [""])[0].splitlines())
    show(result, lines, show_ocr)

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{path.stem}.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  saved      {dest}")
    return result


def main(argv: list[str] | None = None) -> int:
    # Thai item names are unreadable through the console's legacy codepage.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="folder of photos")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help="where responses are written")
    ap.add_argument("--base", default=os.environ.get("ENGINE_URL", "http://localhost:8000"))
    ap.add_argument("--photo", help="one photo: a number (7 -> photo-7) or a filename")
    ap.add_argument("--start", type=int, default=1, help="resume the walk at the Nth photo")
    ap.add_argument("--no-pause", action="store_true", help="do not wait between photos")
    ap.add_argument("--no-ocr-text", action="store_true", help="hide the raw OCR text")
    ap.add_argument("--open", action="store_true",
                    help="open each photo in the image viewer so you can compare (Windows)")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-request seconds")
    args = ap.parse_args(argv)

    if not args.input.is_dir():
        print(f"No such folder: {args.input}")
        return 1

    photos = sorted((p for p in args.input.iterdir()
                     if p.is_file() and p.suffix.lower() in EXTENSIONS), key=natural_key)
    if args.photo:
        wanted = f"photo-{args.photo}" if args.photo.isdigit() else args.photo
        photos = [p for p in photos if p.stem == wanted or p.name == wanted]
        if not photos:
            print(f"No photo matching {args.photo!r} in {args.input}")
            return 1
    else:
        photos = photos[max(args.start - 1, 0):]

    if not photos:
        print(f"No images in {args.input}")
        return 1

    if not check_ready(args.base):
        return 1

    key = api_key(ROOT / ".env")
    if not key:
        print("No ENGINE_API_KEY found in train/.env -- sending unauthenticated; "
              "expect 401 if the engine has auth on.")

    failures = []
    for index, path in enumerate(photos, start=1):
        print(f"\n{RULE}\n [{index}/{len(photos)}]  {path.name}\n{RULE}")
        if args.open and hasattr(os, "startfile"):
            try:
                os.startfile(path)  # noqa: S606 -- a local file the user chose
            except OSError as exc:
                print(f"  (could not open the image: {exc})")
        try:
            result = run_one(args.base, key, path, args.output, args.timeout,
                             not args.no_ocr_text)
        except OSError as exc:
            print(f"  request failed: {exc}")
            failures.append(path.name)
            continue
        if not result.get("ok"):
            failures.append(path.name)

        if args.no_pause or index == len(photos):
            continue
        prompt = "\n  [Enter] next   r = redo this one   q = quit  > "
        try:
            answer = input(prompt).strip().lower()
            while answer == "r":
                run_one(args.base, key, path, args.output, args.timeout,
                        not args.no_ocr_text)
                answer = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if answer == "q":
            break

    print(f"\n{RULE}\nReviewed {len(photos)} photo(s). Responses in {args.output}")
    if failures:
        print(f"Returned ok=false or errored: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
