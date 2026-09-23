"""The multi-photo contract of /v1/extract and /v1/extract-text.

No GPU: the model and the OCR sidecar are both replaced with stubs, so what is
under test is the HTTP layer's own behaviour -- how many photos it accepts, in
what order it OCRs them, that it makes exactly ONE model call for a receipt
however many photos it arrived as, and the shape of what comes back.

Needs fastapi + httpx, which the model venv already has. If a bare test venv
does not, the whole module skips rather than failing.

    .venv-test\\Scripts\\python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("fastapi", reason="service tests need fastapi + httpx")
pytest.importorskip("multipart", reason="multipart uploads need python-multipart")

from fastapi.testclient import TestClient  # noqa: E402

from app import api, extraction, ocr_client  # noqa: E402


def configure(monkeypatch, **overrides) -> None:
    """Swap app.api's Settings for a copy with these fields changed.

    Settings is a frozen dataclass, so a field cannot be assigned in place;
    and everything in api.py reads through the module-level `settings` name,
    so rebinding that one name is enough to reconfigure the service.
    """
    monkeypatch.setattr(api, "settings", replace(api.settings, **overrides))

# Two photos of one receipt, overlapping by two lines -- the shape the whole
# feature exists for.
PAGE_1 = """MINI MART
water            10.00
bread            25.00
milk             18.00"""

PAGE_2 = """bread            25.OO
milk             18.00
eggs             40.00
Total            93.00"""


@pytest.fixture
def engine(monkeypatch):
    """A running service with the GPU and the OCR sidecar stubbed out.

    `seen` records what each stub was handed, which is how the tests check
    page order and the one-call-per-receipt rule.
    """
    seen: dict = {"ocr": [], "prompts": []}

    async def fake_ocr(image: bytes, content_type: str = "image/jpeg") -> dict:
        seen["ocr"].append(image.decode())
        # The image bytes ARE the page text in these tests, so a stub can map
        # one to the other without an OCR engine.
        return {"text": image.decode(), "ocr_seconds": 0.5}

    async def fake_extract(text: str):
        seen["prompts"].append(text)
        return (
            {"shop_name": "Mini Mart",
             "items": [{"name": "water", "price": "10.00", "c": "Groceries", "s": None},
                       {"name": "bread", "price": "25.00",
                        "c": "Food & Dining", "s": "Bakery & desserts"},
                       {"name": "milk", "price": "18.00",
                        "c": "Groceries", "s": "Drinks & beverages"},   # wrong category's sub
                       {"name": "eggs", "price": "40.00",
                        "c": "Coffee & cafe", "s": None}],             # not in the taxonomy
             "total_price": "93.00"},
            "raw",
            0.1,
        )

    monkeypatch.setattr(ocr_client, "run", fake_ocr)
    monkeypatch.setattr(ocr_client, "healthy", lambda: _true())
    monkeypatch.setattr(extraction, "extract", fake_extract)
    monkeypatch.setattr(extraction, "is_loaded", lambda: True)
    monkeypatch.setattr(extraction, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(extraction, "count_prompt_tokens", lambda text: len(text) // 4)
    monkeypatch.setattr(extraction, "token_budget", lambda: 3328)
    configure(monkeypatch, warmup=False, api_key="")

    with TestClient(api.app) as client:
        yield client, seen


async def _true() -> bool:
    return True


def _files(*pages: str):
    return [("files", (f"page{i}.jpg", page.encode(), "image/jpeg"))
            for i, page in enumerate(pages, 1)]


# --------------------------------------------------------------------------
# Several photos of one receipt
# --------------------------------------------------------------------------

def test_two_photos_are_ocrd_in_order_and_prompted_once(engine):
    client, seen = engine
    response = client.post("/v1/extract", files=_files(PAGE_1, PAGE_2))
    assert response.status_code == 200

    assert seen["ocr"] == [PAGE_1, PAGE_2]      # capture order preserved
    assert len(seen["prompts"]) == 1            # ONE model call, not one per page

    prompt = seen["prompts"][0]
    assert prompt.count("bread") == 1           # the overlap was removed
    assert prompt.count("eggs") == 1
    assert "water" in prompt and "Total" in prompt


def test_the_response_reports_the_seam(engine):
    client, _ = engine
    body = client.post("/v1/extract", files=_files(PAGE_1, PAGE_2)).json()

    assert body["ok"] is True
    assert body["ocrTexts"] == [PAGE_1, PAGE_2]     # raw pages, untouched
    assert body["stitchedText"].count("bread") == 1
    engine_block = body["engine"]
    assert engine_block["pages"] == 2
    assert engine_block["stitch"]["duplicate_lines_removed"] == 2
    seams = engine_block["stitch"]["seams"]
    assert len(seams) == 1
    assert seams[0]["page"] == 2 and seams[0]["overlap_lines"] == 2
    # Not 1.0: page 2 read "25.OO" where page 1 read "25.00", which is the
    # whole reason the match is fuzzy.
    assert 0.85 <= seams[0]["score"] < 1.0


def test_one_photo_keeps_the_single_page_shape(engine):
    client, _ = engine
    body = client.post("/v1/extract", files=_files(PAGE_1)).json()
    assert body["ocrTexts"] == [PAGE_1]
    # No seam to describe, so neither key appears -- a client reading
    # ocrTexts[0] behaves exactly as it did before multi-page existed.
    assert "stitchedText" not in body
    assert "stitch" not in body["engine"]


def test_raw_body_upload_still_works(engine):
    """The contract's option B: one image as the request body, no multipart."""
    client, seen = engine
    response = client.post("/v1/extract", content=PAGE_1.encode(),
                           headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 200
    assert seen["ocr"] == [PAGE_1]


def test_any_field_name_is_accepted(engine):
    client, seen = engine
    response = client.post("/v1/extract", files=[
        ("image", ("a.jpg", PAGE_1.encode(), "image/jpeg")),
        ("photo", ("b.jpg", PAGE_2.encode(), "image/jpeg")),
    ])
    assert response.status_code == 200
    assert seen["ocr"] == [PAGE_1, PAGE_2]


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------

def test_more_photos_than_max_pages_is_rejected(engine, monkeypatch):
    client, seen = engine
    configure(monkeypatch, warmup=False, api_key="", max_pages=2)
    response = client.post("/v1/extract", files=_files(PAGE_1, PAGE_2, PAGE_1))
    assert response.status_code == 413
    assert "limit is 2" in response.json()["detail"]
    # Rejected before anything was OCR'd, not after paying for three pages.
    assert seen["ocr"] == []


def test_no_image_is_a_400(engine):
    client, _ = engine
    assert client.post("/v1/extract", content=b"").status_code == 400


def test_an_empty_page_is_named_in_the_error(engine):
    client, _ = engine
    response = client.post("/v1/extract", files=[
        ("files", ("a.jpg", PAGE_1.encode(), "image/jpeg")),
        ("files", ("b.jpg", b"", "image/jpeg")),
    ])
    assert response.status_code == 400
    assert "Image 2" in response.json()["detail"]


@pytest.mark.parametrize("send", ["multipart", "raw"])
def test_the_size_limit_applies_to_both_upload_routes(engine, monkeypatch, send):
    """A limit only one of two documented routes enforces is not a limit."""
    client, seen = engine
    configure(monkeypatch, warmup=False, api_key="", max_upload_bytes=64)
    oversized = b"x" * 65
    if send == "multipart":
        response = client.post("/v1/extract",
                               files=[("files", ("a.jpg", oversized, "image/jpeg"))])
    else:
        response = client.post("/v1/extract", content=oversized,
                               headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 413
    assert "limit is 64" in response.json()["detail"]
    assert seen["ocr"] == []


def test_over_budget_text_is_rejected_rather_than_truncated(engine, monkeypatch):
    client, seen = engine
    monkeypatch.setattr(extraction, "token_budget", lambda: 10)
    response = client.post("/v1/extract", files=_files(PAGE_1, PAGE_2))
    assert response.status_code == 413
    assert "2 photo(s)" in response.json()["detail"]
    assert seen["prompts"] == []           # the GPU was never touched


# --------------------------------------------------------------------------
# /v1/extract-text -- the same path without a camera
# --------------------------------------------------------------------------

def test_extract_text_accepts_pages(engine):
    client, seen = engine
    body = client.post("/v1/extract-text",
                       json={"pages": [PAGE_1, PAGE_2]}).json()
    assert body["engine"]["pages"] == 2
    assert seen["prompts"][0].count("bread") == 1


def test_extract_text_still_accepts_a_single_string(engine):
    client, seen = engine
    body = client.post("/v1/extract-text", json={"text": PAGE_1}).json()
    assert body["engine"]["pages"] == 1
    assert seen["prompts"] == [PAGE_1]


def test_extract_text_and_extract_agree(engine):
    """The published promise: replaying the text gives the photo path's answer."""
    client, _ = engine
    photos = client.post("/v1/extract", files=_files(PAGE_1, PAGE_2)).json()
    text = client.post("/v1/extract-text", json={"pages": [PAGE_1, PAGE_2]}).json()
    for key in ("shopName", "items", "totalAmount", "stitchedText", "reconciles"):
        assert photos[key] == text[key]


def test_extract_text_rejects_a_bad_body(engine):
    client, _ = engine
    assert client.post("/v1/extract-text", json={}).status_code == 400
    assert client.post("/v1/extract-text",
                       json={"pages": [1, 2]}).status_code == 400


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------

def test_ready_advertises_the_page_limit(engine):
    client, _ = engine
    body = client.get("/ready").json()
    assert body["max_pages"] == api.settings.max_pages


def test_names_are_never_translated_but_the_slots_exist(engine):
    """Translation is a display layer owned by the app and the backend
    (MyCents server migration 0012). The engine emits the printed text and
    an explicit null in every English slot, so the draft is the categorize
    body key for key and a client never has to add fields before forwarding."""
    client, _ = engine
    body = client.post("/v1/extract", files=_files(PAGE_1)).json()
    assert body["shopNameEn"] is None
    assert body["items"], "fixture has items"
    for item in body["items"]:
        assert set(item) == {"name", "nameEn", "price", "category", "subcategory"}
        assert item["nameEn"] is None


# --------------------------------------------------------------------------
# Categories: a fallback for the backend's own pipeline
# --------------------------------------------------------------------------

def test_each_item_carries_the_models_category(engine):
    client, _ = engine
    items = client.post("/v1/extract", files=_files(PAGE_1)).json()["items"]
    assert [i["category"] for i in items[:3]] == ["Groceries", "Food & Dining", "Groceries"]


def test_a_category_outside_the_taxonomy_is_nulled_not_repaired(engine):
    """'Coffee & cafe' is not promoted to 'Coffee & café'. The null routes the
    item to the backend's classifier and user review -- and the item's name and
    price still arrive, which is the point of validating per item."""
    client, _ = engine
    eggs = client.post("/v1/extract", files=_files(PAGE_1)).json()["items"][3]
    assert eggs["category"] is None and eggs["subcategory"] is None
    assert (eggs["name"], eggs["price"]) == ("eggs", "40.00")


def test_subcategories_are_withheld_by_default(engine):
    client, _ = engine
    items = client.post("/v1/extract", files=_files(PAGE_1)).json()["items"]
    assert all(i["subcategory"] is None for i in items)


def test_subcategories_when_enabled_must_belong_to_their_category(engine, monkeypatch):
    configure(monkeypatch, emit_subcategory=True, warmup=False, api_key="")
    client, _ = engine
    items = client.post("/v1/extract", files=_files(PAGE_1)).json()["items"]
    assert items[1]["subcategory"] == "Bakery & desserts"
    assert items[2]["category"] == "Groceries"          # the category survives...
    assert items[2]["subcategory"] is None              # ...its misplaced sub does not
