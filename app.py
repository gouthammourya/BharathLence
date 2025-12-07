#!/usr/bin/env python3
"""
Faster app.py for BharatLens (fast-patch applied + robust TTS)

- quick digits OCR shortcut
- lightweight OpenCV preprocessing
- cached translations
- optional TTS (controlled by form param tts=1/true/yes)
- improved TTS fallback logic so audio is generated reliably
"""

import os
import re
import csv
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from io import BytesIO
from functools import lru_cache

from flask import Flask, request, render_template, url_for, jsonify, send_from_directory

# optional libs
try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None

try:
    from googletrans import Translator as GoogleTranslator
except Exception:
    GoogleTranslator = None

# gTTS (online)
try:
    from gtts import gTTS
    try:
        # new gTTS exposes a languages mapping callable or dict
        from gtts.lang import tts_langs as _gtts_langs
    except Exception:
        _gtts_langs = {}
except Exception:
    gTTS = None
    _gtts_langs = {}

# pyttsx3 offline fallback
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# OpenCV / numpy
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

# allow setting tesseract path via env
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if TESSERACT_CMD and pytesseract is not None:
    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    except Exception as e:
        print("Could not set pytesseract.tesseract_cmd:", e)

# Toggle debug images (False for speed)
DEBUG_OCR = False

# ---------------- OCR utilities ----------------
RE_KANNADA = re.compile(r"[\u0C80-\u0CFF]+")
RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]+")


def count_script_chars(s: str, script_re: re.Pattern) -> int:
    if not s:
        return 0
    return len(script_re.findall(s))


# ---------------- routes CSV loader ----------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
ROUTES_CSV = DATA_DIR / "routes.csv"


def load_routes_from_csv(path: Path):
    routes = {}
    if not path.exists():
        print("Routes CSV not found at", path)
        try:
            if DATA_DIR.exists():
                csvs = list(DATA_DIR.glob("*.csv"))
                if csvs:
                    path = csvs[0]
                    print("Auto-selecting CSV:", path)
                else:
                    return routes
        except Exception as e:
            print("CSV auto-select error:", e)
            return routes
    try:
        with path.open("r", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                route_no = (row.get("Bus Route") or row.get("route") or row.get("route_no") or "").strip()
                if not route_no:
                    route_no = (row.get("route_no") or "").strip()
                if not route_no:
                    continue
                routes[route_no] = {
                    "route_no": route_no,
                    "starting_from": row.get("Starting From") or row.get("from") or "",
                    "destination": row.get("Destination") or row.get("to") or "",
                    "via": row.get("VIA") or row.get("via") or "",
                    "stops": [s.strip() for s in (row.get("VIA") or "").split(",") if s.strip()],
                }
    except Exception as e:
        print("Error loading routes CSV:", e)
    return routes


ROUTES = load_routes_from_csv(ROUTES_CSV)


def lookup_bus_route(route_no: str) -> Optional[Dict[str, Any]]:
    if not route_no:
        return None
    info = ROUTES.get(route_no)
    if info:
        return info
    norm = re.sub(r"\D", "", route_no)
    for k, v in ROUTES.items():
        if re.sub(r"\D", "", k) == norm and norm != "":
            return v
    return None


# ---------------- OCR heuristics ----------------
def extract_route_from_text(raw_text: str) -> Dict[str, str]:
    result = {"route_no": "", "city": "", "raw_text": raw_text or ""}
    if not raw_text:
        return result
    tokens = re.split(r"[\s,;|/]+", raw_text)
    candidate = ""
    for t in tokens:
        if re.search(r"\d", t) and len(re.sub(r"\W", "", t)) <= 6:
            candidate = t.strip()
            break
    result["route_no"] = candidate
    lt = raw_text.lower()
    if any(x in lt for x in ["bengaluru", "bangalore", "bengalooru", "kempegowda"]):
        result["city"] = "Bengaluru"
    elif any(x in lt for x in ["ksrtc", "kerala"]):
        result["city"] = "Kerala"
    else:
        result["city"] = ""
    return result


# ---------------- DB logging ----------------
DB_PATH = PROJECT_ROOT / "scan_logs.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            route_no TEXT,
            city TEXT,
            raw_text TEXT,
            final_text TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_scan(route_no: str, city: str, raw_text: str, final_text: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO scan_logs (route_no, city, raw_text, final_text, timestamp) VALUES (?, ?, ?, ?, ?)",
        (route_no, city, raw_text, final_text, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


# ---------------- translation & tts helpers ----------------
def translate_text_local(text: str, target_lang: str) -> str:
    if not text:
        return text
    try:
        if GoogleTranslator is None:
            return text
        tr = GoogleTranslator()
        out = tr.translate(text, dest=target_lang)
        return getattr(out, "text", str(out))
    except Exception as e:
        print("translate_text_local error:", e)
        return text


@lru_cache(maxsize=2048)
def _cached_translate_internal(text: str, lang: str) -> str:
    return translate_text_local(text, lang)


def cached_translate(text: str, lang: str) -> str:
    if not text:
        return text
    key = " ".join(text.split())[:400]
    try:
        return _cached_translate_internal(key, lang)
    except Exception:
        return text


def _gtts_try_save(text: str, out_path: str, lang_code: str):
    """
    Internal helper: try saving with gTTS for lang_code, return (ok:bool, err_msg:str)
    """
    if gTTS is None:
        return False, "gTTS not installed"
    try:
        tts = gTTS(text=text, lang=lang_code)
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        tts.save(out_path)
        return True, ""
    except Exception as e:
        return False, str(e)


def _pyttsx3_save(text: str, out_path: str):
    """
    Save audio using pyttsx3 (offline). Returns (ok:bool, err_msg:str).
    Note: pyttsx3 doesn't support language selection via code; it uses system voices.
    """
    if pyttsx3 is None:
        return False, "pyttsx3 not installed"
    try:
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        engine = pyttsx3.init()
        # Optionally, choose a voice by examining engine.getProperty('voices')
        # voices = engine.getProperty('voices')
        # for v in voices:
        #     if 'hindi' in v.name.lower() or 'kannada' in v.name.lower():
        #         engine.setProperty('voice', v.id); break
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass
        return True, ""
    except Exception as e:
        return False, str(e)


def text_to_speech_local(text: str, out_path: str, lang: str = "en") -> bool:
    """
    Robust TTS wrapper:
    - tries gTTS first (online, better voices)
    - falls back to pyttsx3 (offline) when gTTS fails
    - returns True if audio file created, False otherwise
    """
    if not text:
        return False

    req = (lang or "en").lower().strip()
    mapping = {
        "hi": "hi", "hindi": "hi",
        "kn": "kn", "kannada": "kn",
        "en": "en", "english": "en",
    }

    candidates = []
    if req in mapping:
        candidates.append(mapping[req])
    else:
        candidates.append(req)
        if len(req) > 2:
            candidates.append(req[:2])
    if "en" not in candidates:
        candidates.append("en")

    # Determine supported languages from gTTS if available
    supported = set()
    try:
        if callable(_gtts_langs):
            supported = set(_gtts_langs().keys())
        elif isinstance(_gtts_langs, dict):
            supported = set(_gtts_langs.keys())
    except Exception:
        supported = set()

    tried = set()
    # Try gTTS first for candidate languages (if gTTS available)
    for c in candidates:
        if not c or c in tried:
            continue
        tried.add(c)
        # optionally skip unsupported languages (but we'll still attempt)
        ok, err = _gtts_try_save(text, out_path, c)
        if ok:
            print(f"[TTS:gTTS] Saved audio using lang='{c}' -> {out_path}")
            return True
        else:
            print(f"[TTS:gTTS] Failed for lang='{c}' err={err}")

    # gTTS either not installed or failed for all candidates -> try offline pyttsx3
    ok, err = _pyttsx3_save(text, out_path)
    if ok:
        print(f"[TTS:pyttsx3] Saved audio (offline) -> {out_path}")
        return True
    else:
        print(f"[TTS:pyttsx3] Failed to save audio: {err}")

    print("[TTS] All attempts failed, no audio generated.")
    return False


# ---------------- OCR (fast tuned) ----------------
DEBUG_DIR = PROJECT_ROOT / "static" / "debug"
if DEBUG_OCR:
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def ocr_debug_save(img, name: str):
    if not DEBUG_OCR or cv2 is None or np is None:
        return
    try:
        p = DEBUG_DIR / f"{name}.png"
        im = img
        if isinstance(im, np.ndarray) and im.dtype != np.uint8:
            im = (np.clip(im, 0, 1) * 255).astype("uint8")
        if len(im.shape) == 2:
            im2 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        else:
            im2 = im
        cv2.imwrite(str(p), im2)
    except Exception as e:
        print("ocr_debug_save error:", e)


def try_pytess_configs(img_for_tess, langs="eng+hin+kan"):
    """Fast/limited tesseract config attempts."""
    best = {"text": "", "cfg": None}
    configs = [
        "--psm 7",
        "--psm 6",
        "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-",
    ]
    pil_img = None
    try:
        if img_for_tess is None:
            return best
        if cv2 is not None:
            pil_img = Image.fromarray(cv2.cvtColor(img_for_tess, cv2.COLOR_BGR2RGB))
        else:
            return best
    except Exception:
        try:
            pil_img = Image.fromarray(img_for_tess)
        except Exception:
            pil_img = None

    for cfg in configs:
        try:
            txt = pytesseract.image_to_string(pil_img, lang=langs, config=cfg) if pil_img is not None else ""
            clean = (txt or "").strip()
            if len(clean) > len(best["text"]):
                best = {"text": clean, "cfg": cfg}
        except Exception:
            continue
    return best


def local_ocr_image(image_bytes: bytes) -> str:
    """
    Fast-preference OCR:
    - quick digits-only attempt first (very fast)
    - lightweight OpenCV preprocessing and limited language/config attempts
    - fallback to simple PIL multi-lang if cv2 missing
    """
    # 1) quick digits-only (very fast)
    try:
        if Image is not None and pytesseract is not None:
            pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
            quick_cfg = "--psm 7 -c tessedit_char_whitelist=0123456789"
            quick_txt = pytesseract.image_to_string(pil_img, lang="eng", config=quick_cfg)
            quick_clean = (quick_txt or "").strip()
            if quick_clean and re.search(r"\d", quick_clean):
                if len(re.sub(r"\D", "", quick_clean)) <= 5:
                    return quick_clean
    except Exception:
        pass

    # If cv2 or pytesseract missing -> PIL fallback (fast)
    if (cv2 is None or np is None) or (Image is None or pytesseract is None):
        try:
            if Image is None or pytesseract is None:
                return ""
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            candidates = ["eng+kan", "eng+hin", "eng"]
            best = {"text": "", "score": -1}
            for langs in candidates:
                try:
                    txt = pytesseract.image_to_string(img, lang=langs)
                    if not txt:
                        continue
                    score = len(txt.strip())
                    if "kan" in langs:
                        score += count_script_chars(txt, RE_KANNADA) * 2
                    if "hin" in langs:
                        score += count_script_chars(txt, RE_DEVANAGARI) * 2
                    if score > best["score"]:
                        best = {"text": txt.strip(), "score": score}
                except Exception:
                    continue
            return best["text"] or ""
        except Exception as e:
            print("local_ocr_image (PIL fallback) error:", e)
            return ""

    # Enhanced OpenCV path (lightweight)
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("cv2 failed to decode image")
            return ""

        h, w = img.shape[:2]
        if max(h, w) < 800:
            scale = 1.8
        elif max(h, w) < 1200:
            scale = 1.3
        else:
            scale = 1.0

        if scale != 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        den = cv2.bilateralFilter(gray_blur, d=5, sigmaColor=50, sigmaSpace=50)
        _, thresh = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        proc = thresh
        proc_bgr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)

        ocr_debug_save(img, "orig_resized")
        ocr_debug_save(gray, "gray")
        ocr_debug_save(den, "denoised")
        ocr_debug_save(thresh, "thresh")
        ocr_debug_save(proc_bgr, "proc_final")

        cand_langs = ["eng+kan", "eng+hin", "eng"]
        best_overall = {"text": "", "cfg": None, "langs": None}
        for langs in cand_langs:
            try:
                res = try_pytess_configs(proc_bgr, langs=langs)
                text = (res.get("text") or "").strip()
                if not text:
                    continue
                score = len(text)
                if "kan" in langs:
                    score += len(RE_KANNADA.findall(text)) * 2
                if "hin" in langs:
                    score += len(RE_DEVANAGARI.findall(text)) * 2
                if score > len(best_overall["text"]):
                    best_overall = {"text": text, "cfg": res.get("cfg"), "langs": langs}
            except Exception:
                continue

        result_text = best_overall["text"] or ""
        result_text = re.sub(r"\s{2,}", " ", result_text).strip()
        print("[DEBUG] OCR result length:", len(result_text), "langs_used:", best_overall.get("langs"), "cfg:", best_overall.get("cfg"))
        return result_text or ""
    except Exception as e:
        print("enhanced local_ocr_image error:", e)
        return ""


# ---------------- route translation cache ----------------
ROUTE_TRANSLATION_CACHE: Dict[tuple, dict] = {}


# ---------------- build route response ----------------
def build_route_response(route_no: str, lang: str = "en", raw_text: str = "", city_hint: str = "", generate_tts: bool = False):
    route_info = None
    final_text = ""
    audio_url = None
    detected_city = city_hint or ""

    print(f"[DEBUG] build_route_response called: route_no={route_no} lang={lang} raw_text_len={len(raw_text or '')}")

    if route_no:
        route_info = lookup_bus_route(route_no)
        print(f"[DEBUG] lookup_bus_route returned: {bool(route_info)}")

    if not route_info and raw_text:
        parsed = extract_route_from_text(raw_text)
        if parsed.get("route_no") and not route_no:
            route_no = parsed["route_no"]
            route_info = lookup_bus_route(route_no)
        if parsed.get("city"):
            detected_city = parsed.get("city")

    if route_info:
        ri = dict(route_info)
        final_text = (
            f"Route {ri.get('route_no')} from "
            f"{ri.get('starting_from') or 'Unknown'} to "
            f"{ri.get('destination') or 'Unknown'}"
        )
    else:
        ri = None
        final_text = f"Could not find route details for '{route_no}'."

    # translate if requested (use cached_translate)
    if lang and lang != "en":
        try:
            cache_key = (route_no or "", lang)
            if route_no and cache_key in ROUTE_TRANSLATION_CACHE:
                ri = dict(ROUTE_TRANSLATION_CACHE[cache_key]) if ROUTE_TRANSLATION_CACHE.get(cache_key) else ri
            else:
                try:
                    text_for_translation = raw_text if raw_text else final_text
                    translated_raw = cached_translate(text_for_translation, lang)
                    if translated_raw and translated_raw.strip():
                        raw_text = translated_raw
                    print("[DEBUG] translated_raw:", (raw_text or "")[:120])
                except Exception as e:
                    print("translate raw_text failed:", e)
                try:
                    translated_final = cached_translate(final_text, lang)
                    if translated_final and translated_final.strip():
                        final_text = translated_final
                    print("[DEBUG] translated_final:", (final_text or "")[:120])
                except Exception as e:
                    print("translate final_text failed:", e)

                if ri:
                    for fld in ("starting_from", "destination", "via"):
                        if ri.get(fld):
                            try:
                                t = cached_translate(ri[fld], lang)
                                if t and t.strip():
                                    ri[fld] = t
                            except Exception as e:
                                print("translate route_info field error:", fld, e)
                    if isinstance(ri.get("stops"), list) and ri.get("stops"):
                        new_stops = []
                        for s in ri.get("stops", []):
                            try:
                                t = cached_translate(s, lang)
                                new_stops.append(t if (t and t.strip()) else s)
                            except Exception as e:
                                print("translate stop error:", e)
                                new_stops.append(s)
                        ri["stops"] = new_stops

                if route_no and ri:
                    try:
                        ROUTE_TRANSLATION_CACHE[cache_key] = dict(ri)
                    except Exception:
                        pass
        except Exception as e:
            print("build_route_response translation error:", e)

    # TTS (only when requested)
    if generate_tts:
        try:
            audio_dir = PROJECT_ROOT / "static" / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            audio_filename = f"tts_{timestamp}.mp3"
            audio_path = audio_dir / audio_filename
            tts_lang = lang or "en"
            ok = text_to_speech_local(final_text, str(audio_path), lang=tts_lang)
            if ok:
                audio_url = url_for("static", filename=f"audio/{audio_filename}")
            else:
                print("[TTS] text_to_speech_local returned False; no audio URL.")
        except Exception as e:
            print("TTS generation error:", e)

    return {
        "raw_text": raw_text,
        "route_info": ri,
        "final_text": final_text,
        "audio_url": audio_url,
        "city": detected_city,
    }


# ---------------- Flask app ----------------
app = Flask(__name__, static_folder=str(PROJECT_ROOT / "static"), template_folder=str(PROJECT_ROOT / "templates"))
init_db()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    lang = (request.form.get("target_lang") or "en") or "en"
    generate_tts = (request.form.get("tts") or request.form.get("generate_tts") or "").lower() in ("1", "true", "yes")
    if not file:
        route_no = request.form.get("route_no", "").strip()
        raw_text = ""
        resp = build_route_response(route_no, lang=lang, raw_text=raw_text, generate_tts=generate_tts)
        log_scan(route_no, resp.get("city", ""), raw_text, resp.get("final_text", ""))
        return render_template(
            "result.html",
            raw_text=resp["raw_text"],
            route_no=route_no,
            city=resp["city"],
            route_info=resp["route_info"],
            final_text=resp["final_text"],
            audio_file=resp["audio_url"],
            target_lang=lang,
        )
    image_bytes = file.read()
    raw_text = local_ocr_image(image_bytes)
    parsed = extract_route_from_text(raw_text)
    route_no = parsed.get("route_no", "")
    city_hint = parsed.get("city", "")
    resp = build_route_response(route_no, lang=lang, raw_text=raw_text, city_hint=city_hint, generate_tts=generate_tts)
    try:
        log_scan(route_no, resp.get("city", ""), raw_text, resp.get("final_text", ""))
    except Exception as e:
        print("Failed to log scan:", e)
    return render_template(
        "result.html",
        raw_text=resp["raw_text"],
        route_no=route_no,
        city=resp["city"],
        route_info=resp["route_info"],
        final_text=resp["final_text"],
        audio_file=resp["audio_url"],
        target_lang=lang,
    )


@app.route("/route-search", methods=["POST"])
def route_search():
    try:
        route_no = request.form.get("route_no", "").strip()
        lang = request.form.get("target_lang", None) or request.form.get("lang", None) or "en"
        lang = lang or "en"
        generate_tts = (request.form.get("tts") or request.form.get("generate_tts") or "").lower() in ("1", "true", "yes")
        resp = build_route_response(route_no, lang=lang, raw_text="", city_hint="", generate_tts=generate_tts)
        try:
            log_scan(route_no, resp.get("city", ""), "", resp.get("final_text", ""))
        except Exception as e:
            print("Failed to log route_search:", e)
        return render_template(
            "result.html",
            raw_text=resp["raw_text"],
            route_no=route_no,
            city=resp["city"],
            route_info=resp["route_info"],
            final_text=resp["final_text"],
            audio_file=resp["audio_url"],
            target_lang=lang,
        )
    except Exception as e:
        print("route_search error:", e)
        traceback.print_exc()
        return render_template("index.html", error="Route search failed on server."), 500


@app.route("/history", methods=["GET"])
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, route_no, city, raw_text, final_text, timestamp FROM scan_logs ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    formatted = [
        {"id": r[0], "route_no": r[1], "city": r[2], "raw_text": r[3], "final_text": r[4], "timestamp": r[5]}
        for r in rows
    ]
    return render_template("history.html", rows=formatted)


@app.route("/api/bus/<route_no>", methods=["GET"])
def api_bus(route_no):
    lang = request.args.get("lang", "en") or "en"
    print(f"[DEBUG] api_bus called: route_no={route_no} lang={lang}")
    info = lookup_bus_route(route_no)
    if not info:
        return jsonify({"error": "Route not found"}), 404
    info_out = dict(info)
    if lang and lang != "en":
        try:
            for fld in ("starting_from", "destination", "via"):
                if info_out.get(fld):
                    try:
                        t = cached_translate(info_out[fld], lang)
                        if t and t.strip():
                            info_out[fld] = t
                    except Exception as e:
                        print("api_bus translate field error:", fld, e)
            if isinstance(info_out.get("stops"), list) and info_out.get("stops"):
                new_stops = []
                for s in info_out.get("stops", []):
                    try:
                        t = cached_translate(s, lang)
                        new_stops.append(t if (t and t.strip()) else s)
                    except Exception as e:
                        print("api_bus translate stop error:", e)
                        new_stops.append(s)
                info_out["stops"] = new_stops
        except Exception as e:
            print("api_bus translation overall error:", e)
    return jsonify(info_out)


@app.route("/api/route-search", methods=["POST"])
def api_route_search():
    try:
        route_no = ""
        if request.is_json:
            route_no = (request.json.get("route_no") or "").strip()
        else:
            route_no = (request.form.get("route_no") or "").strip()
        lang = None
        if request.is_json:
            lang = request.json.get("target_lang") or request.json.get("lang")
        if not lang:
            lang = request.form.get("target_lang") or request.form.get("lang")
        lang = (lang or "en")
        resp = build_route_response(route_no, lang=lang, raw_text="", city_hint="")
        try:
            log_scan(route_no, resp.get("city", ""), "", resp.get("final_text", ""))
        except Exception:
            pass
        return jsonify(resp)
    except Exception as e:
        print("api_route_search error:", e)
        traceback.print_exc()
        return jsonify({"error": "server error"}), 500


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(str(PROJECT_ROOT / "static"), filename)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_flag = os.getenv("FLASK_DEBUG", "1") in ("1", "true", "True")
    app.run(host="0.0.0.0", port=port, debug=debug_flag)
