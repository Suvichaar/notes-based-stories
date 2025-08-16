import os
import io
import re
import json
import time
import base64
import zipfile
import random
import string
from io import BytesIO
from datetime import datetime
from pathlib import Path

import requests
import boto3
from PIL import Image
import streamlit as st
from streamlit.components.v1 import html as st_html  # <-- for live HTML preview

# --- Azure Doc Intelligence SDK ---
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except Exception:
    DocumentIntelligenceClient = None  # we'll error nicely later
    AzureKeyCredential = None

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Suvichaar Builder",
    page_icon="🧠",
    layout="centered"
)
st.title("🧠 Suvichaar Builder")
st.caption("OCR (Doc Intelligence) → GPT JSON → DALL·E → S3/CDN → (optional) SEO/TTS/SSML → Fill HTML templates → Upload & Verify")

# ---------------------------
# Secrets / Config
# ---------------------------
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

# Azure OpenAI (GPT)
AZURE_API_KEY     = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT    = get_secret("AZURE_ENDPOINT")  # https://<resource>.openai.azure.com
AZURE_DEPLOYMENT  = get_secret("AZURE_DEPLOYMENT", "gpt-4o")  # your *deployment* name
AZURE_API_VERSION = get_secret("AZURE_API_VERSION", "2024-08-01-preview")

# Azure Document Intelligence (OCR)
AZURE_DI_ENDPOINT = get_secret("AZURE_DI_ENDPOINT")  # https://<your-di>.cognitiveservices.azure.com/
AZURE_DI_KEY      = get_secret("AZURE_DI_KEY")

# Azure DALL·E (Images)
DALE_ENDPOINT     = get_secret("DALE_ENDPOINT")  # e.g. https://.../openai/deployments/dall-e-3/images/generations?api-version=2024-02-01
DAALE_KEY         = get_secret("DAALE_KEY")

# AWS S3
AWS_ACCESS_KEY        = get_secret("AWS_ACCESS_KEY")
AWS_SECRET_KEY        = get_secret("AWS_SECRET_KEY")
AWS_SESSION_TOKEN     = get_secret("AWS_SESSION_TOKEN")  # optional (for temporary creds)
AWS_REGION            = get_secret("AWS_REGION", "ap-south-1")
AWS_BUCKET            = get_secret("AWS_BUCKET", "suvichaarapp")  # default to suvichaarapp
S3_PREFIX             = get_secret("S3_PREFIX", "media")          # used for images/audio

# ---- Hard-lock HTML/JSON at bucket ROOT + root CDN base (no /webstory-html) ----
HTML_S3_PREFIX = ""  # bucket root
CDN_HTML_BASE  = get_secret("CDN_HTML_BASE", "https://cdn.suvichaar.org/")

# CDN image handler prefix (base64-encoded template)
CDN_PREFIX_MEDIA  = get_secret("CDN_PREFIX_MEDIA", "https://media.suvichaar.org/")

# Fallback image
DEFAULT_ERROR_IMAGE = get_secret("DEFAULT_ERROR_IMAGE", "https://media.suvichaar.org/default-error.jpg")

# Azure Speech (TTS)
AZURE_SPEECH_KEY     = get_secret("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION  = get_secret("AZURE_SPEECH_REGION", "eastus")
VOICE_NAME_DEFAULT   = get_secret("VOICE_NAME", "hi-IN-AaravNeural")

# CDN base for audio (CloudFront etc.)
CDN_BASE             = get_secret("CDN_BASE", "https://cdn.suvichaar.org/")

# Sanity checks (warn if missing)
missing_core = []
for k in [
    "AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_DEPLOYMENT",
    "AZURE_DI_ENDPOINT", "AZURE_DI_KEY",
    "DALE_ENDPOINT", "DAALE_KEY",
    "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET"
]:
    if not get_secret(k, None):
        missing_core.append(k)
if missing_core:
    st.warning("Add these secrets in `.streamlit/secrets.toml`: " + ", ".join(missing_core))

# ---------------------------
# AWS helpers (robust client + verified uploads)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_s3_client():
    kwargs = dict(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    if AWS_SESSION_TOKEN:
        kwargs["aws_session_token"] = AWS_SESSION_TOKEN
    return boto3.client("s3", **kwargs)


def s3_put_text_file(bucket: str, key: str, body: bytes, content_type: str, cache_control: str = "public, max-age=31536000, immutable"):
    """Upload a small text file and verify it exists by HEADing it back.
    Returns a dict: {"ok": bool, "etag": str|None, "key": str, "len": int, "error": str|None}
    """
    s3 = get_s3_client()
    put_args = {
        "Bucket": bucket,
        "Key": key,
        "Body": body,
        "ContentType": content_type,
        "CacheControl": cache_control,
    }

    try:
        s3.put_object(**put_args)
    except Exception as e:
        return {"ok": False, "etag": None, "key": key, "len": len(body), "error": f"put_object failed: {e}"}

    # Verify via HEAD
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        etag = head.get("ETag", "").strip('"')
        cl = int(head.get("ContentLength", 0))
        ok = cl == len(body)
        return {"ok": ok, "etag": etag, "key": key, "len": cl, "error": None if ok else f"size mismatch {cl}!={len(body)}"}
    except Exception as e:
        return {"ok": False, "etag": None, "key": key, "len": 0, "error": f"head_object failed: {e}"}

# ---------------------------
# Other helpers
# ---------------------------
def build_resized_cdn_url(bucket: str, key_path: str, width: int, height: int) -> str:
    """Return base64-encoded template URL for your Serverless Image Handler."""
    template = {
        "bucket": bucket,
        "key": key_path,
        "edits": {"resize": {"width": width, "height": height, "fit": "cover"}}
    }
    encoded = base64.urlsafe_b64encode(json.dumps(template).encode()).decode()
    return f"{CDN_PREFIX_MEDIA}{encoded}"

SAFE_FALLBACK = (
    "A joyful, abstract geometric illustration symbolizing unity and learning — "
    "soft shapes, harmonious gradients, friendly silhouettes; "
    "flat vector style, bright colors; family-friendly; "
    "no text, no logos, no watermarks, no real-person likeness."
)

def robust_parse_model_json(raw_reply: str):
    """Parse model reply into a dict or return None."""
    parsed = None
    try:
        parsed = json.loads(raw_reply)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw_reply)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None
    return parsed if isinstance(parsed, dict) else None


def repair_json_with_model(raw_reply: str, chat_url: str, headers: dict):
    """Ask the model to fix its own output into valid JSON per schema; returns dict or None."""
    schema_hint = """
Keys (English), values in detected language. If any field is missing, use an empty string:
{
  "language": "hi|en|bn|ta|te|mr|gu|kn|pa|en-IN|...",
  "storytitle": "...",
  "s1paragraph1": "...",
  "s2paragraph1": "...",
  "s3paragraph1": "...",
  "s4paragraph1": "...",
  "s5paragraph1": "...",
  "s6paragraph1": "...",
  "s1alt1": "...",
  "s2alt1": "...",
  "s3alt1": "...",
  "s4alt1": "...",
  "s5alt1": "...",
  "s6alt1": "..."
}
Return ONLY valid JSON, no code fences, no commentary.
"""
    payload_fix = {
        "messages": [
            {"role": "system",
             "content": "You are a strict JSON formatter. You output ONLY valid minified JSON. No prose."},
            {"role": "user",
             "content": f"This text was intended to be JSON but is invalid/truncated. "
                        f"Repair it into valid JSON that matches the schema.\n\nSchema:\n{schema_hint}\n\nText:\n{raw_reply}"}
        ],
        "temperature": 0.0,
        "max_tokens": 1600,
        "response_format": {"type": "json_object"}
    }
    try:
        res = requests.post(chat_url, headers=headers, json=payload_fix, timeout=150)
        if res.status_code != 200:
            return None
        fixed = res.json()["choices"][0]["message"]["content"]
        return robust_parse_model_json(fixed)
    except Exception:
        return None


def call_azure_chat(messages, *, temperature=0.2, max_tokens=2000, force_json=True):
    """Call Azure Chat (JSON mode default). Returns (ok, content_or_err)."""
    chat_headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    chat_url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}

    body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if force_json:
        body["response_format"] = {"type": "json_object"}

    try:
        res = requests.post(chat_url, headers=chat_headers, params=params, json=body, timeout=150)
    except Exception as e:
        return False, f"Azure request failed: {e}"

    if res.status_code == 200:
        return True, res.json()["choices"][0]["message"]["content"]

    # If JSON mode fails, retry without it
    if force_json:
        body.pop("response_format", None)
        try:
            res2 = requests.post(chat_url, headers=chat_headers, params=params, json=body, timeout=150)
            if res2.status_code == 200:
                return True, res2.json()["choices"][0]["message"]["content"]
            return False, f"Azure Chat error: {res2.status_code} — {res2.text[:300]}"
        except Exception as e:
            return False, f"Azure retry failed: {e}"

    return False, f"Azure Chat error: {res.status_code} — {res.text[:300]}"

# ---------- Language auto-detect (Hindi vs English) ----------
def detect_hi_or_en(text: str) -> str:
    """Return 'hi' if text is mostly Devanagari, else 'en'."""
    if not text:
        return "en"
    devanagari = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    latin = sum(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in text)
    total_letters = devanagari + latin
    if total_letters == 0:
        return "hi" if devanagari > 0 else "en"
    return "hi" if (devanagari / total_letters) >= 0.25 else "en"

# ---------- Azure Document Intelligence (OCR) ----------
def ocr_read_any(bytes_blob: bytes) -> str:
    """
    Uses Azure Document Intelligence 'prebuilt-read' to extract text for images or PDFs.
    Returns merged text with [[PAGE n]] markers.
    """
    if DocumentIntelligenceClient is None or AzureKeyCredential is None:
        return ""
    if not (AZURE_DI_ENDPOINT and AZURE_DI_KEY):
        return ""

    try:
        client = DocumentIntelligenceClient(
            endpoint=AZURE_DI_ENDPOINT.rstrip("/"),
            credential=AzureKeyCredential(AZURE_DI_KEY),
        )
        poller = client.begin_analyze_document("prebuilt-read", body=bytes_blob)
        doc = poller.result()
        parts = []
        if getattr(doc, "pages", None):
            for p in doc.pages:
                lines = [ln.content for ln in getattr(p, "lines", []) or [] if getattr(ln, "content", None)]
                page_txt = "\n".join(lines).strip()
                if page_txt:
                    parts.append(f"[[PAGE {getattr(p, 'page_number', len(parts)+1)}]]\n{page_txt}")
        elif getattr(doc, "paragraphs", None):
            parts.append("[[PAGE 1]]\n" + "\n".join(pp.content for pp in doc.paragraphs if getattr(pp,"content",None)))
        else:
            raw = (getattr(doc, "content", "") or "").strip()
            if raw:
                parts.append("[[PAGE 1]]\n" + raw)
        return "\n\n".join(parts).strip()
    except Exception:
        return ""

def ocr_many(files) -> str:
    """
    Accepts a mixed list of Streamlit UploadedFile (images and/or PDFs).
    Returns concatenated text with [[FILE i: name]] and [[PAGE n]] markers.
    """
    chunks = []
    for i, f in enumerate(files, start=1):
        try:
            text = ocr_read_any(f.getvalue())
            if text:
                chunks.append(f"[[FILE {i}: {f.name}]]\n{text}")
            else:
                st.warning(f"OCR returned empty text for {f.name}")
        except Exception as e:
            st.warning(f"OCR failed for {f.name}: {e}")
    return "\n\n".join(chunks).strip()

# -------- NEW: Enrich alt prompts for better image generation --------
def enrich_alt_prompts_with_model(result_json: dict, language: str) -> dict:
    """
    Turn s1..s6 'alt1' prompts into rich, safe, image-gen prompts using Azure Chat.
    Returns a *new* dict with the same keys but improved alt prompts.
    """
    # If Azure config is missing, just return input unchanged
    if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_VERSION):
        return dict(result_json)

    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    chat_url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"

    sys_msg = (
        "You are an art director who writes SINGLE, detailed prompts for image generation.\n"
        "Return ONLY valid JSON with one key 'alt'.\n"
        "The prompt must:\n"
        "- Be in the requested language when relevant.\n"
        "- Be a single paragraph (no lists), <= 1200 characters.\n"
        "- Specify: subject(s), setting, composition (foreground/mid/background), camera/perspective, "
        "lighting, color palette, mood/emotion, motion/action where relevant, style keywords.\n"
        "- Enforce style: flat vector illustration, clean shapes, smooth gradients, crisp edges, "
        "no text/captions/logos, no watermarks, no trademarks, no real-person likeness.\n"
        "- Keep it family-friendly, safe, inclusive; replace unsafe content with abstract, peaceful motifs.\n"
        "Output JSON ONLY: {\"alt\":\"...\"}"
    )

    improved = dict(result_json)
    for i in range(1, 7):
        base_alt = (result_json.get(f"s{i}alt1") or "").strip()
        slide_txt = (result_json.get(f"s{i}paragraph1") or "").strip()
        # If both are empty, skip
        if not (base_alt or slide_txt):
            continue

        user_msg = (
            f"Language: {language}\n"
            f"Slide text (context): {slide_txt}\n"
            f"Existing short prompt: {base_alt}\n\n"
            "Write a SINGLE improved prompt in the JSON format described."
        )
        payload = {
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.3,
            "max_tokens": 700,
            "response_format": {"type": "json_object"}
        }

        try:
            r = requests.post(chat_url, headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                data = robust_parse_model_json(content)
                if isinstance(data, dict) and data.get("alt"):
                    improved[f"s{i}alt1"] = data["alt"]
                    continue  # next slide
        except Exception:
            pass  # fall through to keep original

        # Fallback: if enrichment fails, ensure at least safe guardrails on original
        improved[f"s{i}alt1"] = (
            (base_alt or slide_txt) +
            " — flat vector illustration, clean geometric shapes, smooth gradients, harmonious palette, "
            "inclusive and family-friendly; no text, no logos, no watermarks, no real-person likeness."
        )

    return improved

# -------- Image generation + S3 upload --------
def _variation_token(k=8) -> str:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))

def _variation_style() -> str:
    return random.choice([
        "exam-focused neutral tone",
        "student-friendly playful tone",
        "teacher’s concise quiz tone",
        "competitive test-prep tone",
        "flashcard-style phrasing",
        "minimalist geometric look",
        "soft gradients and bokeh"
    ])

def sanitize_prompt(chat_url: str, headers: dict, original_prompt: str) -> str:
    """Rewrite any risky prompt into a safe, positive, family-friendly version using Azure Chat."""
    sanitize_payload = {
        "messages": [
            {"role": "system", "content": (
                "Rewrite image prompts to be safe, positive, inclusive, and family-friendly. "
                "Remove any hate/harassment/violence/adult/illegal/extremist content, slogans, logos, "
                "or real-person likenesses. Keep the core educational idea and flat vector art style. "
                "Return ONLY the rewritten prompt text."
            )},
            {"role": "user", "content": f"Original prompt:\n{original_prompt}\n\nRewritten safe prompt:"}
        ],
        "temperature": 0.2,
        "max_tokens": 300
    }
    try:
        sr = requests.post(chat_url, headers=headers, json=sanitize_payload, timeout=60)
        if sr.status_code == 200:
            return sr.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.info(f"Sanitizer call failed; using local guards: {e}")

    # Toughened local fallback
    return (
        original_prompt +
        " — flat vector illustration, clean geometric shapes, smooth gradients, harmonious palette, "
        "inclusive and family-friendly; no text, no logos, no watermarks, no real-person likeness."
    )


def generate_and_upload_images(result_json: dict, *, vary_images: bool = True) -> dict:
    """Generate DALL·E images, upload originals to S3, return CDN resized URLs in JSON."""
    if not all([DALE_ENDPOINT, DAALE_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_BUCKET]):
        st.error("Missing DALL·E and/or AWS S3 secrets.")
        return {**result_json}

    s3 = get_s3_client()

    slug = (
        (result_json.get("storytitle") or "story")
        .lower()
        .replace(" ", "-")
        .replace(":", "")
        .replace(".", "")
    )
    out = {k: result_json[k] for k in result_json}
    first_slide_key = None

    headers_dalle = {"Content-Type": "application/json", "api-key": DAALE_KEY}
    progress = st.progress(0, text="Generating images…")

    for i in range(1, 7):
        raw_prompt = result_json.get(f"s{i}alt1", "") or ""
        chat_headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
        chat_url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
        safe_prompt = sanitize_prompt(chat_url, chat_headers, raw_prompt)

        # 🔀 ensure different images across runs even for same inputs
        if vary_images:
            safe_prompt = (
                f"{safe_prompt}\n\n"
                f"Creative variation code: {_variation_token()}.\n"
                f"Substyle: {_variation_style()}."
            )

        payload = {"prompt": safe_prompt, "n": 1, "size": "1024x1024"}
        image_url = None

        for attempt in range(3):
            r = requests.post(DALE_ENDPOINT, headers=headers_dalle, json=payload, timeout=120)
            if r.status_code == 200:
                try:
                    image_url = r.json()["data"][0]["url"]
                    break
                except Exception as e:
                    st.info(f"Slide {i}: unexpected DALL·E response format: {e}")
                    break
            elif r.status_code in (400, 403):
                st.info(f"Slide {i}: DALL·E blocked, retrying with fallback.")
                payload = {"prompt": SAFE_FALLBACK, "n": 1, "size": "1024x1024"}
                continue
            elif r.status_code == 429:
                st.info(f"Slide {i}: rate-limited, waiting 10s…")
                time.sleep(10)
            else:
                st.info(f"Slide {i}: DALL·E error {r.status_code} — {r.text[:200]}")
                break

        if image_url:
            try:
                img_data = requests.get(image_url, timeout=120).content
                buffer = BytesIO(img_data)  # upload original; no local resize
                key = f"{S3_PREFIX.rstrip('/')}/{slug}/slide{i}.jpg"
                extra_args = {"ContentType": "image/jpeg"}
                # No ACL
                s3.upload_fileobj(buffer, AWS_BUCKET, key, ExtraArgs=extra_args)
                if i == 1:
                    first_slide_key = key

                # build CDN resized URL (720x1200)
                final_url = build_resized_cdn_url(AWS_BUCKET, key, 720, 1200)
                out[f"s{i}image1"] = final_url
            except Exception as e:
                st.info(f"Slide {i}: upload/CDN URL build failed → {e}")
                out[f"s{i}image1"] = DEFAULT_ERROR_IMAGE
        else:
            out[f"s{i}image1"] = DEFAULT_ERROR_IMAGE

        progress.progress(i/6.0, text=f"Generating images… ({i}/6)")

    progress.empty()

    # portrait cover from slide 1 via CDN (640x853) — keep both spellings for template compatibility
    try:
        if first_slide_key:
            cover = build_resized_cdn_url(AWS_BUCKET, first_slide_key, 640, 853)
            out["portraitcoverurl"] = cover
            out["potraitcoverurl"] = cover  # backward-compat
            out["potraightcoverurl"] = cover
        else:
            out["portraitcoverurl"] = DEFAULT_ERROR_IMAGE
            out["potraitcoverurl"] = DEFAULT_ERROR_IMAGE
            out["potraightcoverurl"] = DEFAULT_ERROR_IMAGE
    except Exception as e:
        st.info(f"Portrait cover URL build failed: {e}")
        out["portraitcoverurl"] = DEFAULT_ERROR_IMAGE
        out["potraitcoverurl"] = DEFAULT_ERROR_IMAGE
        out["potraightcoverurl"] = DEFAULT_ERROR_IMAGE

    return out


def generate_seo_metadata(chat_url: str, headers: dict, result_json: dict, lang_code: str):
    """Ask the model for SEO metadata in the detected language."""
    lang_code = (lang_code or "").strip() or "auto"
    seo_prompt = f"""
Generate SEO metadata for a web story. Write ALL outputs in this language: {lang_code}

Title: {result_json.get("storytitle","")}
Slides:
- {result_json.get("s1paragraph1","")}
- {result_json.get("s2paragraph1","")}
- {result_json.get("s3paragraph1","")}
- {result_json.get("s4paragraph1","")}
- {result_json.get("s5paragraph1","")}
- {result_json.get("s6paragraph1","")}

Respond strictly in this JSON format:
{{
  "metadescription": "...",
  "metakeywords": "keyword1, keyword2, ..."
}}
"""
    payload_seo = {
        "messages": [
            {"role": "system", "content": "You are an expert SEO assistant. Answer ONLY with valid JSON."},
            {"role": "user", "content": seo_prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 400,
        "response_format": {"type": "json_object"}
    }
    try:
        r = requests.post(chat_url, headers=headers, json=payload_seo, timeout=90)
        if r.status_code != 200:
            return "Explore this insightful story.", "web story, inspiration"
        content = r.json()["choices"][0]["message"]["content"]
        data = robust_parse_model_json(content) or {}
        return data.get("metadescription", "Explore this insightful story."), \
               data.get("metakeywords", "web story, inspiration")
    except Exception:
        return "Explore this insightful story.", "web story, inspiration"


def pick_voice_for_language(lang_code: str, default_voice: str) -> str:
    """Map detected language → Azure voice name."""
    if not lang_code:
        return default_voice
    l = lang_code.lower()
    if l.startswith("hi"):
        return "hi-IN-AaravNeural"
    if l.startswith("en-in"):
        return "en-IN-NeerjaNeural"
    if l.startswith("en"):
        return "en-IN-AaravNeural"
    if l.startswith("bn"):
        return "bn-IN-BashkarNeural"
    if l.startswith("ta"):
        return "ta-IN-PallaviNeural"
    if l.startswith("te"):
        return "te-IN-ShrutiNeural"
    if l.startswith("mr"):
        return "mr-IN-AarohiNeural"
    if l.startswith("gu"):
        return "gu-IN-DhwaniNeural"
    if l.startswith("kn"):
        return "kn-IN-SapnaNeural"
    if l.startswith("pa"):
        return "pa-IN-GeetikaNeural"
    return default_voice


def _voice_to_lang_tag(voice_name: str) -> str:
    """
    Infer SSML lang tag from Azure voice name.
    Examples: "hi-IN-AaravNeural" -> "hi-IN"; "en-IN-NeerjaNeural" -> "en-IN"
    """
    try:
        parts = voice_name.split("-")
        return "-".join(parts[0:2]) if len(parts) >= 2 else "en-US"
    except Exception:
        return "en-US"


def build_ssml(text: str, lang_tag: str, voice: str, rate_pct: int, pitch_semi: int, add_break_ms: int) -> str:
    prosody = f'rate="{rate_pct}%" pitch="{pitch_semi}st"'
    br = f'<break time="{add_break_ms}ms"/>' if add_break_ms > 0 else ""
    # Escape basic XML entities
    def esc(s): return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<speak version="1.0" xml:lang="{lang_tag}">
  <voice name="{voice}">
    <prosody {prosody}>{esc(text)}{br}</prosody>
  </voice>
</speak>"""


def fill_template_strict(template: str, data: dict):
    """Replace {{key}} and {{key|safe}} with string(value). Also return placeholders detected (for missing-report)."""
    placeholders = set(re.findall(r"\{\{\s*([a-zA-Z0-9_\-]+)(?:\|safe)?\s*\}\}", template))
    for k, v in data.items():
        template = template.replace(f"{{{{{k}}}}}", str(v))
        template = template.replace(f"{{{{{k}|safe}}}}", str(v))
    return template, placeholders

# ---- helpers for root uploads/URLs (hard-locked) ----
def _s3_key(name: str) -> str:
    """Always just filename (bucket root)."""
    return name  # HTML S3 uploads at root


def _cdn_url(name: str) -> str:
    """CDN base + filename at root."""
    return f"{CDN_HTML_BASE.rstrip('/')}/{name}"

# ---------- Template validator & publisher integration ----------
RECOMMENDED_KEYS = [
    "storytitle",
    *[f"s{i}paragraph1" for i in range(1,7)],
    *[f"s{i}image1" for i in range(1,7)],
    *[f"s{i}ssml" for i in range(1,7)],
    *[f"s{i}audio_url" for i in range(1,7)],
    *[f"s{i}audio1" for i in range(1,7)],  # add common template alias
    "metadescription", "metakeywords", "publishedtime", "modifiedtime",
    "portraitcoverurl", "potraitcoverurl"
]

def validate_template_placeholders(html_text: str):
    """Return (missing_recommended, extras_found, all_placeholders_set)."""
    found = set(re.findall(r"\{\{\s*([a-zA-Z0-9_\-]+)(?:\|safe)?\s*\}\}", html_text))
    missing = [k for k in RECOMMENDED_KEYS if k not in found]
    extras = [p for p in sorted(found) if p not in RECOMMENDED_KEYS]
    return missing, extras, found

def inject_publisher_meta(html: str, *, site_name: str, canonical_url: str, publisher_name: str,
                          publisher_logo: str, author_name: str) -> str:
    """Insert canonical + JSON-LD into <head>."""
    json_ld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": site_name or "Web Story",
        "author": {"@type": "Person", "name": author_name} if author_name else None,
        "publisher": {
            "@type": "Organization",
            "name": publisher_name or site_name,
            "logo": {"@type": "ImageObject", "url": publisher_logo} if publisher_logo else None
        },
        "mainEntityOfPage": canonical_url,
        "url": canonical_url
    }
    # prune None
    def prune(d):
        if isinstance(d, dict):
            return {k: prune(v) for k, v in d.items() if v is not None}
        return d
    json_ld = prune(json_ld)
    head_inject = f"""
<link rel="canonical" href="{canonical_url}"/>
<meta property="og:url" content="{canonical_url}"/>
<meta property="og:site_name" content="{site_name}"/>
<script type="application/ld+json">{json.dumps(json_ld, ensure_ascii=False)}</script>
"""
    if "</head>" in html:
        return html.replace("</head>", head_inject + "\n</head>")
    return head_inject + html

# ---------------------------
# UI
# ---------------------------
files = st.file_uploader(
    "Upload notes/quiz files (images or PDFs) — multiple allowed",
    type=["jpg", "jpeg", "png", "webp", "tiff", "pdf"],
    accept_multiple_files=True
)
html_files = st.file_uploader(
    "Upload one or more HTML templates (with {{placeholders}})",
    type=["html", "htm"],
    accept_multiple_files=True
)

# ---------- Content quality & depth ----------
with st.expander("🎛️ Content Quality & Subject Depth", expanded=True):
    cols = st.columns(4)
    subject = cols[0].selectbox("Subject", ["General", "Physics", "Chemistry", "Biology", "Mathematics", "History", "Geography", "Civics", "Computer Science", "Economics", "English", "Hindi"], index=0)
    grade = cols[1].selectbox("Grade/Level", ["Generic", "K-8", "9-10", "11-12", "Undergrad", "Professional"], index=0)
    subtopic = cols[2].text_input("Subtopic (optional)", "")
    depth_level = cols[3].slider("Depth level", 1, 5, 3)
    cols2 = st.columns(3)
    difficulty = cols2[0].selectbox("Target difficulty", ["Easy", "Medium", "Hard"], index=1)
    bloom = cols2[1].selectbox("Bloom emphasis", ["Remember", "Understand", "Apply", "Analyze"], index=1)
    tone = cols2[2].selectbox("Tone", ["Neutral", "Conversational", "Exam-focused", "Teacherly"], index=1)

# ---------- SSML / TTS (no narration placeholders) ----------
with st.expander("🗣️ SSML & TTS", expanded=True):
    add_ssml = st.toggle("Generate SSML (intro + each slide)", value=True)
    include_tts = st.toggle("Synthesize audio with Azure Speech (MP3) & upload to S3", value=True,
                            help="Requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION.")
    vcols = st.columns(4)
    voice_override = vcols[0].text_input("Preferred Azure voice (optional)", "")
    ssml_rate = vcols[1].slider("SSML rate (%)", 60, 140, 100)
    ssml_pitch = vcols[2].slider("SSML pitch (semitones)", -6, 6, 0)
    ssml_break = vcols[3].slider("Trailing pause (ms)", 0, 600, 150)

# ---------- Template & Publisher ----------
with st.expander("🧩 Template Fixing & Publisher Integration", expanded=True):
    validate_now = st.button("🔎 Validate uploaded templates")
    inject_publisher = st.toggle("Inject Publisher metadata (canonical + JSON-LD)", value=True)
    pcols = st.columns(4)
    site_name = pcols[0].text_input("Site/Brand name", "Suvichaar Stories")
    canonical_base = pcols[1].text_input("Canonical base (no trailing slash)", CDN_HTML_BASE.rstrip("/"))
    publisher_name = pcols[2].text_input("Publisher", "Suvichaar")
    publisher_logo = pcols[3].text_input("Publisher logo URL", "")

    author_name = st.text_input("Author (optional)", "Suvichaar Team")

    if validate_now and html_files:
        for f in html_files:
            try:
                html_text = f.read().decode("utf-8", errors="replace")
            except Exception:
                st.error(f"Could not read {f.name} as UTF-8.")
                continue
            missing, extras, found = validate_template_placeholders(html_text)
            st.markdown(f"**Template:** `{f.name}`")
            if missing:
                st.error("Missing recommended placeholders: " + ", ".join(missing))
            else:
                st.success("All recommended placeholders found.")
            if extras:
                st.info("Additional placeholders present: " + ", ".join(extras))
            st.divider()

c1, c2, c3, c4 = st.columns(4)
with c1:
    include_seo = st.checkbox("Generate SEO metadata", value=True)
with c2:
    add_time_fields = st.checkbox("Add time fields", value=True, help="Adds {{publishedtime}} and {{modifiedtime}} (UTC ISO).")
with c3:
    vary_images = st.checkbox("Always vary images", value=True, help="Adds a random variation code/style so repeated runs produce different images.")
with c4:
    show_enriched_alts = st.checkbox("Show enriched alt prompts", value=False)

run = st.button("🚀 Run")

if run:
    # Basic validation
    if not files:
        st.error("Please upload at least one notes/quiz file (image or PDF).")
        st.stop()
    if not html_files:
        st.error("Please upload at least one HTML template.")
        st.stop()

    # Preview thumbnails / list PDFs
    with st.expander("📎 Uploaded files"):
        for i, f in enumerate(files, start=1):
            if f.type.startswith("image/"):
                try:
                    img = Image.open(BytesIO(f.getvalue())).convert("RGB")
                    st.image(img, caption=f"File {i}: {f.name}", use_container_width=True)
                except Exception:
                    st.write(f"🖼️ {f.name} (image)")
            else:
                st.write(f"📄 {f.name} (PDF)")

    # -------- OCR with Azure Document Intelligence (multi-file) --------
    with st.spinner("Reading text from all files with Azure Document Intelligence (prebuilt-read)…"):
        raw_text = ocr_many(files)
        if not raw_text:
            st.error("OCR returned no text from any file.")
            st.stop()

    with st.expander("🔎 OCR extracted text"):
        st.write(raw_text[:20000] if raw_text else "(empty)")

    # -------- Language auto-detect (hi/en) --------
    target_lang = detect_hi_or_en(raw_text)
    st.info(f"Target language (auto): **{target_lang}**")

    # -------- Summarize with GPT into JSON (s1..s6 + s1alt..s6alt) --------
    quality_addendum = f"""
Teaching context & quality controls:
- Subject: {subject}; Grade: {grade}; Subtopic: {subtopic or "N/A"}; Depth: {depth_level}/5
- Target difficulty: {difficulty}; Bloom emphasis: {bloom}; Tone: {tone}
Requirements:
- Ensure factual correctness and single coherent storyline across slides.
- Adjust conceptual detail to depth {depth_level}/5 and difficulty '{difficulty}'.
- Keep each slide ≤ 400 chars, concise and unambiguous.
""".strip()

    system_prompt = """
You are a multilingual teaching assistant.

INPUT:
- You will receive raw OCR text from notes/quiz files (could be multiple pages/files).

MANDATORY:
- Target language = "<<LANG>>" (use concise BCP-47 like "en" or "hi").
- Produce ALL text fields strictly in the Target language.

Your job:
1) Extract a short and catchy title → storytitle (Target language)
2) Summarise the content into 6 slides (s1paragraph1..s6paragraph1), each ≤ 400 characters (Target language).
3) For each paragraph (including slide 1), write a DALL·E prompt (s1alt1..s6alt1) for a 1024x1024 flat vector illustration: bright colors, clean lines, no text/captions/logos.

SAFETY & POSITIVITY RULES (MANDATORY):
- If the source includes hate, harassment, violence, adult content, self-harm, illegal acts, or extremist symbols, DO NOT reproduce them.
- Reinterpret into a positive, inclusive, family-friendly, educational scene (unity, learning, empathy, community, peace).
- Replace any hateful/violent symbol with abstract shapes, nature, or neutral motifs.
- Never include real people’s likeness or sensitive groups in a negative way.
- Avoid slogans, gestures, flags, trademarks, or captions. Absolutely NO TEXT in the image.

Respond strictly in this JSON format (keys in English; values in Target language):
{
  "language": "<<LANG>>",
  "storytitle": "...",
  "s1paragraph1": "...",
  "s2paragraph1": "...",
  "s3paragraph1": "...",
  "s4paragraph1": "...",
  "s5paragraph1": "...",
  "s6paragraph1": "...",
  "s1alt1": "...",
  "s2alt1": "...",
  "s3alt1": "...",
  "s4alt1": "...",
  "s5alt1": "...",
  "s6alt1": "..."
}
""".replace("<<LANG>>", target_lang).strip()

    if vary_images:
        system_prompt += f"\n\nVariation hint: style='{_variation_style()}', code='{_variation_token()}'"

    messages = [
        {"role": "system", "content": system_prompt + "\n\n" + quality_addendum},
        {"role": "user", "content": f"OCR TEXT (multi-file):\n{raw_text}\n\nReturn only the JSON object described above."}
    ]

    with st.spinner("Summarizing OCR text with Azure OpenAI…"):
        ok, content = call_azure_chat(messages, temperature=(0.3 if vary_images else 0.0), max_tokens=2200, force_json=True)
        if not ok:
            st.error(content)
            st.stop()

        result = robust_parse_model_json(content)
        if not isinstance(result, dict):
            # One-shot repair
            chat_headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
            chat_url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
            fixed = repair_json_with_model(content, chat_url, chat_headers)
            if isinstance(fixed, dict):
                result = fixed

        if not isinstance(result, dict):
            st.error("Model did not return a valid JSON object.\n\nRaw reply (truncated):\n" + content[:800])
            st.stop()

    # Enforce target language in parsed result
    result["language"] = target_lang
    detected_lang = target_lang
    st.info(f"Detected/target language: **{detected_lang}**")

    # Keep OCR text too (handy for debugging or templates)
    result["ocr_text"] = raw_text

    st.success("Structured JSON created from OCR.")
    st.json({k: result[k] for k in result if k in ["storytitle","s1paragraph1","s2paragraph1","s3paragraph1","s4paragraph1","s5paragraph1","s6paragraph1"]}, expanded=False)

    # -------- Enrich alt prompts (art-director pass) --------
    with st.spinner("Enhancing image prompts (art-director pass)…"):
        result = enrich_alt_prompts_with_model(result, detected_lang)
        if show_enriched_alts:
            st.json({k: result[k] for k in result if re.match(r"s[1-6]alt1$", k)}, expanded=False)

    # -------- DALL·E images → S3 → CDN --------
    with st.spinner("Generating DALL·E images and uploading to S3…"):
        final_json = generate_and_upload_images(result, vary_images=vary_images)

    # -------- SEO metadata (optional) --------
    chat_headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    chat_url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    if include_seo:
        with st.spinner("Generating SEO metadata…"):
            meta_desc, meta_keywords = generate_seo_metadata(chat_url, chat_headers, final_json, detected_lang)
            final_json["metadescription"] = meta_desc
            final_json["metakeywords"] = meta_keywords

    # -------- SSML build (intro + slides) — from title/paragraphs only --------
    if add_ssml:
        chosen_voice = voice_override.strip() or pick_voice_for_language(detected_lang, VOICE_NAME_DEFAULT)
        lang_tag = _voice_to_lang_tag(chosen_voice)
        st.info(f"SSML voice: **{chosen_voice}**  | lang tag: **{lang_tag}**")

        # s1: use title as intro
        intro_text = final_json.get("storytitle") or final_json.get("s1paragraph1") or ""
        final_json["s1ssml"] = build_ssml(intro_text, lang_tag, chosen_voice, ssml_rate, ssml_pitch, ssml_break)

        # s2..s6: slide paragraphs
        for i in range(2, 7):
            text = final_json.get(f"s{i}paragraph1") or ""
            final_json[f"s{i}ssml"] = build_ssml(text, lang_tag, chosen_voice, ssml_rate, ssml_pitch, ssml_break)
    else:
        # make sure ssml keys exist even if not requested
        for i in range(1, 7):
            final_json.setdefault(f"s{i}ssml", "")

    # -------- Optional: TTS (SSML preferred, with plain-text fallback) --------
    if include_tts:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except Exception as e:
            st.error("`azure-cognitiveservices-speech` is not installed. Add it to requirements.txt.\n"
                     f"Import error: {e}")
            st.stop()

        if not (AZURE_SPEECH_KEY and AZURE_SPEECH_REGION):
            st.error("Azure Speech credentials missing. Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in secrets.")
            st.stop()

        s3 = get_s3_client()

        chosen_voice = voice_override.strip() or pick_voice_for_language(detected_lang, VOICE_NAME_DEFAULT)
        st.info(f"TTS voice: **{chosen_voice}**")

        def synth_and_upload(ssml_text: str, fallback_text: str, out_basename: str):
            """Try SSML synth, then fallback to plain text synth. Return (ok, url_or_err)."""
            ts_local = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fname = f"{out_basename}_{ts_local}.mp3"
            temp_path = f"__tmp_{fname}"

            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            speech_config.speech_synthesis_voice_name = chosen_voice
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
            )

            try:
                audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_path)
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
                # Prefer SSML if provided
                if ssml_text:
                    result_tts = synthesizer.speak_ssml_async(ssml_text).get()
                else:
                    result_tts = synthesizer.speak_text_async(fallback_text or "").get()

                from azure.cognitiveservices.speech import ResultReason, CancellationReason
                if result_tts.reason == ResultReason.SynthesizingAudioCompleted:
                    s3_key = f"{S3_PREFIX.rstrip('/')}/audio/{fname}"
                    extra_args = {"ContentType": "audio/mpeg"}
                    s3.upload_file(temp_path, AWS_BUCKET, s3_key, ExtraArgs=extra_args)
                    url = f"{CDN_BASE.rstrip('/')}/{s3_key}"
                    return True, url

                # If canceled or failed, log details and try fallback if we tried SSML first
                if result_tts.reason == ResultReason.Canceled:
                    cd = result_tts.cancellation_details
                    st.error(f"TTS canceled ({out_basename}): reason={cd.reason}; error='{cd.error_details}'")
                else:
                    st.error(f"TTS failed ({out_basename}): reason={result_tts.reason}")

                # Fallback path: if we attempted SSML, try plain text
                if ssml_text and fallback_text:
                    result2 = synthesizer.speak_text_async(fallback_text).get()
                    if result2.reason == ResultReason.SynthesizingAudioCompleted:
                        s3_key = f"{S3_PREFIX.rstrip('/')}/audio/{fname}"
                        extra_args = {"ContentType": "audio/mpeg"}
                        s3.upload_file(temp_path, AWS_BUCKET, s3_key, ExtraArgs=extra_args)
                        url = f"{CDN_BASE.rstrip('/')}/{s3_key}"
                        return True, url
                    else:
                        if result2.reason == ResultReason.Canceled:
                            cd2 = result2.cancellation_details
                            st.error(f"Text fallback canceled ({out_basename}): reason={cd2.reason}; error='{cd2.error_details}'")
                        else:
                            st.error(f"Text fallback failed ({out_basename}): reason={result2.reason}")

                return False, "synthesis failed"

            except Exception as e:
                return False, f"TTS error: {e}"
            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        created_audio = {}
        with st.spinner("Synthesizing audio and uploading to S3…"):
            base_slug = re.sub(r"[^a-z0-9\-]+", "-", (final_json.get("storytitle") or "story").lower()).strip("-")[:80] or "story"

            # Ensure placeholders exist regardless of outcome
            for i in range(1, 7):
                final_json.setdefault(f"s{i}audio_url", "")
                final_json.setdefault(f"s{i}audio1", "")

            # Intro + slides
            tasks = [("s1ssml", "s1audio_url", final_json.get("storytitle") or final_json.get("s1paragraph1") or "", f"{base_slug}_s1")]
            for i in range(2, 7):
                tasks.append((f"s{i}ssml", f"s{i}audio_url", final_json.get(f"s{i}paragraph1") or "", f"{base_slug}_s{i}"))

            for ssml_key, audio_key, fallback_text, out_base in tasks:
                ok_synth, val = synth_and_upload(final_json.get(ssml_key, ""), fallback_text, out_base)
                if ok_synth:
                    final_json[audio_key] = val
                    final_json[audio_key.replace("_url", "1")] = val  # back-compat alias
                    created_audio[ssml_key] = val
                else:
                    st.error(f"TTS failed for: {ssml_key} → {val}")

            if created_audio:
                st.json({"audio_created": created_audio}, expanded=False)

    # -------- Add time fields (optional) --------
    extra_fields = {}
    if add_time_fields:
        iso_now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        extra_fields["publishedtime"] = iso_now
        extra_fields["modifiedtime"] = iso_now

    merged = dict(final_json)
    merged.update(extra_fields)

    # ALSO pre-seed audio placeholders (prevents template warnings even if TTS off/fails)
    for i in range(1, 7):
        merged.setdefault(f"s{i}audio_url", "")
        merged.setdefault(f"s{i}audio1", "")
        merged.setdefault(f"s{i}ssml", merged.get(f"s{i}ssml", ""))

    # -------- Fill templates and offer downloads + S3 upload --------
    def slugify_filename(text: str) -> str:
        s = (text or "webstory").strip().lower()
        s = re.sub(r"[:/\\]+", "-", s)
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_\-\.]", "", s)
        return s or "webstory"

    base_name = slugify_filename(merged.get("storytitle", "webstory"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    filled_items = []  # (filename, filled_html)

    if html_files:
        with st.spinner("Filling HTML templates…"):
            per_file_reports = []

            # Prepare JSON download first
            json_name = f"{base_name}_{ts}.json"
            buf = io.StringIO()
            json.dump(merged, buf, ensure_ascii=False, indent=2)
            json_str = buf.getvalue()

            for idx, f in enumerate(html_files, start=1):
                try:
                    raw_bytes = f.read()
                    html_text = raw_bytes.decode("utf-8")
                except Exception:
                    st.error(f"Could not read {f.name} as UTF-8.")
                    continue

                # Optional publisher injection needs canonical per-file
                out_filename = f"{base_name}_{ts}.html" if len(html_files) == 1 else f"{base_name}_{ts}_{idx}.html"
                canonical_url = f"{canonical_base.rstrip('/')}/{out_filename}"

                # Template fixing check (placeholders that aren't in data)
                _, placeholders = fill_template_strict(html_text, {})  # detect placeholders only
                missing_in_data = sorted([p for p in placeholders if p not in merged])
                if missing_in_data:
                    per_file_reports.append((f.name, missing_in_data))

                # Replace placeholders
                filled, _ = fill_template_strict(html_text, merged)

                # Inject publisher meta (if toggled)
                if inject_publisher:
                    filled = inject_publisher_meta(
                        filled,
                        site_name=site_name or "Web Story",
                        canonical_url=canonical_url,
                        publisher_name=publisher_name or site_name,
                        publisher_logo=publisher_logo or "",
                        author_name=author_name or "",
                    )

                filled_items.append((out_filename, filled))

        if per_file_reports:
            st.warning("Some templates contain placeholders not found in JSON:")
            for name, missing in per_file_reports:
                st.write(f"• **{name}** → missing: {', '.join(missing)}")

        st.success("✅ Templates filled.")

        # Local download buttons
        st.download_button(
            "⬇️ Download Final JSON",
            data=json_str.encode("utf-8"),
            file_name=json_name,
            mime="application/json"
        )

        if len(filled_items) == 1:
            single_name, single_html = filled_items[0]
            st.download_button(
                "⬇️ Download Filled HTML",
                data=single_html.encode("utf-8"),
                file_name=single_name,
                mime="text/html"
            )
        else:
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
                for name, html in filled_items:
                    z.writestr(name, html)
            zip_buf.seek(0)
            st.download_button(
                "⬇️ Download All Filled HTML (ZIP)",
                data=zip_buf.getvalue(),
                file_name=f"{base_name}__filled_{ts}.zip",
                mime="application/zip"
            )

        # ---------- Upload JSON + HTML to S3 (bucket root) and VERIFY ----------
        st.subheader("🌐 Upload to S3 + Verification")
        uploaded_urls = []
        verifications = []

        # 1) Upload JSON (root)
        json_filename = f"{base_name}_{ts}.json"
        json_key = _s3_key(json_filename)  # root
        res_json = s3_put_text_file(
            bucket=AWS_BUCKET,
            key=json_key,
            body=json_str.encode("utf-8"),
            content_type="application/json"
        )
        if res_json["ok"]:
            json_cdn_url = _cdn_url(json_filename)
            uploaded_urls.append(("JSON", json_cdn_url))
        verifications.append({"file": json_filename, **res_json})

        # 2) Upload each HTML file (root) and verify
        for name, html in filled_items:
            html_key = _s3_key(name)
            res_html = s3_put_text_file(
                bucket=AWS_BUCKET,
                key=html_key,
                body=html.encode("utf-8"),
                content_type="text/html; charset=utf-8"
            )
            if res_html["ok"]:
                html_cdn_url = _cdn_url(name)  # https://cdn.suvichaar.org/<file>.html
                uploaded_urls.append(("HTML", html_cdn_url))
            verifications.append({"file": name, **res_html})

        # Show results
        if uploaded_urls:
            for kind, url in uploaded_urls:
                st.markdown(f"- **{kind}**: {url}")
        st.json({"s3_verification": verifications}, expanded=False)

        if uploaded_urls and all(v.get("ok") for v in verifications):
            st.success("✅ Files uploaded to S3 and verified via HEAD. CDN should serve them at the URLs above (allow a short cache/propagation delay).")
        else:
            st.error("Some uploads failed or could not be verified. Check the errors above — common issues: wrong bucket name, IAM permissions (s3:PutObject and s3:HeadObject), or Public Access Block.")

        # ---------------------------
        # FINAL: Live HTML Preview (at the very end)
        # ---------------------------
        st.markdown("### 👀 Live HTML Preview")
        if not filled_items:
            st.info("No filled templates available to preview.")
        else:
            # Collect uploaded HTML URLs (if any)
            uploaded_html_urls = [u for (k, u) in uploaded_urls if k == "HTML"]

            preview_source = st.radio(
                "Choose preview source",
                options=["Local filled HTML", "Uploaded CDN URL"],
                index=0 if filled_items else 1,
                horizontal=True
            )

            if preview_source == "Local filled HTML":
                # Show a select for local HTMLs
                names = [name for name, _ in filled_items]
                choice = st.selectbox("Select local HTML to preview", names, index=0)
                chosen_html = next(html for (name, html) in filled_items if name == choice)
                # Render the HTML directly
                st_html(chosen_html, height=800, scrolling=True)
            else:
                if not uploaded_html_urls:
                    st.info("No uploaded HTML URLs found yet. Upload step might have failed.")
                else:
                    cdn_choice = st.selectbox("Select uploaded CDN URL to preview", uploaded_html_urls, index=0)
                    # Use an iframe to display remote page
                    iframe = f'''
                        <iframe src="{cdn_choice}" width="1600" height="800" style="border:0;"></iframe>
                    '''
                    st_html(iframe, height=820, scrolling=False)

        # (optional) keep your minimal code preview toggle
        show_preview = st.checkbox("Show raw HTML code of first filled template", value=False)
        if show_preview and filled_items:
            st.code(filled_items[0][1][:5000], language="html")
