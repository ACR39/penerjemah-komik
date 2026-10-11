import base64
import io
import json
import re
import zipfile
from functools import lru_cache

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

st.set_page_config(page_title="Penerjemah Komik", page_icon="🈂️", layout="wide")

MODELS = ["gpt-5.6", "gpt-5.6-luna", "glm-5.3-mod", "glm-5.3-flash-mod", "glm-5.3-flashx"]
FONTS = [
    "fonts/ComicNeue-Bold.ttf",  # opsional: taruh font sendiri di folder fonts/
    "/usr/share/fonts/truetype/comic-neue/ComicNeue-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/comicbd.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# ---------------------------------------------------------------- gambar
@lru_cache(maxsize=64)
def load_font(size):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap(draw, text, font, max_w):
    lines = []
    for par in text.split("\n"):
        line = ""
        for word in par.split():
            t = f"{line} {word}".strip()
            if draw.textlength(t, font=font) <= max_w or not line:
                line = t
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def fit_text(draw, text, w, h):
    start = max(7, min(int(h), 60))
    for size in range(start, 6, -1):
        font = load_font(size)
        lines = wrap(draw, text, font, w)
        lh = size * 1.12
        widest = max(draw.textlength(l, font=font) for l in lines)
        if len(lines) * lh <= h and widest <= w:
            break
    return font, lines, lh


def render(img, regions, pad=0.08, shape="Oval", show_boxes=False):
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    W, H = out.size
    for r in regions:
        if not r["aktif"]:
            continue
        x1, y1, x2, y2 = r["x1"] / 1000 * W, r["y1"] / 1000 * H, r["x2"] / 1000 * W, r["y2"] / 1000 * H
        w, h = x2 - x1, y2 - y1
        bx1, by1, bx2, by2 = x1 - w * pad, y1 - h * pad, x2 + w * pad, y2 + h * pad
        bw, bh = bx2 - bx1, by2 - by1
        cx, cy = bx1 + bw / 2, by1 + bh / 2
        if shape == "Oval":
            ex, ey = bw / 2 * 1.08, bh / 2 * 1.08
            d.ellipse([cx - ex, cy - ey, cx + ex, cy + ey], fill="white")
            tf = 0.78
        else:
            d.rounded_rectangle([bx1, by1, bx2, by2], radius=min(bw, bh) * 0.12, fill="white")
            tf = 0.94
        text = r["terjemahan"]
        if text.strip():
            font, lines, lh = fit_text(d, text, bw * tf, bh * tf)
            y0 = cy - (len(lines) - 1) * lh / 2
            for i, line in enumerate(lines):
                d.text((cx, y0 + i * lh), line, font=font, fill="black", anchor="mm")
        if show_boxes:
            d.rectangle([x1, y1, x2, y2], outline="#e11d48", width=max(2, W // 500))
    return out


def to_png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def to_b64(img, edge=1568):
    im = img.convert("RGB")
    s = min(1, edge / max(im.size))
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- API
def build_prompt(lang, rtl, sfx):
    sfx_line = (
        ", and sound effects"
        if sfx
        else ". Ignore sound effects and background signage unless essential to the story"
    )
    order = "right-to-left, top-to-bottom" if rtl else "left-to-right, top-to-bottom"
    return f"""You are a professional comic/manga translator and typesetter. Look at this comic page and find every piece of text that should be translated: speech bubbles, thought bubbles, narration boxes{sfx_line}.

For each text region return:
- "box": [x1, y1, x2, y2] bounding box of the ORIGINAL text area (the text block inside the bubble/box, not the entire bubble art), normalized to a 0-1000 grid (x1,y1 = top-left; x2,y2 = bottom-right; 0,0 is the top-left corner of the image, 1000,1000 the bottom-right).
- "original": the original text, transcribed.
- "translation": natural, fluent translation into {lang}, keeping tone, humor and character voice. Keep it concise enough to fit in the same bubble.

Order the regions in reading order ({order}). Merge lines belonging to the same bubble into one region.

Return ONLY a valid JSON array, no markdown, no commentary. If there is no text, return []."""


def endpoint(base, fmt):
    base = base.strip().rstrip("/")
    if not base:
        raise RuntimeError("Isi Base URL gateway dulu.")
    if re.search(r"/(chat/completions|messages)$", base):
        return base
    return base + ("/chat/completions" if fmt == "openai" else "/messages")


def call_api(img, cfg, lang, rtl, sfx):
    if not cfg["key"]:
        raise RuntimeError("Isi API key dulu.")
    url = endpoint(cfg["base"], cfg["fmt"])
    b64 = to_b64(img)
    prompt = build_prompt(lang, rtl, sfx)
    if cfg["fmt"] == "openai":
        headers = {"Authorization": f"Bearer {cfg['key']}"}
        body = {
            "model": cfg["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                    ],
                }
            ],
        }
    else:
        headers = {"x-api-key": cfg["key"], "anthropic-version": "2023-06-01"}
        body = {
            "model": cfg["model"],
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
    r = requests.post(url, headers=headers, json=body, timeout=180)
    try:
        j = r.json()
    except ValueError:
        j = {}
    if not r.ok:
        err = j.get("error", j.get("message", f"HTTP {r.status_code}"))
        raise RuntimeError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
    if cfg["fmt"] == "openai":
        c = j["choices"][0]["message"]["content"]
        txt = "".join(x.get("text", "") for x in c) if isinstance(c, list) else (c or "")
    else:
        txt = "".join(b.get("text", "") for b in j.get("content", []))
    m = re.search(r"\[[\s\S]*\]", txt)
    if not m:
        raise RuntimeError("Respons model tidak berisi JSON.")
    regions = []
    for item in json.loads(m.group(0)):
        box = item.get("box")
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        regions.append(
            {
                "aktif": True,
                "asli": str(item.get("original", "")),
                "terjemahan": str(item.get("translation", "")),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            }
        )
    return regions


def clean(df):
    out = []
    for rec in df.to_dict("records"):
        try:
            c = [float(rec[k]) for k in ("x1", "y1", "x2", "y2")]
        except (TypeError, ValueError, KeyError):
            continue
        if any(pd.isna(v) for v in c):
            continue
        t = rec.get("terjemahan")
        out.append(
            {
                "aktif": bool(rec.get("aktif")) if rec.get("aktif") in (True, False) else True,
                "asli": "" if pd.isna(rec.get("asli")) else str(rec.get("asli")),
                "terjemahan": "" if pd.isna(t) else str(t),
                "x1": c[0], "y1": c[1], "x2": c[2], "y2": c[3],
            }
        )
    return out


# ---------------------------------------------------------------- UI
st.title("🈂️ Penerjemah Komik")

with st.sidebar:
    st.header("Pengaturan")
    fmt_label = st.selectbox(
        "Format API gateway", ["OpenAI-compatible (chat/completions)", "Anthropic (messages)"]
    )
    fmt = "openai" if fmt_label.startswith("OpenAI") else "anthropic"
    base = st.text_input("Base URL gateway", value=secret("BASE_URL"), placeholder="https://alamat-gateway/v1")
    key = st.text_input("API key", value=secret("API_KEY"), type="password")
    model_pick = st.selectbox("Model", MODELS)
    model_custom = st.text_input("Model lain (opsional)")
    model = model_custom.strip() or model_pick
    lang = st.selectbox(
        "Bahasa tujuan",
        ["Indonesian", "English", "Japanese", "Korean", "Chinese (Simplified)", "Spanish"],
    )
    rtl = st.radio("Arah baca", ["Kiri ke kanan", "Kanan ke kiri (manga)"]) != "Kiri ke kanan"
    sfx = st.checkbox("Terjemahkan juga efek suara (SFX)", value=False)
    st.divider()
    shape = st.radio("Bentuk penutup", ["Oval", "Kotak"], horizontal=True)
    pad = st.slider("Padding penutup (%)", 0, 30, 8) / 100
    show_boxes = st.checkbox("Tampilkan kotak deteksi")

cfg = {"fmt": fmt, "base": base, "key": key, "model": model}

files = st.file_uploader(
    "Unggah halaman komik", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
)
if not files:
    st.info("Unggah satu atau beberapa halaman komik untuk mulai.")
    st.stop()

files = sorted(files, key=lambda f: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", f.name)])
if "pages" not in st.session_state:
    st.session_state.pages = {}

images, ids = {}, []
for f in files:
    pid = f"{f.name}:{f.size}"
    ids.append(pid)
    images[pid] = ImageOps.exif_transpose(Image.open(f))
    st.session_state.pages.setdefault(pid, {"regions": [], "edited": None, "ver": 0})

names = {pid: f"{i + 1}. {pid.rsplit(':', 1)[0]}" for i, pid in enumerate(ids)}
current = st.selectbox("Halaman", ids, format_func=lambda p: names[p])


def translate(pid):
    page = st.session_state.pages[pid]
    page["regions"] = call_api(images[pid], cfg, lang, rtl, sfx)
    page["edited"] = None
    page["ver"] += 1


c1, c2, c3 = st.columns([1, 1, 3])
if c1.button("Terjemahkan halaman ini", type="primary"):
    try:
        with st.spinner("Menerjemahkan..."):
            translate(current)
    except Exception as e:
        st.error(f"Gagal: {e}")
if c2.button("Terjemahkan semua"):
    bar = st.progress(0.0)
    for i, pid in enumerate(ids):
        try:
            translate(pid)
        except Exception as e:
            st.error(f"{names[pid]}: {e}")
            break
        bar.progress((i + 1) / len(ids))

page = st.session_state.pages[current]
left, right = st.columns([3, 2])

with right:
    st.subheader("Teks terdeteksi")
    cols = ["aktif", "asli", "terjemahan", "x1", "y1", "x2", "y2"]
    df = pd.DataFrame(page["regions"], columns=cols)
    edited = st.data_editor(
        df,
        key=f"ed_{current}_{page['ver']}",
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "aktif": st.column_config.CheckboxColumn("Aktif", default=True, width="small"),
            "asli": "Asli",
            "terjemahan": st.column_config.TextColumn("Terjemahan", width="medium"),
            "x1": st.column_config.NumberColumn("x1", min_value=0, max_value=1000, default=400),
            "y1": st.column_config.NumberColumn("y1", min_value=0, max_value=1000, default=400),
            "x2": st.column_config.NumberColumn("x2", min_value=0, max_value=1000, default=600),
            "y2": st.column_config.NumberColumn("y2", min_value=0, max_value=1000, default=500),
        },
    )
    page["edited"] = clean(edited)
    st.caption("Koordinat memakai skala 0-1000 dari pojok kiri atas. Kamu bisa menyunting teks dan kotak, atau menambah baris baru.")

result = render(images[current], page["edited"], pad, shape, show_boxes)
with left:
    st.image(result, use_container_width=True)
    st.download_button(
        "Unduh PNG halaman ini",
        to_png(result),
        file_name=pid_name if (pid_name := current.rsplit(":", 1)[0].rsplit(".", 1)[0] + "_terjemahan.png") else "hasil.png",
        mime="image/png",
    )

st.divider()
if st.button("Siapkan ZIP semua halaman"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for pid in ids:
            p = st.session_state.pages[pid]
            regs = p["edited"] if p["edited"] is not None else p["regions"]
            out = render(images[pid], regs, pad, shape, False)
            z.writestr(pid.rsplit(":", 1)[0].rsplit(".", 1)[0] + "_terjemahan.png", to_png(out))
    st.session_state.zip = buf.getvalue()
if st.session_state.get("zip"):
    st.download_button("Unduh ZIP", st.session_state.zip, file_name="komik_terjemahan.zip", mime="application/zip")
