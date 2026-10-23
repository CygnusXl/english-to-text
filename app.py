"""Streamlit app: upload a page of English handwriting, get editable text.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud, main file path = app.py

Everything runs inside this app. No external API or online service is used.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.line_detection import (
    crop_line,
    detect_lines,
    draw_boxes,
    manual_line_boxes,
    prepare_page,
)
from src.recognizer import load_recognizer

st.set_page_config(page_title="English Handwriting to Text", page_icon="📝", layout="wide")


@st.cache_resource(show_spinner="Loading recognition model...")
def get_recognizer():
    return load_recognizer()


def read_uploaded_image(uploaded) -> np.ndarray:
    """Decode an upload to a BGR array, honouring the phone's EXIF rotation."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(uploaded.getvalue())))
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


st.title("📝 English Handwriting to Text")
st.caption(
    "Upload a photo or scan of a page of English handwritten notes. "
    "The page is straightened, split into lines, and each line is read by a "
    "CRNN-CTC model trained from scratch."
)

recognizer = get_recognizer()
if recognizer is None:
    st.warning(
        "**No trained recognition model is loaded yet.** Page straightening and "
        "line detection work, but text output stays disabled until a model "
        "checkpoint is added. No text is faked."
    )

with st.expander("Limitations"):
    st.markdown(
        "- English only; conventional left-to-right, top-to-bottom notes.\n"
        "- The first model is trained on synthetic handwriting, so accuracy on real, "
        "messy handwriting can be much lower. Check the output before relying on it.\n"
        "- Automatic line detection can fail on overlapping lines, ruled paper, "
        "shadows or curved pages. Use manual line boundaries in the sidebar if it does.\n"
        "- Not every handwriting style is supported."
    )

with st.sidebar:
    st.header("Settings")
    deskew = st.checkbox("Straighten tilted page", value=True)
    mode = st.radio("Line detection", ["Automatic", "Manual boundaries"])
    ink_fraction = st.slider("Ink sensitivity", 0.01, 0.30, 0.05, 0.01, help="Lower = picks up fainter lines.")
    min_height = st.slider("Minimum line height (% of page)", 0.3, 5.0, 1.2, 0.1) / 100
    merge_gap = st.slider("Merge gap (% of page)", 0.1, 3.0, 0.6, 0.1) / 100

uploaded = st.file_uploader("Upload a page image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

page_bgr = read_uploaded_image(uploaded)
prepared = prepare_page(page_bgr, deskew=deskew)

if mode == "Automatic":
    boxes = detect_lines(
        prepared.ink,
        ink_fraction=ink_fraction,
        min_height_fraction=min_height,
        merge_gap_fraction=merge_gap,
    )
else:
    page_h = prepared.gray.shape[0]
    st.sidebar.caption(f"Enter y-ranges in pixels (page height is {page_h}px). Compare with the numbered preview.")
    auto_guess = detect_lines(prepared.ink)
    default_rows = pd.DataFrame(
        [{"y_start": b.y0, "y_end": b.y1} for b in auto_guess] or [{"y_start": 0, "y_end": 0}]
    )
    edited = st.sidebar.data_editor(default_rows, num_rows="dynamic", key="manual_rows")
    ranges = [(int(r.y_start), int(r.y_end)) for r in edited.dropna().itertuples()]
    boxes = manual_line_boxes(prepared.ink, ranges)

tab_pre, tab_lines, tab_text = st.tabs(["1. Preprocessing", "2. Detected lines", "3. Text"])

with tab_pre:
    left, right = st.columns(2)
    left.image(cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB), caption="Original", width="stretch")
    right.image(prepared.gray, caption=f"Straightened (rotated {prepared.skew_angle:+.1f}°)", width="stretch")
    with st.expander("Ink mask used for line detection"):
        st.image(prepared.ink, width="stretch")

with tab_lines:
    if not boxes:
        st.error("No text lines found. Try a lower ink sensitivity or use manual boundaries.")
    else:
        st.write(f"**{len(boxes)} lines detected**, numbered in reading order.")
        st.image(draw_boxes(prepared.gray, boxes), width="stretch")
        with st.expander("Individual line crops"):
            for i, box in enumerate(boxes, start=1):
                st.image(crop_line(prepared.gray, box), caption=f"Line {i}")

with tab_text:
    if not boxes:
        st.info("Nothing to read yet: no lines were detected.")
    elif recognizer is None:
        st.info("Text recognition is unavailable until a trained model checkpoint is added.")
    else:
        crops = [crop_line(prepared.gray, b) for b in boxes]
        with st.spinner("Recognising lines..."):
            lines = recognizer(crops)
        text = st.text_area("Recognised text (editable)", "\n".join(lines), height=300)
        st.download_button("Download as .txt", text, file_name="handwriting.txt", mime="text/plain")
