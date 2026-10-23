"""Streamlit app: watch a page of English handwriting go through the reading pipeline.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud, main file path = app.py

Every step shows what the image looks like at that point and explains what
happened below it. Everything runs inside this app; no external API or
online service is used.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.line_detection import (
    crop_line,
    detect_lines,
    draw_profile_view,
    manual_line_boxes,
    prepare_page,
)
from src.metrics import cer, wer
from src.recognizer import load_recognizer

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

st.set_page_config(page_title="English Handwriting to Text", page_icon="📝", layout="wide")


@st.cache_resource(show_spinner="Loading recognition model...")
def get_recognizer():
    return load_recognizer()


@st.cache_data
def load_examples() -> list[dict]:
    """Read examples/manifest.csv (columns: file, title, ground_truth, source)."""
    manifest = EXAMPLES_DIR / "manifest.csv"
    if not manifest.exists():
        return []
    rows = pd.read_csv(manifest).fillna("")
    examples = []
    for row in rows.itertuples():
        if (EXAMPLES_DIR / row.file).exists():
            examples.append(
                {
                    "file": row.file,
                    "title": row.title,
                    "ground_truth": str(row.ground_truth).replace("\\n", "\n"),
                    "source": row.source,
                }
            )
    return examples


def read_uploaded_image(uploaded) -> np.ndarray:
    """Decode an upload to a BGR array, honouring the phone's EXIF rotation."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(uploaded.getvalue())))
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def step_header(number: int, title: str) -> None:
    st.subheader(f"Step {number}: {title}")


def describe(text: str) -> None:
    """The explanation shown directly below a step's image."""
    st.markdown("> " + text.replace("\n", "\n> "))


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("📝 English Handwriting to Text")
st.caption(
    "Pick an example or upload a photo or scan of English handwritten notes. Below, you can follow "
    "how the page is cleaned up, split into lines and read, one step at a time."
)

recognizer = get_recognizer()
if recognizer is None:
    st.warning(
        "**No trained recognition model is loaded yet.** Steps 1-7 (cleaning up the page and finding "
        "the lines) work. Step 8 (reading the text) stays disabled until a model checkpoint is added. "
        "No text is faked."
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

# ----------------------------------------------------------------------------
# Input: example or upload
# ----------------------------------------------------------------------------
examples = load_examples()
choices = (["Example page", "Upload my own"] if examples else ["Upload my own"])
source = st.radio("Input", choices, horizontal=True)

page_bgr: np.ndarray | None = None
ground_truth: str | None = None

if source == "Example page":
    titles = [e["title"] for e in examples]
    chosen = examples[titles.index(st.selectbox("Example", titles))]
    page_bgr = cv2.imread(str(EXAMPLES_DIR / chosen["file"]))
    ground_truth = chosen["ground_truth"] or None
    st.caption(f"Source: {chosen['source']}")
else:
    uploaded = st.file_uploader("Upload a page image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None:
        page_bgr = read_uploaded_image(uploaded)

if page_bgr is None:
    st.info("Upload an image to begin.")
    st.stop()

prepared = prepare_page(page_bgr, deskew=deskew)
orig_h, orig_w = page_bgr.shape[:2]
page_h, page_w = prepared.gray.shape

if mode == "Automatic":
    boxes = detect_lines(
        prepared.ink,
        ink_fraction=ink_fraction,
        min_height_fraction=min_height,
        merge_gap_fraction=merge_gap,
    )
else:
    st.sidebar.caption(f"Enter y-ranges in pixels (page height is {page_h}px). Compare with the numbered preview in step 6.")
    auto_guess = detect_lines(prepared.ink)
    default_rows = pd.DataFrame(
        [{"y_start": b.y0, "y_end": b.y1} for b in auto_guess] or [{"y_start": 0, "y_end": 0}]
    )
    edited = st.sidebar.data_editor(default_rows, num_rows="dynamic", key="manual_rows")
    ranges = [(int(r.y_start), int(r.y_end)) for r in edited.dropna().itertuples()]
    boxes = manual_line_boxes(prepared.ink, ranges)

# ----------------------------------------------------------------------------
# Step 1: original
# ----------------------------------------------------------------------------
with st.container(border=True):
    step_header(1, "The image we received")
    st.image(cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB), width="stretch")
    resized_note = (
        f" It was larger than 2000 px on its longest side, so it is shrunk to {prepared.gray_raw.shape[1]}×"
        f"{prepared.gray_raw.shape[0]} px before processing to keep things fast."
        if prepared.gray_raw.shape[:2] != (orig_h, orig_w)
        else ""
    )
    describe(
        f"This is the page exactly as uploaded: **{orig_w}×{orig_h} pixels**, in color.{resized_note}\n"
        "Photos of notes are rarely perfect: the page may be tilted, lit unevenly or have shadows. "
        "The next steps fix those problems so the handwriting is easy to find."
    )

# ----------------------------------------------------------------------------
# Step 2: grayscale
# ----------------------------------------------------------------------------
with st.container(border=True):
    step_header(2, "Convert to grayscale")
    st.image(prepared.gray_raw, width="stretch")
    describe(
        "Handwriting is a shape, not a color, so the three color channels (red, green, blue) are merged into "
        "one brightness value per pixel: **0 is black, 255 is white**. OpenCV does this with a weighted average "
        "(0.299 R + 0.587 G + 0.114 B) that matches how bright each color looks to the human eye. "
        "One channel instead of three makes everything after this smaller and faster, including the model."
    )

# ----------------------------------------------------------------------------
# Step 3: lighting
# ----------------------------------------------------------------------------
with st.container(border=True):
    step_header(3, "Remove shadows and uneven lighting")
    left, right = st.columns(2)
    left.image(prepared.background, caption="Estimated paper brightness (ink erased)", width="stretch")
    right.image(prepared.flat, caption="Page divided by that brightness", width="stretch")
    bg_low, bg_high = np.percentile(prepared.background, [1, 99])
    describe(
        "A shadow can make the paper in one corner as dark as the pen strokes in another, so a single "
        "brightness cut-off would fail. To fix this, the app first **erases the ink** with a large morphological "
        "'close' filter, leaving only an estimate of the paper (left). "
        f"On this page the paper brightness ranges from about **{bg_low:.0f} to {bg_high:.0f}** (out of 255). "
        "Dividing the page by that estimate makes the paper uniformly white while the ink stays dark (right)."
    )

# ----------------------------------------------------------------------------
# Step 4: ink mask
# ----------------------------------------------------------------------------
with st.container(border=True):
    step_header(4, "Separate ink from paper")
    left, right = st.columns(2)
    left.image(prepared.flat, caption="Before", width="stretch")
    right.image(prepared.ink_raw, caption="Ink mask (white = ink)", width="stretch")
    ink_pct = 100 * float((prepared.ink_raw > 0).mean())
    describe(
        "**Otsu's method** looks at the histogram of brightness values and automatically picks the cut-off that "
        "best separates the dark ink pixels from the light paper pixels, so no threshold has to be tuned by hand. "
        "Pixels darker than the cut-off become white in the mask. A light blur before the cut and a median filter "
        f"after it remove isolated specks. Here, **{ink_pct:.1f}%** of the page is ink."
    )

# ----------------------------------------------------------------------------
# Step 5: straighten
# ----------------------------------------------------------------------------
with st.container(border=True):
    step_header(5, "Straighten the page")
    if deskew and prepared.skew_scores is not None:
        left, right = st.columns(2)
        left.image(prepared.gray_raw, caption="Before", width="stretch")
        right.image(prepared.gray, caption=f"After (rotated {prepared.skew_angle:+.1f}°)", width="stretch")
        scores = prepared.skew_scores / max(prepared.skew_scores.max(), 1e-9)
        chart = pd.DataFrame({"line sharpness": scores}, index=pd.Index(prepared.skew_angles, name="trial rotation (degrees)"))
        st.line_chart(chart, height=180)
        describe(
            "The app tries rotations from **-10° to +10° in 0.5° steps** and, for each, adds up the ink in every pixel "
            "row. When the text lines are perfectly horizontal, rows inside a line hold a lot of ink and the gaps "
            "between lines hold almost none, so the row totals vary as much as possible. The chart plots that "
            f"'sharpness' for each trial angle; it peaks at **{prepared.skew_angle:+.1f}°**, so that is the rotation applied. "
            "This only corrects a tilt of up to 10° and assumes the text runs roughly horizontally."
        )
    else:
        st.image(prepared.gray, width="stretch")
        describe("Straightening is switched off in the sidebar, so the page is used as it is.")

# ----------------------------------------------------------------------------
# Step 6: find lines
# ----------------------------------------------------------------------------
with st.container(border=True):
    step_header(6, "Find the lines of text")
    if not boxes:
        st.error("No text lines found. Try a lower ink sensitivity or use manual boundaries.")
    else:
        st.image(draw_profile_view(prepared.gray, prepared.ink, boxes, ink_fraction), width="stretch")
        heights = [b.height for b in boxes]
        how = (
            "The right-hand graph is the **projection profile**: for every pixel row of the page, a bar shows how much "
            "ink that row contains. Rows whose bar passes the **red threshold line** are counted as text. Consecutive text "
            "rows form a line, small gaps are merged (so dots and descenders stay with their line), and runs that are too "
            "short are dropped as noise. "
            if mode == "Automatic"
            else "These lines come from the **y-ranges you entered in the sidebar**; the horizontal extent of each is found "
            "from the ink inside it. The graph on the right shows the ink amount per row to help you choose boundaries. "
        )
        describe(
            f"{how}\n**{len(boxes)} lines** were found, each between {min(heights)} and {max(heights)} px tall, "
            "numbered top to bottom, which is the reading order for normal notes. "
            "This method struggles when lines overlap or the paper is ruled; use manual boundaries if it makes a mistake."
        )

# ----------------------------------------------------------------------------
# Step 7: crop lines
# ----------------------------------------------------------------------------
crops: list[np.ndarray] = []
with st.container(border=True):
    step_header(7, "Cut out each line")
    if not boxes:
        st.info("No lines to cut out.")
    else:
        crops = [crop_line(prepared.gray, b) for b in boxes]
        for i, crop in enumerate(crops, start=1):
            st.image(crop, caption=f"Line {i} ({crop.shape[1]}×{crop.shape[0]} px)")
        describe(
            "Each line is cropped from the straightened page with a small margin so no stroke is cut off. "
            "These strips, not the whole page, are what the recognizer reads. Because every strip has a different "
            "width, they will be resized to a fixed height and padded to a common width before being fed to the model."
        )

# ----------------------------------------------------------------------------
# Step 8: recognise
# ----------------------------------------------------------------------------
with st.container(border=True):
    step_header(8, "Read the text")
    if not crops:
        st.info("Nothing to read yet: no lines were detected.")
    elif recognizer is None:
        st.info("Text recognition is unavailable until a trained model checkpoint is added.")
        if ground_truth:
            st.text_area("Text this example should produce (ground truth)", ground_truth, height=170, disabled=True)
        describe(
            "How the model will read each strip: a **CNN** looks at the picture and turns it into a sequence of "
            "feature vectors, one for each small horizontal slice. A **bidirectional LSTM** reads that sequence left-to-right "
            "and right-to-left so each slice knows its neighbours. A final layer gives, for every slice, a probability for each "
            "character plus a 'blank'. **CTC** then collapses repeated characters and blanks into the final text, which lets the "
            "model learn from whole-line transcriptions without needing to know where each letter starts."
        )
    else:
        with st.spinner("Recognising lines..."):
            lines = recognizer(crops)
        text = st.text_area("Recognised text (editable)", "\n".join(lines), height=260)
        st.download_button("Download as .txt", text, file_name="handwriting.txt", mime="text/plain")
        if ground_truth:
            st.text_area("Expected text (ground truth)", ground_truth, height=170, disabled=True)
            joined = "\n".join(lines)
            st.metric("Character error rate (CER)", f"{100 * cer(ground_truth, joined):.1f}%")
            st.metric("Word error rate (WER)", f"{100 * wer(ground_truth, joined):.1f}%")
            if len(lines) != len(ground_truth.split("\n")):
                st.warning("The number of detected lines differs from the expected number, which inflates the error rates.")
        describe(
            "Each line strip was passed through the model and decoded greedily (the most likely character at every "
            "position, then repeats and blanks removed). The lines are joined in reading order. You can edit the text "
            "above before downloading it. The error rates are only shown for examples that have a known ground truth."
        )
