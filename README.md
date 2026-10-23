# english-to-text

English handwriting-to-text with a custom CRNN-CTC model (PyTorch).
University Deep Learning project: a full-page photo or scan of English
handwritten notes goes in, editable text comes out.

> **Status: Milestone 1 (environment inspection).** Nothing is trained yet.
> No accuracy numbers exist, and none will be claimed until they are measured.

## Goal

1. Preprocess a page image (OpenCV).
2. Detect and crop text lines.
3. Recognize each line with a CNN + BiLSTM + CTC model trained from scratch.
4. Join the lines in reading order and return text.

## Limitations (to be extended as we learn more)

- English only, conventional left-to-right, top-to-bottom notes.
- The first training data (OpenHand-Synth) is synthetic, so results on real
  handwriting must be measured separately on real pages.
- Not every handwriting style is supported.

## Project structure

```
data/            raw, processed, real_page_test (kept out of Git)
notebooks/       dataset analysis
scripts/         helper scripts (check_env.py)
src/             dataset, preprocessing, line_detection, vocabulary,
                 model, train, decode, evaluate, page_pipeline
tests/           unit tests
checkpoints/     model weights (kept out of Git)
results/         curves, tables, example predictions
```

Files in `src/` and `tests/` are added milestone by milestone.

## Milestone 1: inspect your environment

```bash
python3 scripts/check_env.py
```

The script only reads information; it installs nothing.

## Web app (Streamlit)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Current state: page straightening and line detection work. Text output is
disabled until a trained checkpoint exists at `checkpoints/best.pt`; the app
shows a notice instead of inventing text. No external API is used.

Run the tests with `pip install pytest` then `python3 -m pytest tests -v`.
The line-detection tests use a synthetic page, so they check the geometry
code, not real handwriting accuracy.

### Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. In Streamlit Community Cloud choose **New app**, pick this repo, branch
   `main`, and main file path `app.py`.
3. Once a model is trained, the weights must be in the repo for the deployed
   app to see them. `checkpoints/*.pt` is git-ignored, so add the final model
   on purpose: `git add -f checkpoints/best.pt` (keep it under GitHub's
   100 MB file limit, or use Git LFS). Also add `torch` to `requirements.txt`
   at that point.

## Dataset attribution

OpenHand-Synth: https://huggingface.co/datasets/to-be/OpenHand-Synth
(license and citation to be confirmed from the dataset card in Milestone 2).

## AI assistance

This project was developed with help from an AI assistant (Claude). A
detailed log will be kept in `docs/ai_usage_log.md` to match university
requirements.
