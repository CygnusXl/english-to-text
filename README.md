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

## Dataset attribution

OpenHand-Synth: https://huggingface.co/datasets/to-be/OpenHand-Synth
(license and citation to be confirmed from the dataset card in Milestone 2).

## AI assistance

This project was developed with help from an AI assistant (Claude). A
detailed log will be kept in `docs/ai_usage_log.md` to match university
requirements.
