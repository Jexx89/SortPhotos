# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
# Streamlit GUI
.venv/Scripts/streamlit run app.py

# CLI (headless)
PYTHONIOENCODING=utf-8 python sort_photos.py [SORT_FOLDER] --example NAME=PATH --mode move|copy --threshold 0.30 --margin 0.05 --dry-run

# CLI with tkinter dialogs (no args = interactive folder/example pickers)
PYTHONIOENCODING=utf-8 python sort_photos.py
```

## Architecture

Three files, no framework beyond Streamlit:

- **`classifier.py`** — Pure computation, no I/O. `extract_signature()` produces a 432-value weighted feature vector (48 color histogram + 256 grayscale thumbnail + 128 gradient texture blocks). `build_references()` averages signatures per class. `classify()` returns cosine distance ranking and a `(decision, best_cls, best_score, runner_cls, runner_score)` tuple — `decision` is `None` when the image goes to `unknown/`.

- **`app.py`** — Streamlit GUI. Wraps `classifier.py`. Manages class definitions in `st.session_state`. Writes a per-run `sort_photos.log` in the sort folder and a persistent `log/app.log` for crash diagnostics.

- **`sort_photos.py`** — CLI entry point. Same classification logic as the GUI; uses tkinter dialogs when arguments are omitted.

## Critical Streamlit constraints

- **No Streamlit calls inside the processing loop** — any `st.progress()` or `st.write()` inside the loop triggers a `RerunException` that aborts mid-sort. All Streamlit updates happen before and after the `st.spinner` block.
- **`st.image` takes raw bytes, not a PIL `Image` object** — `Image.open()` is lazy; the underlying `UploadedFile` stream is closed by the time Streamlit reads pixels, causing a C-level crash.
- **`.streamlit/config.toml`** sets `fileWatcherType = "none"` (file moves during sort would restart the server mid-run) and `scriptRunTimeoutSeconds = 0` (large batches exceed the default timeout).

## Classifier tuning

Two parameters control the `unknown/` bucket:
- `threshold` — cosine distance above which a photo is unconditionally unknown (default 0.30)
- `margin` — minimum gap between top-2 class scores; below this the result is considered ambiguous and sent to unknown (default 0.05)

Lower `threshold` = stricter matching. The dry-run mode shows score distributions per class to help calibrate both parameters.

## Environment

- Python via `.venv/` — use `python`, never `python3`
- Always prefix with `PYTHONIOENCODING=utf-8` to avoid cp1252 errors on Windows
