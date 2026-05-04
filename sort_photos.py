#!/usr/bin/env python3
"""
sort_photos.py — Sort photos by visual similarity to example images.

Usage:
    python sort_photos.py [SORT_FOLDER] [--example NAME=PATH ...] [options]

If SORT_FOLDER is omitted, a folder picker dialog opens.
If --example is omitted, an interactive dialog opens to pick examples.

Options:
    --example NAME=PATH    Reference image (repeat for each class; NAME=p1,p2 for multi-example)
    --mode move|copy       File operation (default: move)
    --threshold FLOAT      Cosine distance above which photo goes to unknown/ (default: 0.30)
    --margin FLOAT         Ambiguity margin between top-2 classes (default: 0.05)
    --dry-run              Analyse only — no files moved or copied
    --log-file PATH        Log path (default: SORT_FOLDER/sort_photos.log)
"""

import argparse
import logging
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from classifier import build_references, classify, extract_signature

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _pick_folder() -> Path:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder to sort")
    root.destroy()
    if not folder:
        print("No folder selected. Exiting.")
        sys.exit(0)
    return Path(folder)


def _pick_examples() -> dict[str, list[str]]:
    import tkinter as tk
    from tkinter import filedialog, simpledialog

    root = tk.Tk()
    root.withdraw()
    examples: dict[str, list[str]] = {}
    while True:
        name = simpledialog.askstring(
            "Class name",
            "Class name (leave empty to finish):",
            parent=root,
        )
        if not name:
            break
        files = filedialog.askopenfilenames(
            title=f"Example image(s) for '{name}'",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")],
        )
        if files:
            examples[name] = list(files)
    root.destroy()
    if not examples:
        print("No examples provided. Exiting.")
        sys.exit(0)
    return examples


def _setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("sort_photos")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def _collect_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _safe_dest(dest: Path) -> Path:
    """Return dest unchanged if it doesn't exist, else dest_001, dest_002, …"""
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 1
    while True:
        candidate = dest.with_name(f"{stem}_{i:03d}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def _parse_example(value: str) -> tuple[str, list[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got: {value}")
    name, paths_str = value.split("=", 1)
    paths = [p.strip() for p in paths_str.split(",") if p.strip()]
    return name.strip(), paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sort photos by visual similarity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("sort_folder", nargs="?", help="Folder to sort")
    parser.add_argument("--example", action="append", metavar="NAME=PATH")
    parser.add_argument("--mode", choices=["move", "copy"], default="move")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file")
    args = parser.parse_args()

    sort_folder = Path(args.sort_folder) if args.sort_folder else _pick_folder()
    if not sort_folder.exists():
        sys.exit(f"Folder not found: {sort_folder}")

    log_file = Path(args.log_file) if args.log_file else sort_folder / "sort_photos.log"
    logger = _setup_logger(log_file)

    raw_examples: dict[str, list[str]] = {}
    if args.example:
        for ex in args.example:
            name, paths = _parse_example(ex)
            raw_examples.setdefault(name, []).extend(paths)
    else:
        raw_examples = _pick_examples()

    images = _collect_images(sort_folder)
    if not images:
        logger.info("No images found in %s. Exiting.", sort_folder)
        sys.exit(0)

    classes = list(raw_examples.keys())
    logger.info(
        "START | folder=%s | classes=%s | total=%d | mode=%s | dry_run=%s | threshold=%.3f | margin=%.3f",
        sort_folder.name, classes, len(images), args.mode, args.dry_run, args.threshold, args.margin,
    )

    logger.info("Building reference signatures...")
    references = build_references(raw_examples)

    t0 = time.monotonic()
    counts: dict[str, int] = defaultdict(int)
    scores_by_class: dict[str, list[float]] = defaultdict(list)

    for img_path in images:
        try:
            sig = extract_signature(img_path)
            decision, best_cls, best_score, runner_cls, runner_score = classify(
                sig, references, args.threshold, args.margin
            )
            scores_by_class[best_cls].append(best_score)

            if decision is not None:
                counts["matched"] += 1
                logger.info(
                    "MATCH | %s | %s | score=%.4f | runner_up=%s@%.4f",
                    img_path.name, decision, best_score, runner_cls, runner_score,
                )
                if not args.dry_run:
                    dest_dir = sort_folder / decision
                    dest_dir.mkdir(exist_ok=True)
                    dest = _safe_dest(dest_dir / img_path.name)
                    if args.mode == "move":
                        shutil.move(str(img_path), str(dest))
                    else:
                        shutil.copy2(str(img_path), str(dest))
            else:
                counts["unknown"] += 1
                reason = "above_threshold" if best_score > args.threshold else "ambiguous"
                logger.info(
                    "UNKNOWN | %s | best=%s@%.4f | runner_up=%s@%.4f | reason=%s",
                    img_path.name, best_cls, best_score, runner_cls, runner_score, reason,
                )
                if not args.dry_run:
                    unk_dir = sort_folder / "unknown"
                    unk_dir.mkdir(exist_ok=True)
                    dest = _safe_dest(unk_dir / img_path.name)
                    if args.mode == "move":
                        shutil.move(str(img_path), str(dest))
                    else:
                        shutil.copy2(str(img_path), str(dest))

        except Exception as e:
            counts["errors"] += 1
            logger.error("ERROR | %s | %s", img_path.name, e)

    duration = time.monotonic() - t0

    if args.dry_run:
        logger.info("--- DRY-RUN SCORE DISTRIBUTION ---")
        for cls, scores in sorted(scores_by_class.items()):
            arr = np.array(scores)
            logger.info(
                "  %s: n=%d | min=%.4f | p25=%.4f | mean=%.4f | p75=%.4f | max=%.4f",
                cls, len(arr), arr.min(),
                np.percentile(arr, 25), arr.mean(),
                np.percentile(arr, 75), arr.max(),
            )

    logger.info(
        "END | matched=%d | unknown=%d | errors=%d | duration=%.1fs",
        counts["matched"], counts["unknown"], counts["errors"], duration,
    )


if __name__ == "__main__":
    main()
