import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional

THUMB_SIZE = (128, 128)
GRAY_SIZE = (16, 16)
GRAD_SIZE = 64
GRAD_GRID = 8
HIST_BINS = 16

WEIGHT_COLOR = 5.0
WEIGHT_STRUCT = 1.0
WEIGHT_TEXTURE = 2.0


def extract_signature(image_path: Path | str) -> np.ndarray:
    img = Image.open(str(image_path)).convert("RGB")

    # Feature 1: RGB color histogram (16 bins × 3 channels = 48 values)
    arr = np.array(img.resize(THUMB_SIZE, Image.LANCZOS), dtype=np.float32)
    hist = np.concatenate([
        np.histogram(arr[:, :, c], bins=HIST_BINS, range=(0, 256))[0]
        for c in range(3)
    ]).astype(np.float32)
    hist /= hist.sum() + 1e-8

    # Feature 2: grayscale structure thumbnail (16×16 = 256 values)
    gray = np.array(
        img.resize(GRAY_SIZE, Image.LANCZOS).convert("L"), dtype=np.float32
    ).flatten() / 255.0

    # Feature 3: gradient texture — block mean+std on 8×8 grid (128 values)
    g = np.array(
        img.resize((GRAD_SIZE, GRAD_SIZE), Image.LANCZOS).convert("L"), dtype=np.float32
    )
    gx = g[:, 1:] - g[:, :-1]   # (64, 63)
    gy = g[1:, :] - g[:-1, :]   # (63, 64)
    grad = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)  # (63, 63)
    n = grad.shape[0]
    bh = bw = n // GRAD_GRID
    texture = []
    for i in range(GRAD_GRID):
        for j in range(GRAD_GRID):
            block = grad[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            if block.size == 0:
                texture.extend([0.0, 0.0])
                continue
            texture.extend([block.mean(), block.std()])
    texture = np.array(texture, dtype=np.float32)
    texture /= texture.max() + 1e-8

    return np.concatenate([
        hist * WEIGHT_COLOR,
        gray * WEIGHT_STRUCT,
        texture * WEIGHT_TEXTURE,
    ])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def build_references(examples: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """Compute mean signature per class from one or more example images."""
    refs = {}
    for cls, paths in examples.items():
        sigs = [extract_signature(p) for p in paths]
        refs[cls] = np.mean(sigs, axis=0)
    return refs


def classify(
    signature: np.ndarray,
    references: Dict[str, np.ndarray],
    threshold: float = 0.30,
    ambiguity_margin: float = 0.05,
) -> Tuple[Optional[str], str, float, str, float]:
    """
    Returns (decision, best_class, best_score, runner_up_class, runner_up_score).
    decision is None when photo goes to unknown.
    """
    scores = {cls: cosine_distance(signature, ref) for cls, ref in references.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1])

    best_cls, best_score = ranked[0]
    runner_cls, runner_score = ranked[1] if len(ranked) > 1 else ("N/A", 1.0)

    if best_score > threshold:
        return None, best_cls, best_score, runner_cls, runner_score

    if len(ranked) > 1 and (runner_score - best_score) < ambiguity_margin:
        return None, best_cls, best_score, runner_cls, runner_score

    return best_cls, best_cls, best_score, runner_cls, runner_score
