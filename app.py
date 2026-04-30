"""
SortPhotos — Streamlit GUI
Lance avec : .venv/Scripts/streamlit run app.py
"""

import logging
import shutil
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from classifier import build_references, classify, extract_signature

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ── Logging applicatif ───────────────────────────────────────────────────────
# Écrit dans log/app.log — chaque run Streamlit append au même fichier.
# Permet de diagnostiquer les crashs WebSocket invisibles dans l'UI.

_LOG_DIR = Path(__file__).parent / "log"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "app.log"

_log_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_app_logger = logging.getLogger("sortphotos.app")
# Évite les handlers dupliqués lors des re-runs Streamlit (le module est rechargé mais
# les loggers Python persistent dans le process)
if not _app_logger.handlers:
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _fh.setFormatter(_log_fmt)
    _app_logger.addHandler(_fh)
    _app_logger.setLevel(logging.DEBUG)

def _log(level: str, msg: str, *args) -> None:
    getattr(_app_logger, level)(msg, *args)


def _excepthook(exc_type, exc_value, exc_tb):
    """Capture toutes les exceptions non gérées dans le log avant le crash."""
    _log("critical", "UNHANDLED EXCEPTION:\n%s",
         "".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _excepthook

_log("info", "=== Script run start ===")

# ── Page config ──────────────────────────────────────────────────────────────

try:
    st.set_page_config(page_title="SortPhotos", page_icon="📂", layout="wide")
    _log("debug", "set_page_config OK")
except Exception:
    _log("error", "set_page_config FAILED:\n%s", traceback.format_exc())
    raise

st.title("SortPhotos")
st.caption("Tri automatique de photos par similarité visuelle")

# ── Sidebar — Configuration ──────────────────────────────────────────────────

_log("debug", "Rendering sidebar...")
try:
    with st.sidebar:
        st.header("Configuration")

        sort_folder_input = st.text_input(
            "Dossier à trier",
            placeholder="C:/chemin/vers/TO_SORT",
            help="Chemin absolu du dossier contenant les photos à trier.",
        )

        st.divider()
        st.subheader("Classes exemples")
        st.caption("Ajoute une ou plusieurs images par classe.")

        # session_state maintient les classes entre les re-runs Streamlit
        if "classes" not in st.session_state:
            st.session_state.classes = [{"name": "", "files": None}]
            _log("debug", "session_state.classes initialized")

        def add_class():
            st.session_state.classes.append({"name": "", "files": None})
            _log("debug", "Class added — total: %d", len(st.session_state.classes))

        def remove_class(idx):
            st.session_state.classes.pop(idx)
            _log("debug", "Class %d removed — remaining: %d", idx, len(st.session_state.classes))

        for i, cls in enumerate(st.session_state.classes):
            col_name, col_del = st.columns([3, 1])
            with col_name:
                st.session_state.classes[i]["name"] = st.text_input(
                    f"Nom classe {i + 1}", value=cls["name"], key=f"cls_name_{i}"
                )
            with col_del:
                st.write("")
                if st.button("✕", key=f"del_{i}", help="Supprimer"):
                    remove_class(i)
                    st.rerun()
            st.session_state.classes[i]["files"] = st.file_uploader(
                f"Exemples — {st.session_state.classes[i]['name'] or f'Classe {i + 1}'}",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                accept_multiple_files=True,
                key=f"cls_files_{i}",
            )

        # use_container_width est déprécié mais reste valide pour st.button
        # (contrairement à st.image où width='stretch' est requis)
        st.button("+ Ajouter une classe", on_click=add_class, use_container_width=True)

        st.divider()
        st.subheader("Options")
        mode = st.radio("Mode", ["move", "copy"], horizontal=True,
                        help="Déplacer ou copier les fichiers.")
        threshold = st.slider("Seuil inconnu", 0.05, 0.50, 0.30, 0.01,
                              help="Distance cosine au-dessus de laquelle la photo va dans unknown/.")
        margin = st.slider("Marge ambiguïté", 0.01, 0.20, 0.05, 0.01,
                           help="Écart minimal entre les 2 meilleures classes. En dessous → unknown/.")
        dry_run = st.toggle("Dry-run (simulation)", value=False,
                            help="Analyse les scores sans déplacer ni copier de fichiers.")

    _log("debug", "Sidebar rendered OK | folder='%s' | classes=%d | dry_run=%s",
         sort_folder_input, len(st.session_state.classes), dry_run)

except Exception:
    _log("error", "Sidebar FAILED:\n%s", traceback.format_exc())
    raise


# ── Validation des paramètres ────────────────────────────────────────────────

def _validate() -> tuple[bool, str]:
    if not sort_folder_input.strip():
        return False, "Renseigne le dossier à trier."
    folder = Path(sort_folder_input.strip())
    if not folder.exists():
        return False, f"Dossier introuvable : {folder}"
    valid = [c for c in st.session_state.classes if c["name"].strip() and c["files"]]
    if len(valid) < 2:
        return False, "Il faut au moins 2 classes avec un nom et des images exemples."
    return True, ""


# ── Bouton principal ─────────────────────────────────────────────────────────

col_btn, col_info = st.columns([2, 5])
with col_btn:
    run_clicked = st.button("Lancer le tri", type="primary", use_container_width=True)

ok, err_msg = _validate()
if run_clicked and not ok:
    _log("warning", "Run blocked — validation failed: %s", err_msg)
    st.error(err_msg)
    run_clicked = False
elif not ok:
    with col_info:
        st.info(err_msg)

_log("debug", "Script run end (run_clicked=%s, ok=%s)", run_clicked, ok)

# ── Traitement principal ─────────────────────────────────────────────────────

if run_clicked and ok:
    folder = Path(sort_folder_input.strip())
    _log("info", "RUN START | folder=%s | mode=%s | dry_run=%s | threshold=%.2f | margin=%.2f",
         folder, mode, dry_run, threshold, margin)

    # Capture la liste AVANT tout déplacement — les sous-dossiers créés pendant
    # le tri apparaîtraient sinon dans iterdir() lors d'un re-run Streamlit
    try:
        images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    except Exception:
        _log("error", "iterdir FAILED:\n%s", traceback.format_exc())
        st.error(f"Impossible de lire le dossier : {folder}")
        st.stop()

    _log("info", "Found %d images in %s", len(images), folder)
    if not images:
        st.warning("Aucune image trouvée dans ce dossier.")
        st.stop()

    valid_classes = [c for c in st.session_state.classes if c["name"].strip() and c["files"]]

    # Matérialise les UploadedFile Streamlit sur disque pour Pillow
    with st.spinner("Construction des signatures de référence..."):
        try:
            tmp_dir = Path(tempfile.mkdtemp())
            raw_examples: dict[str, list[str]] = {}
            for cls in valid_classes:
                name = cls["name"].strip()
                paths = []
                for uf in cls["files"]:
                    uf.seek(0)
                    tmp_path = tmp_dir / f"{name}_{uf.name}"
                    tmp_path.write_bytes(uf.read())
                    paths.append(str(tmp_path))
                raw_examples[name] = paths
            _log("info", "Example files written to tmp — classes: %s", list(raw_examples))
            references = build_references(raw_examples)
            _log("info", "References built OK — %d classes", len(references))
        except Exception:
            _log("error", "build_references FAILED:\n%s", traceback.format_exc())
            st.error("Erreur lors de la construction des références. Voir log/app.log.")
            st.stop()

    # Aperçu des images de référence
    # On passe les bytes bruts à st.image plutôt qu'un objet PIL —
    # Image.open() est lazy et garde le UploadedFile ouvert, ce qui cause
    # un crash C-level quand Streamlit essaie de lire les pixels après le read().
    st.subheader("Références")
    ref_cols = st.columns(len(valid_classes))
    for col, cls in zip(ref_cols, valid_classes):
        with col:
            st.write(f"**{cls['name']}**")
            for uf in cls["files"][:3]:
                try:
                    uf.seek(0)
                    img_bytes = uf.read()
                    _log("debug", "Displaying thumbnail for %s (%d bytes)", cls["name"], len(img_bytes))
                    st.image(img_bytes, width="stretch")
                    _log("debug", "Thumbnail OK for %s", cls["name"])
                except Exception:
                    _log("warning", "Could not display thumbnail for %s: %s",
                         cls["name"], traceback.format_exc().splitlines()[-1])

    st.divider()
    st.subheader("Traitement" + (" (simulation)" if dry_run else ""))

    progress = st.progress(0, text="Initialisation...")
    log_placeholder = st.empty()

    results: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    scores_by_class: dict[str, list[float]] = defaultdict(list)
    log_lines: list[str] = []

    _log("info", "Processing loop START — %d images", len(images))

    # Tout le traitement dans un st.spinner — AUCUN appel Streamlit à l'intérieur.
    # Les appels progress/text en cours de loop déclenchaient des RerunException
    # qui interrompaient le script et laissaient les fichiers à moitié déplacés.
    spinner_label = f"{'Simulation' if dry_run else 'Tri'} en cours — {len(images)} images…"
    with st.spinner(spinner_label):
        t0 = time.monotonic()
        for idx, img_path in enumerate(images, 1):
            if idx == 1 or idx % 100 == 0:
                _log("debug", "Processing image %d/%d — %s", idx, len(images), img_path.name)
            try:
                sig = extract_signature(img_path)
                decision, best_cls, best_score, runner_cls, runner_score = classify(
                    sig, references, threshold, margin
                )
                scores_by_class[best_cls].append(best_score)

                if decision is not None:
                    counts["matched"] += 1
                    tag = "MATCH"
                    detail = f"{decision} | score={best_score:.4f} | runner={runner_cls}@{runner_score:.4f}"
                    if not dry_run:
                        dest_dir = folder / decision
                        dest_dir.mkdir(exist_ok=True)
                        dest = dest_dir / img_path.name
                        # copy2 préserve les métadonnées EXIF/dates
                        if mode == "move":
                            shutil.move(str(img_path), str(dest))
                        else:
                            shutil.copy2(str(img_path), str(dest))
                else:
                    counts["unknown"] += 1
                    tag = "UNKNOWN"
                    reason = "above_threshold" if best_score > threshold else "ambiguous"
                    detail = f"best={best_cls}@{best_score:.4f} | {reason}"
                    decision = "unknown"
                    if not dry_run:
                        unk_dir = folder / "unknown"
                        unk_dir.mkdir(exist_ok=True)
                        dest = unk_dir / img_path.name
                        if mode == "move":
                            shutil.move(str(img_path), str(dest))
                        else:
                            shutil.copy2(str(img_path), str(dest))

                results.append({
                    "Fichier": img_path.name,
                    "Classe": decision,
                    "Score": round(best_score, 4),
                    "Runner-up": f"{runner_cls}@{runner_score:.4f}",
                    "Statut": tag,
                })
                log_lines.append(f"{tag} | {img_path.name} | {detail}")

            except Exception:
                tb = traceback.format_exc()
                counts["errors"] += 1
                log_lines.append(f"ERROR | {img_path.name} | {tb.splitlines()[-1]}")
                results.append({
                    "Fichier": img_path.name, "Classe": "ERROR",
                    "Score": None, "Runner-up": "", "Statut": "ERROR",
                })
                _log("error", "Image processing FAILED for %s:\n%s", img_path.name, tb)

        duration = time.monotonic() - t0

    _log("info", "Processing loop END — duration=%.1fs | matched=%d | unknown=%d | errors=%d",
         duration, counts["matched"], counts["unknown"], counts["errors"])
    _log("info", "RUN END | matched=%d | unknown=%d | errors=%d | duration=%.1fs",
         counts["matched"], counts["unknown"], counts["errors"], duration)

    # ── Résultats ─────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("Résultats")
    _log("debug", "Rendering results — %d rows", len(results))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", len(images))
    m2.metric("Classées", counts["matched"],
              delta=f"{counts['matched'] / len(images) * 100:.1f}%")
    m3.metric("Unknown", counts["unknown"])
    m4.metric("Durée", f"{duration:.1f}s")

    st.dataframe(
        results,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.4f"),
            "Statut": st.column_config.TextColumn(width="small"),
        },
    )

    # Distribution des scores — aide à calibrer threshold et margin
    if scores_by_class:
        st.subheader("Distribution des scores")
        dist_cols = st.columns(len(scores_by_class))
        for col, (cls_name, scores) in zip(dist_cols, sorted(scores_by_class.items())):
            arr = np.array(scores)
            hist_counts, _ = np.histogram(arr, bins=20)
            with col:
                st.write(f"**{cls_name}** — n={len(arr)}")
                st.write(f"min={arr.min():.4f} | mean={arr.mean():.4f} | max={arr.max():.4f}")
                st.bar_chart({"score": hist_counts}, height=150)

    with st.expander("Log complet"):
        st.text("\n".join(log_lines))

    # Écriture du log fichier — append pour conserver l'historique des runs
    log_file = folder / "sort_photos.log"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                f"{ts} | START | folder={folder.name} | classes={list(raw_examples)} "
                f"| total={len(images)} | mode={mode} | dry_run={dry_run}\n"
            )
            for line in log_lines:
                f.write(f"{ts} | {line}\n")
            f.write(
                f"{ts} | END | matched={counts['matched']} | unknown={counts['unknown']} "
                f"| errors={counts['errors']} | duration={duration:.1f}s\n"
            )
    except Exception:
        _log("warning", "Could not write sort_photos.log: %s", traceback.format_exc().splitlines()[-1])

    if counts["errors"]:
        st.warning(f"{counts['errors']} erreur(s) — consulte log/app.log pour les tracebacks.")

    st.success(
        f"{'Simulation terminée' if dry_run else 'Tri terminé'} — "
        f"{counts['matched']} classées · {counts['unknown']} unknown · "
        f"{counts['errors']} erreurs | log → {log_file.name}"
    )
