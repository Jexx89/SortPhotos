"""
SortPhotos — Streamlit GUI
Lance avec : .venv/Scripts/streamlit run app.py
"""

import shutil
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

st.set_page_config(page_title="SortPhotos", page_icon="📂", layout="wide")

st.title("SortPhotos")
st.caption("Tri automatique de photos par similarité visuelle")

# ── Sidebar — Configuration ──────────────────────────────────────────────────

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

    # session_state maintient les classes entre les re-runs de Streamlit
    if "classes" not in st.session_state:
        st.session_state.classes = [{"name": "", "files": None}]

    def add_class():
        st.session_state.classes.append({"name": "", "files": None})

    def remove_class(idx):
        st.session_state.classes.pop(idx)

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

    st.button("+ Ajouter une classe", on_click=add_class, width="stretch")

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
    run_clicked = st.button("Lancer le tri", type="primary", width="stretch")

ok, err_msg = _validate()
if run_clicked and not ok:
    st.error(err_msg)
    run_clicked = False
elif not ok:
    with col_info:
        st.info(err_msg)


# ── Traitement principal ─────────────────────────────────────────────────────

if run_clicked and ok:
    folder = Path(sort_folder_input.strip())

    # Capture la liste AVANT tout déplacement — nécessaire car les sous-dossiers
    # créés pendant le tri apparaîtraient sinon dans iterdir() lors d'un re-run
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        st.warning("Aucune image trouvée dans ce dossier.")
        st.stop()

    valid_classes = [c for c in st.session_state.classes if c["name"].strip() and c["files"]]

    # Sauvegarde les fichiers uploadés dans un dossier temp pour Pillow.
    # Les UploadedFile Streamlit ne sont pas des paths disque — on matérialise.
    with st.spinner("Construction des signatures de référence..."):
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
        references = build_references(raw_examples)

    # Aperçu des images de référence
    st.subheader("Références")
    ref_cols = st.columns(len(valid_classes))
    for col, cls in zip(ref_cols, valid_classes):
        with col:
            st.write(f"**{cls['name']}**")
            for uf in cls["files"][:3]:
                uf.seek(0)
                # width=None → Streamlit choisit une taille adaptée au conteneur
                st.image(Image.open(uf), width="stretch")

    st.divider()
    st.subheader("Traitement" + (" (simulation)" if dry_run else ""))

    progress = st.progress(0, text="Initialisation...")
    log_placeholder = st.empty()

    results: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    scores_by_class: dict[str, list[float]] = defaultdict(list)
    log_lines: list[str] = []
    t0 = time.monotonic()

    for idx, img_path in enumerate(images, 1):
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
                    # copy2 préserve les métadonnées (EXIF, dates) contrairement à copy
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
            # Capture le traceback complet pour aider au debug
            tb = traceback.format_exc()
            counts["errors"] += 1
            log_lines.append(f"ERROR | {img_path.name} | {tb.splitlines()[-1]}")
            results.append({
                "Fichier": img_path.name, "Classe": "ERROR",
                "Score": None, "Runner-up": "", "Statut": "ERROR",
            })
            # Affiche le traceback complet dans le terminal pour debug
            print(tb)

        # Mise à jour UI toutes les 10 images pour réduire la charge WebSocket
        if idx % 10 == 0 or idx == len(images):
            progress.progress(idx / len(images), text=f"{idx}/{len(images)} — {img_path.name}")
            log_placeholder.text("\n".join(log_lines[-12:]))

    duration = time.monotonic() - t0
    progress.empty()
    log_placeholder.empty()

    # ── Résultats ─────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("Résultats")

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

    # Distribution des scores par classe — aide à calibrer threshold et margin
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

    if counts["errors"]:
        st.warning(f"{counts['errors']} erreur(s) — consulte le log complet ou le terminal.")

    st.success(
        f"{'Simulation terminée' if dry_run else 'Tri terminé'} — "
        f"{counts['matched']} classées · {counts['unknown']} unknown · "
        f"{counts['errors']} erreurs | log → {log_file.name}"
    )
