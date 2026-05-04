"""
SortPhotos — GUI natif Tkinter
Lance avec : python gui_tk.py

Avantage sur Streamlit : s'exécute dans un thread séparé — la barre de
progression se met à jour en temps réel sans risque de RerunException.
"""

import logging
import queue
import shutil
import threading
import time
from pathlib import Path
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox, scrolledtext

from classifier import build_references, classify, extract_signature

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _safe_dest(dest: Path) -> Path:
    """Retourne dest intact s'il n'existe pas, sinon dest_001, dest_002…"""
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 1
    while True:
        candidate = dest.with_name(f"{stem}_{i:03d}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


# ── Widget ligne de classe ────────────────────────────────────────────────────

class ClassRow(tk.Frame):
    """Une ligne dans le panneau de gestion des classes."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._name_var = tk.StringVar()
        self._paths: list[str] = []

        tk.Label(self, text="Classe :", width=7, anchor="e").pack(side=tk.LEFT, padx=(0, 2))
        self._name_entry = tk.Entry(self, textvariable=self._name_var, width=14)
        self._name_entry.pack(side=tk.LEFT, padx=(0, 6))

        self._paths_label = tk.Label(self, text="(aucun exemple)", fg="gray",
                                     width=42, anchor="w")
        self._paths_label.pack(side=tk.LEFT, padx=(0, 4))

        tk.Button(self, text="Choisir…", command=self._pick_images).pack(side=tk.LEFT, padx=(0, 4))
        self._del_btn = tk.Button(self, text="✕", fg="red")
        self._del_btn.pack(side=tk.LEFT)

    def _pick_images(self):
        files = filedialog.askopenfilenames(
            title=f"Exemple(s) pour « {self._name_var.get() or 'cette classe'} »",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp")],
        )
        if files:
            self._paths = list(files)
            label = ", ".join(Path(f).name for f in self._paths)
            if len(label) > 50:
                label = label[:47] + "…"
            self._paths_label.config(text=label, fg="black")

    @property
    def name(self) -> str:
        return self._name_var.get().strip()

    @property
    def paths(self) -> list[str]:
        return self._paths

    def set_delete_command(self, cmd):
        self._del_btn.config(command=cmd)


# ── Application principale ────────────────────────────────────────────────────

class SortPhotosApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("SortPhotos")
        self.resizable(True, True)
        self.minsize(720, 560)

        self._class_rows: list[ClassRow] = []
        self._worker: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue()

        self._build_ui()

    # ── Construction UI ───────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        # Dossier à trier
        folder_frame = ttk.LabelFrame(self, text="Dossier à trier")
        folder_frame.pack(fill=tk.X, **pad)

        self._folder_var = tk.StringVar()
        tk.Entry(folder_frame, textvariable=self._folder_var, width=64).pack(
            side=tk.LEFT, padx=6, pady=5)
        tk.Button(folder_frame, text="Parcourir…", command=self._pick_folder).pack(
            side=tk.LEFT, padx=4)

        # Classes
        classes_frame = ttk.LabelFrame(self, text="Classes (nom + exemple(s))")
        classes_frame.pack(fill=tk.X, **pad)

        self._classes_inner = tk.Frame(classes_frame)
        self._classes_inner.pack(fill=tk.X, padx=6, pady=(4, 0))

        tk.Button(classes_frame, text="+ Ajouter une classe",
                  command=self._add_class_row).pack(anchor="w", padx=6, pady=(2, 6))

        self._add_class_row()
        self._add_class_row()

        # Options
        opts_frame = ttk.LabelFrame(self, text="Options")
        opts_frame.pack(fill=tk.X, **pad)

        row1 = tk.Frame(opts_frame)
        row1.pack(fill=tk.X, padx=6, pady=4)
        tk.Label(row1, text="Mode :").pack(side=tk.LEFT, padx=(0, 4))
        self._mode_var = tk.StringVar(value="move")
        ttk.Radiobutton(row1, text="Déplacer", variable=self._mode_var, value="move").pack(side=tk.LEFT)
        ttk.Radiobutton(row1, text="Copier",   variable=self._mode_var, value="copy").pack(side=tk.LEFT, padx=(4, 20))
        self._dry_run_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="Simulation (dry-run)", variable=self._dry_run_var).pack(side=tk.LEFT)

        row2 = tk.Frame(opts_frame)
        row2.pack(fill=tk.X, padx=6, pady=(0, 6))
        tk.Label(row2, text="Threshold :").pack(side=tk.LEFT)
        self._threshold_var = tk.StringVar(value="0.30")
        tk.Entry(row2, textvariable=self._threshold_var, width=6).pack(side=tk.LEFT, padx=(4, 16))
        tk.Label(row2, text="Margin :").pack(side=tk.LEFT)
        self._margin_var = tk.StringVar(value="0.05")
        tk.Entry(row2, textvariable=self._margin_var, width=6).pack(side=tk.LEFT, padx=4)

        # Bouton lancer
        self._run_btn = tk.Button(self, text="▶  Lancer le tri",
                                   font=("", 11, "bold"), bg="#1a6e3c", fg="white",
                                   activebackground="#145530", activeforeground="white",
                                   padx=16, pady=6, command=self._start_sort)
        self._run_btn.pack(pady=(8, 4))

        # Barre de progression
        prog_frame = tk.Frame(self)
        prog_frame.pack(fill=tk.X, padx=10, pady=(0, 4))

        self._progress_var = tk.IntVar(value=0)
        self._progress_bar = ttk.Progressbar(prog_frame, variable=self._progress_var, maximum=100)
        self._progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._progress_label = tk.Label(prog_frame, text="", width=12, anchor="w")
        self._progress_label.pack(side=tk.LEFT, padx=6)

        # Journal
        log_frame = ttk.LabelFrame(self, text="Journal")
        log_frame.pack(fill=tk.BOTH, expand=True, **pad)

        self._log_text = scrolledtext.ScrolledText(
            log_frame, height=10, font=("Courier New", 8),
            state=tk.DISABLED, bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white",
        )
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Résumé bas de fenêtre
        self._summary_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._summary_var, anchor="w",
                 font=("", 9, "bold")).pack(fill=tk.X, padx=10, pady=(0, 6))

    # ── Gestion des classes ───────────────────────────────────────────────────

    def _add_class_row(self):
        row = ClassRow(self._classes_inner)
        row.pack(fill=tk.X, pady=2)
        self._class_rows.append(row)
        self._rebind_deletes()

    def _delete_class_row(self, idx: int):
        if idx < len(self._class_rows):
            self._class_rows[idx].destroy()
            self._class_rows.pop(idx)
            self._rebind_deletes()

    def _rebind_deletes(self):
        for i, row in enumerate(self._class_rows):
            row.set_delete_command(lambda i=i: self._delete_class_row(i))

    # ── Helpers UI ────────────────────────────────────────────────────────────

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Sélectionner le dossier à trier")
        if folder:
            self._folder_var.set(folder)

    def _append_log(self, msg: str):
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> tuple[bool, str]:
        folder = self._folder_var.get().strip()
        if not folder or not Path(folder).is_dir():
            return False, "Dossier invalide ou introuvable."

        classes = [(r.name, r.paths) for r in self._class_rows if r.name or r.paths]
        if not classes:
            return False, "Aucune classe définie."
        for name, paths in classes:
            if not name:
                return False, "Une classe n'a pas de nom."
            if not paths:
                return False, f"La classe « {name} » n'a pas d'exemple image."
            for p in paths:
                if not Path(p).exists():
                    return False, f"Fichier introuvable : {p}"
        try:
            float(self._threshold_var.get())
            float(self._margin_var.get())
        except ValueError:
            return False, "Threshold et Margin doivent être des nombres décimaux."

        return True, ""

    # ── Lancement du tri ──────────────────────────────────────────────────────

    def _start_sort(self):
        ok, msg = self._validate()
        if not ok:
            messagebox.showerror("Paramètres invalides", msg)
            return

        if self._worker and self._worker.is_alive():
            messagebox.showwarning("En cours", "Un tri est déjà en cours.")
            return

        # Réinitialiser l'UI
        self._run_btn.config(state=tk.DISABLED)
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)
        self._summary_var.set("")
        self._progress_var.set(0)
        self._progress_label.config(text="")

        folder   = Path(self._folder_var.get().strip())
        examples = {r.name: r.paths for r in self._class_rows if r.name and r.paths}
        mode     = self._mode_var.get()
        dry_run  = self._dry_run_var.get()
        threshold = float(self._threshold_var.get())
        margin    = float(self._margin_var.get())

        self._worker = threading.Thread(
            target=self._sort_worker,
            args=(folder, examples, mode, dry_run, threshold, margin),
            daemon=True,
        )
        self._worker.start()
        self.after(50, self._poll_queue)

    # ── Worker (thread séparé) ────────────────────────────────────────────────

    def _sort_worker(self, folder, examples, mode, dry_run, threshold, margin):
        q = self._queue

        def emit(kind, **data):
            q.put({"kind": kind, **data})

        log_file = folder / "sort_photos.log"
        logger = logging.getLogger(f"sortphotos.tk.{id(self)}")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s",
                                              datefmt="%Y-%m-%d %H:%M:%S"))
            logger.addHandler(fh)

        try:
            images = sorted(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            total = len(images)
            if total == 0:
                emit("log", msg="Aucune image trouvée dans ce dossier.")
                emit("done", matched=0, unknown=0, errors=0, duration=0.0)
                return

            emit("total", value=total)
            emit("log", msg=f"START | {total} images | mode={mode} | dry_run={dry_run} | "
                             f"threshold={threshold} | margin={margin}")
            logger.info("START | folder=%s | total=%d | mode=%s | dry_run=%s",
                        folder.name, total, mode, dry_run)

            references = build_references(examples)
            emit("log", msg=f"Références OK — {len(references)} classe(s) : {list(references)}")

            t0 = time.monotonic()
            matched = unknown = errors = 0

            for idx, img_path in enumerate(images, 1):
                try:
                    sig = extract_signature(img_path)
                    decision, best_cls, best_score, runner_cls, runner_score = classify(
                        sig, references, threshold, margin
                    )

                    if decision is not None:
                        matched += 1
                        tag = "MATCH"
                        detail = (f"{decision} | score={best_score:.4f} | "
                                  f"runner={runner_cls}@{runner_score:.4f}")
                        if not dry_run:
                            dest_dir = folder / decision
                            dest_dir.mkdir(exist_ok=True)
                            dest = _safe_dest(dest_dir / img_path.name)
                            if mode == "move":
                                shutil.move(str(img_path), str(dest))
                            else:
                                shutil.copy2(str(img_path), str(dest))
                    else:
                        unknown += 1
                        tag = "UNKNOWN"
                        reason = "above_threshold" if best_score > threshold else "ambiguous"
                        detail = f"best={best_cls}@{best_score:.4f} | {reason}"
                        if not dry_run:
                            unk_dir = folder / "unknown"
                            unk_dir.mkdir(exist_ok=True)
                            dest = _safe_dest(unk_dir / img_path.name)
                            if mode == "move":
                                shutil.move(str(img_path), str(dest))
                            else:
                                shutil.copy2(str(img_path), str(dest))

                    line = f"{tag} | {img_path.name} | {detail}"
                    emit("log", msg=line)
                    logger.info(line)

                except Exception as exc:
                    errors += 1
                    line = f"ERROR | {img_path.name} | {exc}"
                    emit("log", msg=line)
                    logger.error(line)

                emit("progress", idx=idx, total=total)

            duration = time.monotonic() - t0
            summary = (f"END | matched={matched} | unknown={unknown} | "
                       f"errors={errors} | duration={duration:.1f}s")
            logger.info(summary)
            emit("log", msg=summary)
            emit("done", matched=matched, unknown=unknown, errors=errors, duration=duration)

        except Exception as exc:
            emit("log", msg=f"FATAL : {exc}")
            emit("done", matched=0, unknown=0, errors=0, duration=0.0)

    # ── Polling de la queue (thread principal) ────────────────────────────────

    def _poll_queue(self):
        done = False
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg["kind"]

                if kind == "log":
                    self._append_log(msg["msg"])

                elif kind == "total":
                    self._progress_bar.config(maximum=msg["value"])

                elif kind == "progress":
                    idx, total = msg["idx"], msg["total"]
                    self._progress_var.set(idx)
                    self._progress_label.config(text=f"{idx} / {total}")

                elif kind == "done":
                    m, u, e, d = msg["matched"], msg["unknown"], msg["errors"], msg["duration"]
                    label = f"Classées : {m}   Unknown : {u}   Erreurs : {e}   Durée : {d:.1f} s"
                    self._summary_var.set(label)
                    self._run_btn.config(state=tk.NORMAL)
                    done = True

        except queue.Empty:
            pass

        if not done:
            self.after(50, self._poll_queue)


# ── Entrée ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = SortPhotosApp()
    app.mainloop()
