"""Results grid grouped by DVD: thumbnails, status badges, click-to-edit,
step-through. Re-runs a shot through the pipeline when the editor applies edits.
"""
from pathlib import Path

from ..models import Face

try:
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
except Exception:
    tk = None


def badge_for(group) -> str:
    has_title = bool(group.title) and not group.title.startswith("Untitled DVD")
    if not has_title:
        return "⚠ needs review"
    return "✓ title + barcode" if group.barcode else "✓ title"


THUMB = 180


class ResultsGrid:
    def __init__(self, root, groups, settings, reprocess_cb):
        """reprocess_cb(shot, edits) -> (PIL, ShotResult); applied on editor save."""
        self.root = root
        self.groups = groups
        self.settings = settings
        self.reprocess_cb = reprocess_cb
        self._thumbs = []

        self.top = tk.Toplevel(root)
        self.top.title("Review results")
        self.top.geometry("1000x720")

        top_bar = ttk.Frame(self.top); top_bar.pack(fill="x", padx=10, pady=6)
        ttk.Button(top_bar, text="Step through all",
                   command=self._step_through).pack(side="left")
        ttk.Label(top_bar, text="Click any photo to fix it.").pack(side="left", padx=12)

        canvas = tk.Canvas(self.top, highlightthickness=0)
        scroll = ttk.Scrollbar(self.top, orient="vertical", command=canvas.yview)
        self.frame = ttk.Frame(canvas)
        self.frame.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self._render()

    def _render(self):
        for child in self.frame.winfo_children():
            child.destroy()
        self._thumbs.clear()
        for g in self.groups:
            box = ttk.LabelFrame(self.frame, text=f"DVD {g.index}: {g.title}  [{badge_for(g)}]")
            box.pack(fill="x", padx=8, pady=6)
            row = ttk.Frame(box); row.pack(fill="x", padx=6, pady=6)
            for shot in g.shots:
                cell = ttk.Frame(row); cell.pack(side="left", padx=8)
                img = self._load_thumb(shot.output_path)
                if img is not None:
                    self._thumbs.append(img)
                    btn = tk.Button(cell, image=img,
                                    command=lambda s=shot: self._edit(s))
                    btn.pack()
                ttk.Label(cell, text=shot.face.value).pack()

    def _load_thumb(self, path):
        if not path or not Path(path).exists():
            return None
        pil = Image.open(path).convert("RGB")
        pil.thumbnail((THUMB, THUMB), LANCZOS)
        return ImageTk.PhotoImage(pil)

    def _edit(self, shot):
        from .editor import ShotEditor
        pil = Image.open(shot.input_path).convert("RGB") \
            if Path(shot.input_path).exists() else Image.new("RGB", (600, 400), "white")
        editor = ShotEditor(self.root, pil, shot, self.settings)
        edits = editor.get()
        if edits:
            self.reprocess_cb(shot, edits)
            self._render()

    def _step_through(self):
        for g in self.groups:
            for shot in g.shots:
                self._edit(shot)
