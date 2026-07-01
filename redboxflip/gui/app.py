"""Main Tkinter window: folders, settings, Run, progress, then the review grid.

All three flows live here: fire-and-forget (Run, auto-open output), review grid
(opens after a run), and one-by-one (Step through in the grid).
"""
import copy
import os
import queue
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

from .. import pipeline
from ..config import load_settings, save_settings
from ..models import Settings

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    tk = None

ENGINES = ["rembg", "sam", "grabcut", "geometric"]


def settings_for_edit(settings: Settings, edits: dict) -> Settings:
    """Return a copy of settings with per-edit overrides (engine) applied."""
    over = replace(settings)
    if edits.get("engine"):
        over.cutout_engine = edits["engine"]
    return over


def _open_folder(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)             # noqa: P204
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            # WSL: open in Windows Explorer if available, else xdg-open
            if subprocess.run(["which", "explorer.exe"], capture_output=True).returncode == 0:
                subprocess.Popen(["explorer.exe", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


class App:
    def __init__(self, root):
        self.root = root
        self.settings = load_settings()
        self.msg_q = queue.Queue()
        self.run_dir = None
        self.groups = None

        root.title("Red-Box DVD Flip")
        root.geometry("760x560")

        # folders
        ff = ttk.LabelFrame(root, text="Folders"); ff.pack(fill="x", padx=10, pady=8)
        self.in_var = tk.StringVar(value=self.settings.input_dir)
        self.out_var = tk.StringVar(value=self.settings.output_dir)
        self._folder_row(ff, "Input", self.in_var, 0)
        self._folder_row(ff, "Output", self.out_var, 1)
        ff.columnconfigure(1, weight=1)

        # settings
        sf = ttk.LabelFrame(root, text="Settings"); sf.pack(fill="x", padx=10, pady=8)
        self.engine_var = tk.StringVar(value=self.settings.cutout_engine)
        ttk.Label(sf, text="Cutout engine").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(sf, textvariable=self.engine_var, values=ENGINES,
                     state="readonly", width=14).grid(row=0, column=1, sticky="w")
        self.region_var = tk.StringVar(value=self.settings.default_region)
        ttk.Label(sf, text="Region").grid(row=0, column=2, sticky="w", padx=6)
        ttk.Entry(sf, textvariable=self.region_var, width=24).grid(row=0, column=3, sticky="w")
        self.margin_var = tk.IntVar(value=int(self.settings.margin_pct))
        ttk.Label(sf, text="Margin %").grid(row=1, column=0, sticky="w", padx=6)
        ttk.Spinbox(sf, from_=0, to=30, textvariable=self.margin_var, width=6).grid(row=1, column=1, sticky="w")
        self.quality_var = tk.IntVar(value=self.settings.jpeg_quality)
        ttk.Label(sf, text="JPEG quality").grid(row=1, column=2, sticky="w", padx=6)
        ttk.Spinbox(sf, from_=70, to=100, textvariable=self.quality_var, width=6).grid(row=1, column=3, sticky="w")
        self.colour_var = tk.BooleanVar(value=self.settings.colour_tidy)
        ttk.Checkbutton(sf, text="Colour tidy", variable=self.colour_var).grid(row=2, column=0, sticky="w", padx=6)
        self.lookup_var = tk.BooleanVar(value=self.settings.title_lookup)
        ttk.Checkbutton(sf, text="OCR title", variable=self.lookup_var).grid(row=2, column=1, sticky="w")

        # run
        rb = ttk.Frame(root); rb.pack(fill="x", padx=10, pady=8)
        self.run_btn = ttk.Button(rb, text="▶ Run", command=self._start)
        self.run_btn.pack(side="left")
        self.status = ttk.Label(rb, text="Ready"); self.status.pack(side="left", padx=12)

        pf = ttk.LabelFrame(root, text="Progress"); pf.pack(fill="both", expand=True, padx=10, pady=8)
        self.pb = ttk.Progressbar(pf, mode="determinate"); self.pb.pack(fill="x", padx=8, pady=6)
        self.log = tk.Text(pf, height=12); self.log.pack(fill="both", expand=True, padx=8, pady=6)

        root.after(80, self._pump)
        root.protocol("WM_DELETE_WINDOW", self._close)

    def _folder_row(self, parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(parent, text="Browse",
                   command=lambda: var.set(filedialog.askdirectory() or var.get())
                   ).grid(row=row, column=2, padx=6)

    def _collect_settings(self) -> Settings:
        s = self.settings
        s.input_dir = self.in_var.get()
        s.output_dir = self.out_var.get()
        s.cutout_engine = self.engine_var.get()
        s.default_region = self.region_var.get()
        s.margin_pct = float(self.margin_var.get())
        s.jpeg_quality = int(self.quality_var.get())
        s.colour_tidy = bool(self.colour_var.get())
        s.title_lookup = bool(self.lookup_var.get())
        return s

    def _start(self):
        s = self._collect_settings()
        if not s.input_dir or not s.output_dir:
            messagebox.showwarning("Folders", "Choose input and output folders.")
            return
        save_settings(s)
        self.run_btn.config(state="disabled")
        self.log.delete("1.0", "end")
        threading.Thread(target=self._worker, args=(copy.copy(s),), daemon=True).start()

    def _worker(self, s):
        try:
            def cb(done, total, name):
                self.msg_q.put(("progress", (done, total, name)))
            run_dir, groups = pipeline.run_batch(s, progress_cb=cb)
            self.msg_q.put(("done", (run_dir, groups)))
        except Exception as e:
            self.msg_q.put(("error", str(e)))

    def _pump(self):
        try:
            while True:
                kind, payload = self.msg_q.get_nowait()
                if kind == "progress":
                    done, total, name = payload
                    self.pb["maximum"] = total; self.pb["value"] = done
                    self.status.config(text=f"{done}/{total}")
                    self.log.insert("end", f"[{done}/{total}] {name}\n"); self.log.see("end")
                elif kind == "done":
                    self.run_dir, self.groups = payload
                    self.status.config(text=f"Done. {len(self.groups)} DVD(s).")
                    self.run_btn.config(state="normal")
                    self._after_run()
                elif kind == "error":
                    self.status.config(text="Error")
                    self.log.insert("end", f"ERROR: {payload}\n")
                    self.run_btn.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(80, self._pump)

    def _after_run(self):
        if self.settings.auto_open_output and self.run_dir:
            _open_folder(self.run_dir)
        if self.settings.auto_open_grid and self.groups:
            from .grid import ResultsGrid
            ResultsGrid(self.root, self.groups, self.settings, self._reprocess)

    def _reprocess(self, shot, edits):
        over = settings_for_edit(self.settings, edits)
        pil, new_res = pipeline.process_shot(
            Path(shot.input_path),
            edits.get("face", shot.face),
            over,
            manual_quad=edits.get("quad"),
            extra_rotation=edits.get("extra_rotation", 0),
        )
        # apply title/barcode overrides and re-save under the (possibly new) title
        title = edits.get("title") or shot.title or "Untitled"
        from .. import naming
        from ..imaging import save_jpeg
        out = Path(shot.output_path).parent / naming.output_filename(title, new_res.face)
        save_jpeg(pil, out, self.settings.jpeg_quality)
        shot.output_path = str(out)
        shot.face = new_res.face
        shot.title = title
        shot.cutout_method = new_res.cutout_method

    def _close(self):
        save_settings(self._collect_settings())
        self.root.destroy()


def _ensure_tcl():
    """Point Tcl/Tk at the base install so tk.Tk() works inside a Windows venv.

    A venv created on Windows doesn't copy the tcl/ runtime, so tkinter fails
    with "Can't find a usable init.tcl". CPython only auto-sets TCL_LIBRARY for
    the base interpreter; the venv launcher inherits nothing. Locate the
    tcl8.x / tk8.x folders under the base prefix and export them. No-op when the
    vars are already set or off Windows (Linux/WSL find them on the system)."""
    if os.environ.get("TCL_LIBRARY") or not sys.platform.startswith("win"):
        return
    import glob
    for base in (sys.base_prefix, sys.prefix):
        tcl = glob.glob(os.path.join(base, "tcl", "tcl8.*"))
        tkl = glob.glob(os.path.join(base, "tcl", "tk8.*"))
        if tcl:
            os.environ["TCL_LIBRARY"] = tcl[0]
            if tkl:
                os.environ["TK_LIBRARY"] = tkl[0]
            return


def launch():
    if tk is None:
        print("Tkinter not available. Install: sudo apt install python3-tk python3-pil.imagetk",
              file=sys.stderr)
        return 1
    _ensure_tcl()
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0
