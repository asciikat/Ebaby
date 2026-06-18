"""Per-shot editor: drag the 4 crop corners (zoom magnifier), rotate, set face,
pick cutout engine + re-run matte, edit title/region, re-scan/hand-type barcode.

Ported and extended from V1's CornerDialog. Returns an edit dict the grid applies.
"""
import math

import numpy as np

from ..models import Face

try:
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
except Exception:   # headless test import
    tk = None


def image_to_canvas(corners, scale):
    return [(int(x * scale), int(y * scale)) for x, y in corners]


def canvas_to_image(corners, scale):
    return [(int(round(x / scale)), int(round(y / scale))) for x, y in corners]


# --- The dialog below is exercised by the manual smoke test. ---

# "exact" leads: a manual quad is cut precisely (perspective-warp, no matte),
# which is what hand-placed corners should do. AI mattes follow for cases where
# the user would rather let a model find the edge.
ENGINES = ["exact", "rembg", "sam", "grabcut", "geometric"]


class ShotEditor:
    HANDLE_R = 12
    GRAB = 36
    MAG_SRC = 70
    MAG_DISP = 200

    def __init__(self, root, pil_image, shot, settings):
        self.root = root
        self.original = pil_image.convert("RGB")
        self.shot = shot           # ShotResult (mutated on apply)
        self.settings = settings
        self.result = None         # dict of edits or None if cancelled
        self.extra_rotation = 0

        self.top = tk.Toplevel(root)
        self.top.title(f"Edit — {shot.input_path}")
        self.top.transient(root)
        self.top.grab_set()

        bar = ttk.Frame(self.top); bar.pack(side="bottom", fill="x", padx=10, pady=8)
        ttk.Button(bar, text="✓ Apply", command=self._apply).pack(side="left")
        ttk.Button(bar, text="Cancel", command=self._cancel).pack(side="left", padx=8)
        ttk.Button(bar, text="Reset corners",
                   command=self._reset_corners).pack(side="left", padx=8)

        opts = ttk.Frame(self.top); opts.pack(side="bottom", fill="x", padx=10, pady=4)
        ttk.Label(opts, text="Face:").grid(row=0, column=0, sticky="w")
        self.face_var = tk.StringVar(value=shot.face.value)
        for i, f in enumerate(Face):
            ttk.Radiobutton(opts, text=f.value, value=f.value,
                            variable=self.face_var).grid(row=0, column=1 + i, padx=4)
        ttk.Label(opts, text="Engine:").grid(row=1, column=0, sticky="w")
        # Default manual edits to the exact crop: hand-placed corners should cut
        # exactly what was selected, not be re-matted by an AI that clips clear cases.
        self.engine_var = tk.StringVar(value="exact")
        ttk.Combobox(opts, textvariable=self.engine_var, values=ENGINES,
                     state="readonly", width=12).grid(row=1, column=1, columnspan=2,
                                                       sticky="w", padx=4)
        ttk.Label(opts, text="Title:").grid(row=2, column=0, sticky="w")
        self.title_var = tk.StringVar(value=shot.title or shot.ocr_title or "")
        ttk.Entry(opts, textvariable=self.title_var, width=40).grid(
            row=2, column=1, columnspan=4, sticky="w", padx=4)

        rot = ttk.Frame(self.top); rot.pack(side="bottom", pady=4)
        ttk.Label(rot, text="Rotate:").pack(side="left")
        ttk.Button(rot, text="Left", command=lambda: self._rotate(-90)).pack(side="left", padx=3)
        ttk.Button(rot, text="Right", command=lambda: self._rotate(90)).pack(side="left", padx=3)
        ttk.Button(rot, text="180", command=lambda: self._rotate(180)).pack(side="left", padx=3)

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.max_w, self.max_h = min(1100, sw - 120), min(620, sh - 360)
        self.canvas = tk.Canvas(self.top, bg="#202020", highlightthickness=0)
        self.canvas.pack(padx=10, pady=6)
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.dragging = None
        self._mag_tk = None
        self._rebuild()

    def _rebuild(self):
        pil = self.original if self.extra_rotation % 360 == 0 else \
            self.original.rotate(-self.extra_rotation, expand=True)
        w, h = pil.size
        self.scale = min(self.max_w / w, self.max_h / h, 1.0)
        dw, dh = max(100, int(w * self.scale)), max(100, int(h * self.scale))
        self._tk = ImageTk.PhotoImage(pil.resize((dw, dh), LANCZOS))
        self._disp = pil
        self.canvas.config(width=dw, height=dh)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk)
        self._reset_corners(redraw=False)
        self._redraw()

    def _reset_corners(self, redraw=True):
        w = int(self.canvas["width"]); h = int(self.canvas["height"])
        m = max(30, int(min(w, h) * 0.10))
        self.corners = [(m, m), (w - m, m), (w - m, h - m), (m, h - m)]
        if redraw:
            self._redraw()

    def _redraw(self):
        self.canvas.delete("quad")
        pts = self.corners + [self.corners[0]]
        for i in range(4):
            self.canvas.create_line(*pts[i], *pts[i + 1], fill="#ff3030",
                                    width=2, tags="quad")
        for x, y in self.corners:
            r = self.HANDLE_R
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="#ff3030",
                                    outline="white", width=2, tags="quad")

    def _closest(self, x, y):
        d = [math.hypot(x - cx, y - cy) for cx, cy in self.corners]
        i = min(range(4), key=lambda k: d[k])
        return i, d[i]

    def _press(self, e):
        i, dist = self._closest(e.x, e.y)
        self.dragging = i
        if dist >= self.GRAB:
            self.corners[i] = (e.x, e.y); self._redraw()
        self._magnify(e.x, e.y)

    def _drag(self, e):
        if self.dragging is None:
            return
        w = self.canvas.winfo_width(); h = self.canvas.winfo_height()
        self.corners[self.dragging] = (max(0, min(e.x, w - 1)), max(0, min(e.y, h - 1)))
        self._redraw(); self._magnify(*self.corners[self.dragging])

    def _release(self, _e):
        self.dragging = None
        self.canvas.delete("magnifier"); self._mag_tk = None

    def _magnify(self, x, y):
        try:
            pw, ph = self._disp.size
            ox, oy = x / max(self.scale, 1e-6), y / max(self.scale, 1e-6)
            half = self.MAG_SRC // 2
            crop = self._disp.crop((max(0, int(ox - half)), max(0, int(oy - half)),
                                    min(pw, int(ox + half)), min(ph, int(oy + half))))
            if crop.width < 8 or crop.height < 8:
                return
            region = crop.resize((self.MAG_DISP, self.MAG_DISP), Image.NEAREST)
            self._mag_tk = ImageTk.PhotoImage(region)
            self.canvas.delete("magnifier")
            self.canvas.create_image(x + 20, y - self.MAG_DISP - 20, anchor="nw",
                                    image=self._mag_tk, tags="magnifier")
            self.canvas.tag_raise("magnifier")
        except Exception:
            pass

    def _rotate(self, delta):
        self.extra_rotation = (self.extra_rotation + delta) % 360
        self._rebuild()

    def _apply(self):
        quad_img = canvas_to_image(self.corners, self.scale)
        self.result = {
            "quad": np.array(quad_img, dtype=np.float32),
            "face": Face(self.face_var.get()),
            "engine": self.engine_var.get(),
            "title": self.title_var.get().strip(),
            "extra_rotation": self.extra_rotation,
        }
        self.top.destroy()

    def _cancel(self):
        self.result = None
        self.top.destroy()

    def get(self):
        self.root.wait_window(self.top)
        return self.result
