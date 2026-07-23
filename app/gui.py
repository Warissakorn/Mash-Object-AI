"""Tkinter desktop GUI for the vehicle Re-ID A/B matcher.

Workflow:
    1. Pick folder A and folder B.
    2. Click "Process" — detection + embedding runs in a background thread.
    3. The left gallery lists every vehicle detected at A (thumbnail + time).
    4. Click an A vehicle — the right panel shows its best matches from B with
       similarity score and time gap.
    5. Sliders re-rank live (similarity threshold, travel-time window) without
       re-running the models.
    6. "View full frame" draws the bounding box on the source image.

Run:  python app/gui.py
"""
from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Make repo root + src importable when launched directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import config  # noqa: E402

try:
    import cv2  # noqa: E402
    from PIL import Image, ImageTk, ImageDraw  # noqa: E402
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "The GUI needs opencv-python and pillow installed:\n"
        "    pip install -r requirements.txt\n"
        f"(import error: {exc})"
    )

THUMB_SIZE = (120, 120)
BIG_THUMB_SIZE = (200, 200)


def bgr_to_thumbnail(crop, size=THUMB_SIZE) -> "ImageTk.PhotoImage":
    """Convert a BGR numpy crop to a Tk-displayable thumbnail."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img.thumbnail(size, Image.LANCZOS)
    return ImageTk.PhotoImage(img)


class ReIDApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Vehicle Re-ID — Point A / Point B Matcher")
        self.geometry("1100x760")

        # State
        self.dir_a = tk.StringVar()
        self.dir_b = tk.StringVar()
        self.records_a: list = []
        self.records_b: list = []
        self._emb_a = None
        self._emb_b = None
        self._times_a: list = []
        self._times_b: list = []
        self._pipeline = None
        self._thumb_refs: list = []  # keep PhotoImage refs alive
        self._match_thumb_refs: list = []
        self._selected_a: int | None = None

        # Tunables
        self.threshold = tk.DoubleVar(value=config.SIMILARITY_THRESHOLD)
        self.min_travel = tk.DoubleVar(value=config.MIN_TRAVEL_SECONDS)
        self.max_travel = tk.DoubleVar(value=config.MAX_TRAVEL_SECONDS)
        self.status = tk.StringVar(value="Select folders for point A and B, then Process.")

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Point A folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.dir_a, width=55).grid(row=0, column=1, padx=4)
        ttk.Button(top, text="Browse…", command=lambda: self._pick(self.dir_a)).grid(row=0, column=2)

        ttk.Label(top, text="Point B folder:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.dir_b, width=55).grid(row=1, column=1, padx=4)
        ttk.Button(top, text="Browse…", command=lambda: self._pick(self.dir_b)).grid(row=1, column=2)

        self.process_btn = ttk.Button(top, text="Process", command=self._on_process)
        self.process_btn.grid(row=0, column=3, rowspan=2, padx=10, sticky="ns")

        # Sliders
        sliders = ttk.LabelFrame(self, text="Matching controls", padding=8)
        sliders.pack(fill=tk.X, padx=8)

        self._add_slider(sliders, 0, "Similarity ≥", self.threshold, 0.0, 1.0, 0.01,
                         self._on_control_change)
        self._add_slider(sliders, 1, "Min travel (s)", self.min_travel, 0, 3600, 1,
                         self._on_control_change)
        self._add_slider(sliders, 2, "Max travel (s)", self.max_travel, 0, 7200, 1,
                         self._on_control_change)

        # Progress + status
        bar = ttk.Frame(self, padding=(8, 0))
        bar.pack(fill=tk.X)
        self.progress = ttk.Progressbar(bar, mode="determinate")
        self.progress.pack(fill=tk.X, side=tk.LEFT, expand=True)
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill=tk.X, padx=8, pady=2)

        # Main panes: A gallery | B matches
        panes = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.LabelFrame(panes, text="Vehicles at A (click one)")
        right = ttk.LabelFrame(panes, text="Best matches at B")
        panes.add(left, weight=1)
        panes.add(right, weight=1)

        self.a_canvas, self.a_inner = self._scrollable(left)
        self.b_canvas, self.b_inner = self._scrollable(right)

    def _add_slider(self, parent, row, label, var, lo, hi, step, cmd):
        ttk.Label(parent, text=label, width=14).grid(row=row, column=0, sticky="w")
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, command=lambda _=None: cmd())
        scale.grid(row=row, column=1, sticky="ew", padx=6)
        parent.columnconfigure(1, weight=1)
        val = ttk.Label(parent, width=8)
        val.grid(row=row, column=2, sticky="e")

        def _sync(*_):
            val.config(text=f"{var.get():.2f}" if step < 1 else f"{int(var.get())}")
        var.trace_add("write", _sync)
        _sync()

    def _scrollable(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0)
        vbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        # Mouse-wheel scrolling.
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        return canvas, inner

    # -------------------------------------------------------------- actions
    def _pick(self, var: tk.StringVar) -> None:
        folder = filedialog.askdirectory()
        if folder:
            var.set(folder)

    def _on_process(self) -> None:
        dir_a, dir_b = self.dir_a.get().strip(), self.dir_b.get().strip()
        if not (os.path.isdir(dir_a) and os.path.isdir(dir_b)):
            messagebox.showerror("Missing folders", "Please choose valid A and B folders.")
            return
        self.process_btn.config(state=tk.DISABLED)
        self.status.set("Loading models and processing… (first run downloads weights)")
        self.progress.config(mode="indeterminate")
        self.progress.start(12)
        threading.Thread(target=self._process_worker, args=(dir_a, dir_b), daemon=True).start()

    def _process_worker(self, dir_a: str, dir_b: str) -> None:
        try:
            from mash_reid.pipeline import Pipeline, stack_embeddings, record_timestamps
            if self._pipeline is None:
                self._pipeline = Pipeline()

            def progress(done, total, message):
                self.after(0, lambda: self.status.set(message))

            records_a = self._pipeline.process_folder(dir_a, "A", progress=progress)
            records_b = self._pipeline.process_folder(dir_b, "B", progress=progress)

            self.records_a, self.records_b = records_a, records_b
            self._emb_a = stack_embeddings(records_a)
            self._emb_b = stack_embeddings(records_b)
            self._times_a = record_timestamps(records_a)
            self._times_b = record_timestamps(records_b)
            self.after(0, self._on_process_done)
        except Exception as exc:  # surface errors on the GUI thread
            self.after(0, lambda: self._on_process_error(exc))

    def _on_process_error(self, exc: Exception) -> None:
        self.progress.stop()
        self.progress.config(mode="determinate")
        self.process_btn.config(state=tk.NORMAL)
        self.status.set(f"Error: {exc}")
        messagebox.showerror("Processing failed", str(exc))

    def _on_process_done(self) -> None:
        self.progress.stop()
        self.progress.config(mode="determinate")
        self.process_btn.config(state=tk.NORMAL)
        self.status.set(
            f"Detected {len(self.records_a)} vehicles at A, "
            f"{len(self.records_b)} at B. Click a vehicle on the left."
        )
        self._populate_a_gallery()
        # Clear match panel.
        for w in self.b_inner.winfo_children():
            w.destroy()

    def _populate_a_gallery(self) -> None:
        for w in self.a_inner.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        cols = 3
        for i, rec in enumerate(self.records_a):
            if rec.crop is None:
                continue
            thumb = bgr_to_thumbnail(rec.crop)
            self._thumb_refs.append(thumb)
            cell = ttk.Frame(self.a_inner, padding=4, relief=tk.RIDGE)
            cell.grid(row=i // cols, column=i % cols, padx=4, pady=4, sticky="n")
            btn = tk.Button(cell, image=thumb, command=lambda idx=i: self._on_select_a(idx))
            btn.pack()
            ttk.Label(cell, text=f"A[{i}] {rec.class_name}").pack()
            ttk.Label(cell, text=rec.timestamp.strftime("%H:%M:%S")).pack()

    def _on_select_a(self, idx: int) -> None:
        self._selected_a = idx
        self._refresh_matches()

    def _on_control_change(self) -> None:
        # Re-rank live if a vehicle is selected (no model re-run).
        if self._selected_a is not None:
            self._refresh_matches()

    def _refresh_matches(self) -> None:
        for w in self.b_inner.winfo_children():
            w.destroy()
        self._match_thumb_refs.clear()

        if self._selected_a is None or self._emb_a is None or not len(self.records_b):
            return

        from mash_reid import matcher

        result = matcher.match(
            self._emb_a,
            self._emb_b,
            self._times_a,
            self._times_b,
            similarity_threshold=float(self.threshold.get()),
            min_travel_seconds=float(self.min_travel.get()),
            max_travel_seconds=float(self.max_travel.get()),
            top_k=config.TOP_K,
        )
        cands = result.get(self._selected_a, [])

        ra = self.records_a[self._selected_a]
        header = ttk.Label(
            self.b_inner,
            text=f"Query A[{self._selected_a}] {ra.class_name} @ "
                 f"{ra.timestamp.strftime('%H:%M:%S')} — {len(cands)} match(es)",
        )
        header.grid(row=0, column=0, columnspan=3, sticky="w", pady=4)

        if not cands:
            ttk.Label(self.b_inner, text="No match above threshold in the time window.").grid(
                row=1, column=0, columnspan=3, sticky="w"
            )
            return

        cols = 3
        for j, c in enumerate(cands):
            rb = self.records_b[c.b_index]
            thumb = bgr_to_thumbnail(rb.crop) if rb.crop is not None else None
            if thumb is not None:
                self._match_thumb_refs.append(thumb)
            cell = ttk.Frame(self.b_inner, padding=4, relief=tk.RIDGE)
            cell.grid(row=1 + j // cols, column=j % cols, padx=4, pady=4, sticky="n")
            if thumb is not None:
                tk.Button(
                    cell, image=thumb,
                    command=lambda r=rb: self._show_full_frame(r),
                ).pack()
            ttk.Label(cell, text=f"B[{c.b_index}] {rb.class_name}").pack()
            ttk.Label(cell, text=f"sim {c.similarity:.3f}").pack()
            ttk.Label(cell, text=f"Δt {c.delta_seconds:+.0f}s").pack()

    def _show_full_frame(self, record) -> None:
        """Open the source frame with the detection's bounding box drawn."""
        image = cv2.imread(record.frame_path)
        if image is None:
            messagebox.showerror("Cannot open image", record.frame_path)
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        x1, y1, x2, y2 = record.bbox
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        # Scale down large frames to fit a reasonable window.
        max_w, max_h = 1000, 750
        pil.thumbnail((max_w, max_h), Image.LANCZOS)

        win = tk.Toplevel(self)
        win.title(record.frame_name)
        photo = ImageTk.PhotoImage(pil)
        lbl = ttk.Label(win, image=photo)
        lbl.image = photo  # keep ref
        lbl.pack()
        ttk.Label(
            win,
            text=f"{record.point} — {record.frame_name} — "
                 f"{record.class_name} — {record.timestamp}",
        ).pack(pady=4)


def main() -> None:
    app = ReIDApp()
    app.mainloop()


if __name__ == "__main__":
    main()
