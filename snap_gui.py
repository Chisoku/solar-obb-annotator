#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
snap_gui.py — Giao dien (GUI) sua tay OBB tam pin + can thang theo DUONG GUIDE.

Tkinter + matplotlib nhung (khong can cv2 co GUI). Co nut bam, thanh truot, danh sach anh.

Tinh nang:
  - Chon che do bang nut: SUA (keo goc) / VE GUIDE (2 cham) / THEM BOX (4 cham).
  - Ve guide: 2 cham (tu bat vao goc pin gan nhat) -> 1 duong thang chuan.
  - Keo goc pin gan guide -> tha ra tu dinh vao duong.  Nut "Snap tat ca vao guide".
  - Nut "Auto-snap anh nay" -> chay thuat toan tu dong (snap_obb) cho rieng anh dang mo.
  - Truoc/Sau anh (tu luu), danh sach anh de nhay nhanh, thanh truot nguong snap.
  - Chuot phai keo = pan, lan chuot = zoom.

Chay:
  py snap_gui.py
  py snap_gui.py --labels "solar-segment-obb.yolov8-obb/train/labels" \
                 --images "solar-segment-obb.yolov8-obb/train/images" \
                 --out    "solar-segment-obb.yolov8-obb/train/labels_fixed"
"""
import os, sys, glob, argparse, shutil, json
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from snap_editor import find_image, load_boxes, save_boxes, project_to_line, dist_to_line
try:
    import snap_obb
except Exception:
    snap_obb = None


class App:
    def __init__(self, root, labels="", images="", out=None):
        self.root = root
        self.root.title("Snap Editor — căn thẳng OBB tấm pin")
        self.labels_dir = labels
        self.images_dir = images
        self.out_dir = out
        self.files = []
        self.idx = 0
        self.boxes = []
        self.cxy = np.empty((0, 2))
        self.img = None; self.w = self.h = 0; self.stem = ""
        self.guides = []            # nhieu duong guide: moi cai = [p1, p2] (np array)
        self.guide_link = {}        # cap song song: index -> index ban (di chuyen cung nhau)
        self.attach = {}            # (bi,ci) -> (guide_idx, t): goc dinh vao guide, keo guide thi di theo
        self.weld_groups = []       # list cac set {(gi,pi),...}: cac dinh guide HAN chung 1 cham
        self.undo_stack = []; self.redo_stack = []   # Ctrl+Z hoan tac / Ctrl+Y lam lai (tung thao tac)
        self._pre = None; self._drag_moved = False   # snapshot truoc khi keo
        self.guide_pending = []     # cac diem dang cham (chua du 2)
        self.done = set()           # ten file da danh dau XONG
        self.guides_locked = False  # khoa guide (chong sua nham)
        self.hover_guide = None     # dau guide dang duoc rE chuot toi (to sang)
        self.hover_box = None       # tam dang re chuot toi (spotlight, lam mo xung quanh)
        self.mode = "edit"          # edit | guide | add
        self.add_pts = []
        self.drag = None            # ("corner",bi,ci) | ("guide",gi,pi)
        self.panning = None
        self.dirty = False
        self._last_xy = (0, 0)
        self._click_cand = None     # (mode,xd,yd,ex,ey) de phan biet click vs keo
        self._press_xy = None

        self._build_ui()
        if self.labels_dir and self.images_dir:
            self._scan_files()

    # ------------------------------------------------ UI ------------------------------------------------
    def _build_ui(self):
        left = ttk.Frame(self.root, padding=6)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        def sep():
            ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=6)

        ttk.Button(left, text="📂 Mở thư mục...", command=self.choose_dirs).pack(fill=tk.X)
        sep()
        nav = ttk.Frame(left); nav.pack(fill=tk.X)
        ttk.Button(nav, text="◀ Trước", command=lambda: self.go(-1)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(nav, text="Sau ▶", command=lambda: self.go(+1)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.pos_lbl = ttk.Label(left, text="0 / 0", anchor="center"); self.pos_lbl.pack(fill=tk.X)
        self.done_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="✔ Đánh dấu ảnh này ĐÃ XONG", variable=self.done_var,
                        command=self.toggle_done).pack(anchor="w")
        self.done_lbl = ttk.Label(left, text="✔ Đã xong: 0 / 0", foreground="#0a6a0a")
        self.done_lbl.pack(fill=tk.X)
        ttk.Button(left, text="💾 Lưu (S)", command=self.save).pack(fill=tk.X, pady=2)
        self.save_lbl = ttk.Label(left, text="", foreground="#0a6a0a", wraplength=210,
                                  font=("", 8)); self.save_lbl.pack(fill=tk.X)
        self.spotlight_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="🔦 Rê chuột: làm mờ tấm xung quanh",
                        variable=self.spotlight_var, command=self.refresh_all).pack(anchor="w")
        ttk.Button(left, text="🗑 Xóa ẢNH này (→ _deleted)",
                   command=self.delete_image).pack(fill=tk.X, pady=(4, 0))
        sep()

        ttk.Label(left, text="Chế độ:").pack(anchor="w")
        self.mode_var = tk.StringVar(value="edit")
        for val, txt in [("edit", "✋ Sửa (kéo góc / đầu guide)"),
                         ("guide", "📏 Vẽ Guide (nhiều đường)"),
                         ("genguide", "✨ Tự tạo guide (bấm 1 dãy)"),
                         ("snapside", "🧲 Snap (bấm để dính góc vào line)"),
                         ("add", "➕ Thêm box (4 chấm)")]:
            ttk.Radiobutton(left, text=txt, value=val, variable=self.mode_var,
                            command=self._mode_changed).pack(anchor="w")
        sep()

        ttk.Label(left, text="Snap: chọn 🧲 rồi BẤM 1 phát → mỗi góc gần line (≤5px) tự dính vào line gần nhất.",
                  wraplength=210, foreground="#555", font=("", 8)).pack(anchor="w")
        ttk.Button(left, text="Xóa hết Guide (C)", command=self.clear_guide).pack(fill=tk.X, pady=1)
        ttk.Button(left, text="🗑 Xóa box đang trỏ vào (D)", command=self.delete_box).pack(fill=tk.X, pady=1)
        ttk.Label(left, text="Vẽ nhiều đường tuỳ ý · kéo chấm hồng để chỉnh · rê vào chấm rồi bấm D = xóa riêng đường đó",
                  wraplength=210, foreground="#555", font=("", 8)).pack(anchor="w")
        self.lock_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="🔒 Khoá guide (ưng rồi, chống sửa nhầm)",
                        variable=self.lock_var, command=self.toggle_lock).pack(anchor="w")
        self.parallel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="➕ Vẽ kèm 1 đường song song (cách 0.75px)",
                        variable=self.parallel_var).pack(anchor="w")
        sep()

        ttk.Button(left, text="⚡ Auto-snap ảnh này", command=self.auto_snap).pack(fill=tk.X)
        ttk.Label(left, text="(dùng thuật toán tự động cho riêng ảnh đang mở)",
                  wraplength=200, foreground="#555").pack(anchor="w")
        sep()

        zoom = ttk.Frame(left); zoom.pack(fill=tk.X, pady=2)
        ttk.Button(zoom, text="－", width=3, command=lambda: self.zoom_center(1.25)).pack(side=tk.LEFT)
        ttk.Button(zoom, text="＋", width=3, command=lambda: self.zoom_center(0.8)).pack(side=tk.LEFT)
        ttk.Button(zoom, text="Đặt lại (R)", command=self.reset_view).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(left, text="Mọi chế độ: Ctrl+lăn = zoom · kéo chuột trái vùng trống = di ảnh. "
                             "(Chế độ vẽ: bấm-nhả tại chỗ = chấm điểm, bấm-kéo = di ảnh)",
                  wraplength=210, foreground="#555", font=("", 8)).pack(anchor="w")
        sep()

        ttk.Label(left, text="Danh sách ảnh:").pack(anchor="w")
        lb_frame = ttk.Frame(left); lb_frame.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(lb_frame, width=30, height=16, activestyle="dotbox")
        sb = ttk.Scrollbar(lb_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        # canvas
        self.fig = Figure(figsize=(11, 8.5))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # tu lay focus khi re chuot vao -> Ctrl+lan zoom duoc NGAY (khong can click truoc)
        cw = self.canvas.get_tk_widget()
        cw.configure(takefocus=True)
        cw.bind("<Enter>", lambda e: cw.focus_set())
        self.status = ttk.Label(right, text="Mở thư mục để bắt đầu.", anchor="w", relief="sunken")
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

        self.im = None
        # lop phu toi TOAN ANH co "lo" tai tam dang hover (spotlight)
        self.dim_patch = PathPatch(MplPath([(0, 0), (0, 0)]), facecolor=(0, 0, 0, 0.55),
                                   edgecolor="none", zorder=3)
        self.dim_patch.set_visible(False)
        self.ax.add_patch(self.dim_patch)
        (self.box_line,) = self.ax.plot([], [], "-", color="#00dd00", lw=1.0, zorder=4)
        self.corner_sc = self.ax.scatter([], [], s=14, c="#ff8c00", zorder=5)
        (self.guide_line,) = self.ax.plot([], [], "-", color="#ff40ff", lw=1.2)
        self.guide_sc = self.ax.scatter([], [], s=75, c="#ff40ff", edgecolors="white",
                                        linewidths=0.8, zorder=6)
        self.hover_sc = self.ax.scatter([], [], s=240, facecolors="none",
                                        edgecolors="#ffee00", linewidths=2.2, zorder=7)
        self.add_sc = self.ax.scatter([], [], s=45, c="#00ffff", marker="x", zorder=6)

        c = self.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("motion_notify_event", self.on_motion)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("scroll_event", self.on_scroll)
        c.mpl_connect("key_press_event", self.on_key)
        # phim tat Tk (backup)
        self.root.bind("<Key>", self._tk_key)
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-Z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-Y>", lambda e: self.redo())
        self.root.bind("<Control-h>", lambda e: self.unweld())
        self.root.bind("<Control-H>", lambda e: self.unweld())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------ file mgmt ------------------------------------------------
    def choose_dirs(self):
        ld = filedialog.askdirectory(title="Chọn thư mục LABELS (.txt)")
        if not ld:
            return
        idr = filedialog.askdirectory(title="Chọn thư mục IMAGES")
        if not idr:
            return
        self.labels_dir, self.images_dir = ld, idr
        self.out_dir = None  # _scan_files se mac dinh sang thu muc _fixed (khong de len goc)
        self._scan_files()

    def _default_out(self):
        """Mac dinh: thu muc anh em '<labels>_fixed' -> KHONG BAO GIO de len labels goc."""
        base = self.labels_dir.rstrip("/\\")
        return base + "_fixed"

    def _scan_files(self):
        if not self.out_dir:
            self.out_dir = self._default_out()
        # an toan: khong cho out_dir trung labels goc
        if os.path.abspath(self.out_dir) == os.path.abspath(self.labels_dir):
            self.out_dir = self.labels_dir.rstrip("/\\") + "_fixed"
        self.save_lbl.config(text=f"Lưu vào: {os.path.basename(self.out_dir)}/  "
                                  f"(nhãn gốc labels/ + ảnh giữ nguyên)")
        self.files = sorted(glob.glob(os.path.join(self.labels_dir, "*.txt")))
        self.load_done()
        self._populate_list()
        if self.files:
            self.idx = 0
            self.load_current()
        else:
            self.status.config(text="Không thấy .txt trong " + self.labels_dir)

    def out_path(self):
        base = os.path.basename(self.files[self.idx])
        return os.path.join(self.out_dir, base) if self.out_dir else self.files[self.idx]

    def load_current(self):
        lf = self.files[self.idx]
        self.stem = os.path.splitext(os.path.basename(lf))[0]
        ip = find_image(self.images_dir, self.stem)
        img = cv2.imread(ip) if ip else None
        if img is None:
            img = np.full((768, 1024, 3), 60, np.uint8)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.h, self.w = self.img.shape[:2]
        src = self.out_path() if (self.out_dir and os.path.exists(self.out_path())) else lf
        self.boxes = load_boxes(src)
        self.guides = []; self.guide_link = {}; self.attach = {}; self.guide_pending = []
        self.weld_groups = []
        self.hover_guide = None; self.hover_box = None
        self.undo_stack = []; self.redo_stack = []; self._pre = None; self._drag_moved = False
        self.add_pts = []; self.drag = None; self.dirty = False
        if self.im is None:
            self.im = self.ax.imshow(self.img)
        else:
            self.im.set_data(self.img)
        self.ax.set_xlim(0, self.w); self.ax.set_ylim(self.h, 0)
        self.done_var.set(os.path.basename(lf) in self.done)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.idx); self.listbox.see(self.idx)
        self.refresh_all()

    # ------------------------------------------------ draw ------------------------------------------------
    def rebuild_cxy(self):
        self.cxy = (np.concatenate([b["pts"] * [self.w, self.h] for b in self.boxes], 0)
                    if self.boxes else np.empty((0, 2)))

    def refresh_all(self):
        self.rebuild_cxy()
        xs, ys = [], []
        for b in self.boxes:
            p = b["pts"] * [self.w, self.h]
            for k in (0, 1, 2, 3, 0):
                xs.append(p[k, 0]); ys.append(p[k, 1])
            xs.append(np.nan); ys.append(np.nan)
        self.box_line.set_data(xs, ys)
        self.corner_sc.set_offsets(self.cxy if len(self.cxy) else np.empty((0, 2)))
        self._refresh_guide(); self._refresh_add(); self._refresh_hover(); self._refresh_dim()
        self._update_status()
        self.canvas.draw_idle()

    def _refresh_guide(self):
        endpoints, gx, gy = [], [], []
        for g in self.guides:
            a, b = np.array(g[0], float), np.array(g[1], float)
            endpoints.append(a); endpoints.append(b)
            gx += [a[0], b[0], np.nan]; gy += [a[1], b[1], np.nan]   # ve DUNG doan (khong keo ra ngoai)
        for p in self.guide_pending:
            endpoints.append(np.array(p, float))
        col = "#9a9a9a" if self.guides_locked else "#ff40ff"
        self.guide_line.set_data(gx, gy); self.guide_line.set_color(col)
        self.guide_sc.set_offsets(np.array(endpoints) if endpoints else np.empty((0, 2)))
        self.guide_sc.set_facecolor(col)

    def _refresh_add(self):
        self.add_sc.set_offsets(np.array(self.add_pts) if self.add_pts else np.empty((0, 2)))

    def _refresh_hover(self):
        hv = self.hover_guide
        if hv is not None and hv[0] < len(self.guides):
            self.hover_sc.set_offsets([self.guides[hv[0]][hv[1]]])
        else:
            self.hover_sc.set_offsets(np.empty((0, 2)))

    @staticmethod
    def _signed_area(poly):
        s = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
            s += x1 * y2 - x2 * y1
        return s

    def _refresh_dim(self):
        """Spotlight: phu lop toi TOAN ANH, khoet 1 lo dung tai tam dang hover."""
        if self.hover_box is None or not self.spotlight_var.get():
            self.dim_patch.set_visible(False); return
        w, h = self.w, self.h
        outer = [(0, 0), (w, 0), (w, h), (0, h)]                 # bao ca anh
        inner = [tuple(p) for p in (self.boxes[self.hover_box]["pts"] * [w, h])]
        if (self._signed_area(outer) > 0) == (self._signed_area(inner) > 0):
            inner = inner[::-1]                                  # nguoc chieu outer -> tao LO
        verts = outer + [outer[0]] + inner + [inner[0]]
        codes = ([MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY] +
                 [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY])
        self.dim_patch.set_path(MplPath(verts, codes))
        self.dim_patch.set_visible(True)

    @staticmethod
    def _pt_in_convex(p, poly):
        s = None
        for i in range(4):
            a, b = poly[i], poly[(i + 1) % 4]
            cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            sg = cr >= 0
            if s is None:
                s = sg
            elif sg != s:
                return False
        return True

    def _box_at(self, xd, yd):
        """Tra ve index tam chua diem (xd,yd) px, hoac None."""
        if not self.boxes or not len(self.cxy):
            return None
        centers = self.cxy.reshape(-1, 4, 2).mean(1)
        order = np.argsort(np.hypot(centers[:, 0] - xd, centers[:, 1] - yd))[:6]
        p = np.array([xd, yd])
        for bi in order:
            if self._pt_in_convex(p, self.boxes[bi]["pts"] * [self.w, self.h]):
                return int(bi)
        return None

    def _update_status(self):
        star = "  *chưa lưu" if self.dirty else ""
        modes = {"edit": "SỬA", "guide": "VẼ GUIDE", "genguide": "TẠO GUIDE",
                 "snapside": "SNAP", "add": "THÊM BOX"}
        self.pos_lbl.config(text=f"{self.idx+1} / {len(self.files)}")
        lock = " 🔒" if self.guides_locked else ""
        self.status.config(text=f"{self.stem[:60]}{star}   |   box={len(self.boxes)}   |   "
                                f"chế độ={modes[self.mode]}   |   guide={len(self.guides)}{lock}")

    # ------------------------------------------------ helpers ------------------------------------------------
    def thresh(self):
        return 5.0   # nguong snap CO DINH = 5px

    def nearest_corner(self, ex, ey, rad_px=12):
        if not len(self.cxy):
            return None
        disp = self.ax.transData.transform(self.cxy)
        d = np.hypot(disp[:, 0] - ex, disp[:, 1] - ey)
        j = int(d.argmin())
        return (j // 4, j % 4) if d[j] <= rad_px else None

    def nearest_box(self, xd, yd):
        if not self.boxes:
            return None
        centers = np.array([b["pts"].mean(0) * [self.w, self.h] for b in self.boxes])
        return int(np.hypot(centers[:, 0] - xd, centers[:, 1] - yd).argmin())

    def snap_pt_to_corner(self, xd, yd, ex, ey):
        nc = self.nearest_corner(ex, ey, rad_px=16)
        return (self.boxes[nc[0]]["pts"][nc[1]] * [self.w, self.h]) if nc else np.array([xd, yd])

    def min_edge(self):
        return 1.0   # cach canh toi thieu CO DINH (chi chong lam bep tam)

    def _nearest_guide_line(self, pt):
        best, bd = None, 1e18
        for g in self.guides:
            a, b = np.array(g[0], float), np.array(g[1], float)
            d = dist_to_line(pt, a, b)
            if d < bd:
                bd, best = d, (a, b)
        return best, bd

    def _nearest_guide_endpoint(self, ex, ey, rad_px=12):
        items = [(gi, pi, g[pi]) for gi, g in enumerate(self.guides) for pi in (0, 1)]
        if not items:
            return None
        disp = self.ax.transData.transform(np.array([it[2] for it in items], dtype=float))
        d = np.hypot(disp[:, 0] - ex, disp[:, 1] - ey)
        j = int(d.argmin())
        return (items[j][0], items[j][1]) if d[j] <= rad_px else None

    @staticmethod
    def _min_edge_len(pts):
        return min(float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4))

    def _clamp_pt(self, p):
        """Kep 1 diem nam TRONG anh (khong cho goc box ra ngoai hinh)."""
        return np.array([min(max(float(p[0]), 0.0), float(self.w)),
                         min(max(float(p[1]), 0.0), float(self.h))])

    @staticmethod
    def _line_intersect(a1, a2, b1, b2):
        """Giao diem 2 duong (a1a2) va (b1b2). None neu song song."""
        d1 = a2 - a1; d2 = b2 - b1
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-9:
            return None
        t = ((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / den
        return a1 + t * d1

    def _guide_geoms(self):
        """Tra ve (G, inters): G = list (a,b) px; inters = list (pt, i, j) giao diem trong ca 2 doan."""
        G = [(np.array(g[0], float), np.array(g[1], float)) for g in self.guides]

        def inseg(pt, a, b):
            ab2 = float(np.dot(b - a, b - a))
            return ab2 >= 1e-9 and -0.05 <= float(np.dot(pt - a, b - a) / ab2) <= 1.05

        inters = []
        for i in range(len(G)):
            for j in range(i + 1, len(G)):
                if self.guide_link.get(i) == j:      # cap song song -> khong giao
                    continue
                pt = self._line_intersect(G[i][0], G[i][1], G[j][0], G[j][1])
                if pt is not None and inseg(pt, *G[i]) and inseg(pt, *G[j]):
                    inters.append((pt, i, j))
        return G, inters

    def _snap_corner(self, bi, ci, G, inters, th):
        """Snap goc (bi,ci) vao GIAO DIEM (uu tien) hoac LINE gan nhat, trong nguong; ghi attach."""
        pts = self.boxes[bi]["pts"] * [self.w, self.h]
        c = pts[ci]; old = c.copy()
        # 1) giao diem gan nhat
        bestX, bdX = None, 1e18
        for (pt, i, j) in inters:
            d = float(np.linalg.norm(c - pt))
            if d < bdX:
                bdX, bestX = d, (pt, i, j)
        if bestX is not None and bdX <= th:
            pt, i, j = bestX
            pts[ci] = self._clamp_pt(pt)
            if self._min_edge_len(pts) < self.min_edge():
                pts[ci] = old; return False
            self.boxes[bi]["pts"] = pts / [self.w, self.h]
            self.attach[(bi, ci)] = ("X", i, j); return True
        # 2) line gan nhat (theo doan)
        bestL, bdL, gidx, bt = None, 1e18, -1, 0.0
        for gi, (a, b) in enumerate(G):
            ab2 = float(np.dot(b - a, b - a))
            if ab2 < 1e-9:
                continue
            t = float(np.dot(c - a, b - a) / ab2)
            if t < -0.05 or t > 1.05:
                continue
            d = float(np.linalg.norm(c - (a + t * (b - a))))
            if d < bdL:
                bdL, bestL, gidx, bt = d, (a, b), gi, t
        if bestL is not None and bdL <= th:
            a, b = bestL
            pts[ci] = self._clamp_pt(a + bt * (b - a))
            if self._min_edge_len(pts) < self.min_edge():
                pts[ci] = old; return False
            self.boxes[bi]["pts"] = pts / [self.w, self.h]
            self.attach[(bi, ci)] = ("L", gidx, bt); return True
        return False

    # --- danh dau da xong (luu local trong snap_done.json) ---
    def _done_path(self):
        return os.path.join(os.path.dirname(self.labels_dir.rstrip("/\\")), "snap_done.json")

    def load_done(self):
        self.done = set()
        p = self._done_path()
        if os.path.exists(p):
            try:
                self.done = set(json.load(open(p, encoding="utf-8")))
            except Exception:
                self.done = set()

    def save_done(self):
        try:
            json.dump(sorted(self.done), open(self._done_path(), "w", encoding="utf-8"),
                      ensure_ascii=False)
        except OSError:
            pass

    def toggle_done(self):
        if not self.files:
            return
        name = os.path.basename(self.files[self.idx])
        if self.done_var.get():
            self.done.add(name)
        else:
            self.done.discard(name)
        self.save_done(); self._color_item(self.idx); self._update_done_count()

    def toggle_lock(self):
        self.guides_locked = bool(self.lock_var.get())
        self.refresh_all()

    def _populate_list(self):
        self.listbox.delete(0, tk.END)
        for i, f in enumerate(self.files):
            self.listbox.insert(tk.END, f"{i+1}. {os.path.basename(f)}")
        for i in range(len(self.files)):
            self._color_item(i)
        self._update_done_count()

    def _color_item(self, i):
        done = os.path.basename(self.files[i]) in self.done
        self.listbox.itemconfig(i, {"bg": "#c7f0c7" if done else "white",
                                    "selectbackground": "#5aa85a" if done else "#3579d8"})

    def _update_done_count(self):
        names = {os.path.basename(f) for f in self.files}
        self.done_lbl.config(text=f"✔ Đã xong: {len(self.done & names)} / {len(self.files)}")

    # ------------------------------------------------ actions ------------------------------------------------
    def _mode_changed(self):
        self.mode = self.mode_var.get()
        self.add_pts = []; self.guide_pending = []
        self.refresh_all()

    def gen_guides_for_strip(self, xd, yd):
        """Bam vao 1 dãy -> tu tao cac guide (duong cot doc + duong hang ngang) cho dãy do."""
        if snap_obb is None or not self.boxes:
            self.status.config(text="Không tạo được (thiếu box/snap_obb)."); return
        polys = [b["pts"] * [self.w, self.h] for b in self.boxes]
        theta = snap_obb.estimate_theta(polys)
        strips, pw, ph = snap_obb.group_strips(polys, theta)
        bi = self._box_at(xd, yd)
        if bi is None:
            bi = self.nearest_box(xd, yd)
        strip = next((s for s in strips if bi in s), None)
        if not strip or len(strip) < 3:
            self.status.config(text="Không thấy dãy ở đây (bấm trúng 1 tấm trong dãy)."); return
        sub = [polys[i] for i in strip]
        th_t = snap_obb.estimate_theta(sub)
        center = np.concatenate(sub, 0).mean(0)
        cx, cy, L, R, T, B = snap_obb.edges_in_frame(sub, th_t, center)
        pw2 = float(np.median(R - L)); ph2 = float(np.median(B - T))
        # tach canh trai/phai, tren/duoi rieng -> moi CANH TAM 1 line:
        #   vien ngoai chi co 1 canh -> LINE DON; giua 2 tam co 2 canh (khe) -> LINE DOI
        xL, _ = snap_obb.cluster_1d(L, 0.3 * pw2)
        xR, _ = snap_obb.cluster_1d(R, 0.3 * pw2)
        yT, _ = snap_obb.cluster_1d(T, 0.3 * ph2)
        yB, _ = snap_obb.cluster_1d(B, 0.3 * ph2)
        ymin, ymax = float(T.min()) - 2, float(B.max()) + 2
        xmin, xmax = float(L.min()) - 2, float(R.max()) + 2
        backM = snap_obb.rot_matrix(-th_t)
        back = lambda p: np.asarray(p, float) @ backM + center           # he xoay -> anh
        new_g = []
        for x in list(xL) + list(xR):                                    # vach doc
            new_g.append([self._clamp_pt(back([x, ymin])), self._clamp_pt(back([x, ymax]))])
        for y in list(yT) + list(yB):                                    # vach ngang
            new_g.append([self._clamp_pt(back([xmin, y])), self._clamp_pt(back([xmax, y]))])
        if not new_g:
            self.status.config(text="Không tạo được guide."); return
        self._push_undo()
        self.guides.extend(new_g)
        self.refresh_all()
        self.status.config(text=f"Đã tạo {len(new_g)} guide cho dãy — chỉnh lại rồi 🧲 snap.")

    def _snap_side(self, xd, yd):
        """Bam 1 phat -> moi goc gan line se dinh vao GIAO DIEM (uu tien) / LINE gan nhat & gan (attach)."""
        if not self.guides:
            self.status.config(text="Chưa có guide. Vẽ guide trước."); return
        G, inters = self._guide_geoms()
        pre = self._snapshot()
        th = self.thresh(); n = 0
        for bi in range(len(self.boxes)):
            for ci in range(4):
                if self._snap_corner(bi, ci, G, inters, th):
                    n += 1
        if n:
            self._push_undo(pre); self.dirty = True
        self.refresh_all()
        self.status.config(text=f"Snap {n} góc (giao điểm/line gần nhất ≤{th:.0f}px) — kéo line là góc theo.")

    def _delete_guide(self, gk):
        """Xoa RIENG 1 duong guide (index gk), cap nhat lai guide_link + attach theo index moi."""
        if gk < 0 or gk >= len(self.guides):
            return
        self._push_undo()
        del self.guides[gk]

        def remap(i):                                # index cu -> index moi (None neu la duong bi xoa)
            return None if i == gk else (i if i < gk else i - 1)

        self.guide_link = {na: nb for a, b in self.guide_link.items()
                           for na, nb in [(remap(a), remap(b))]
                           if na is not None and nb is not None}
        newatt = {}
        for key, val in self.attach.items():
            if val[0] == "L":
                ng = remap(val[1])
                if ng is not None:
                    newatt[key] = ("L", ng, val[2])
            else:                                    # ("X", gA, gB)
                ngA, ngB = remap(val[1]), remap(val[2])
                if ngA is not None and ngB is not None:
                    newatt[key] = ("X", ngA, ngB)
        self.attach = newatt
        newgroups = []                               # reindex weld theo index moi
        for grp in self.weld_groups:
            ng = {(remap(gi), pi) for (gi, pi) in grp if remap(gi) is not None}
            if len(ng) >= 2:
                newgroups.append(ng)
        self.weld_groups = newgroups
        self.hover_guide = None
        self.refresh_all()
        self.status.config(text=f"Đã xóa 1 guide (còn {len(self.guides)}).")

    def clear_guide(self):
        if self.guides or self.guide_pending:
            self._push_undo()
        self.guides = []; self.guide_link = {}; self.attach = {}; self.guide_pending = []
        self.weld_groups = []; self.hover_guide = None; self.refresh_all()

    def delete_box(self):
        bi = self._box_at(*self._last_xy)   # chi xoa box DANG TRO VAO (chuot nam trong box)
        if bi is not None:
            self._push_undo()
            self.boxes.pop(bi); self.attach = {}   # index box doi -> bo gan
            self.dirty = True; self.refresh_all()

    def auto_snap(self):
        if snap_obb is None:
            messagebox.showerror("Lỗi", "Không import được snap_obb.py"); return
        items = [(b["cid"], b["pts"]) for b in self.boxes]
        out_items, st = snap_obb.snap_image(items, self.w, self.h)
        self._push_undo()
        self.boxes = [{"cid": c, "pts": p} for c, p in out_items]
        self.attach = {}
        self.dirty = True
        self.refresh_all()
        self.status.config(text=f"Auto-snap: {st['snapped']}/{st['n']} box, {st['strips']} dãy, "
                                f"dịch TB {st['mean_disp_px']:.1f}px.")

    def reset_view(self):
        self.ax.set_xlim(0, self.w); self.ax.set_ylim(self.h, 0); self.canvas.draw_idle()

    def save(self):
        if not self.files:
            return
        save_boxes(self.out_path(), self.boxes); self.dirty = False
        self._update_status()
        self.status.config(text=f"Đã lưu: {os.path.basename(self.out_path())}")

    def go(self, delta):
        if not self.files:
            return
        if self.dirty:
            self.save()
        self.idx = int(np.clip(self.idx + delta, 0, len(self.files) - 1))
        self.load_current()

    def delete_image(self):
        """Chuyen anh + nhan hien tai vao thu muc _deleted (khoi phuc duoc), roi sang anh ke."""
        if not self.files:
            return
        lf = self.files[self.idx]
        stem = self.stem
        ip = find_image(self.images_dir, stem)
        if not messagebox.askyesno(
                "Xóa ảnh?",
                f"Chuyển ảnh này + nhãn vào thư mục _deleted (có thể khôi phục)?\n\n{stem}"):
            return
        trash_img = self.images_dir.rstrip("/\\") + "_deleted"
        trash_lbl = self.labels_dir.rstrip("/\\") + "_deleted"
        os.makedirs(trash_img, exist_ok=True)
        os.makedirs(trash_lbl, exist_ok=True)
        try:
            if ip and os.path.exists(ip):
                shutil.move(ip, os.path.join(trash_img, os.path.basename(ip)))
            if os.path.exists(lf):
                shutil.move(lf, os.path.join(trash_lbl, os.path.basename(lf)))
            op = self.out_path()  # ban da sua (derived) -> xoa han
            if self.out_dir and os.path.exists(op):
                os.remove(op)
        except OSError as ex:
            messagebox.showerror("Lỗi", f"Không xóa được:\n{ex}")
            return
        self.dirty = False
        del self.files[self.idx]
        self._populate_list()
        if not self.files:
            self.status.config(text="Đã xóa hết ảnh trong tập.")
            return
        self.idx = min(self.idx, len(self.files) - 1)
        self.load_current()
        self.status.config(text=f"Đã chuyển vào _deleted: {stem}")

    def _on_list_select(self, _e):
        sel = self.listbox.curselection()
        if not sel:
            return
        i = sel[0]
        if i != self.idx:
            if self.dirty:
                self.save()
            self.idx = i; self.load_current()

    # ------------------------------------------------ mouse ------------------------------------------------
    def _place_guide_point(self, xd, yd, ex, ey):
        self.guide_pending.append(list(self._clamp_pt([xd, yd])))   # dat dung cho bam, kep trong anh
        if len(self.guide_pending) >= 2:
            self._push_undo()
            p1 = np.array(self.guide_pending[0], float)
            p2 = np.array(self.guide_pending[1], float)
            self.guides.append([p1, p2])
            if self.parallel_var.get():          # kem 1 duong song song cach 0.75px (lien ket cap)
                d = p2 - p1; nrm = np.linalg.norm(d)
                if nrm > 1e-6:
                    off = np.array([-d[1], d[0]]) / nrm * 0.75
                    i = len(self.guides) - 1
                    self.guides.append([p1 + off, p2 + off])
                    j = len(self.guides) - 1
                    self.guide_link[i] = j; self.guide_link[j] = i
            self.guide_pending = []

    # ------------------------------------------------ undo / redo ------------------------------------------------
    def _snapshot(self):
        return ([{"cid": b["cid"], "pts": b["pts"].copy()} for b in self.boxes],
                [[g[0].copy(), g[1].copy()] for g in self.guides],
                dict(self.guide_link), dict(self.attach), [set(g) for g in self.weld_groups])

    def _restore(self, snap):
        boxes, guides, link, attach, weld = snap
        self.boxes = [{"cid": b["cid"], "pts": b["pts"].copy()} for b in boxes]
        self.guides = [[g[0].copy(), g[1].copy()] for g in guides]
        self.guide_link = dict(link); self.attach = dict(attach)
        self.weld_groups = [set(g) for g in weld]
        self.hover_guide = None; self.hover_box = None; self.dirty = True
        self.refresh_all()

    def _push_undo(self, snap=None):
        self.undo_stack.append(snap if snap is not None else self._snapshot())
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            self.status.config(text="Không còn gì để hoàn tác."); return
        self.redo_stack.append(self._snapshot())
        self._restore(self.undo_stack.pop())
        self.status.config(text=f"↶ Hoàn tác (còn {len(self.undo_stack)}).")

    def redo(self):
        if not self.redo_stack:
            self.status.config(text="Không còn gì để làm lại."); return
        self.undo_stack.append(self._snapshot())
        self._restore(self.redo_stack.pop())
        self.status.config(text=f"↷ Làm lại (còn {len(self.redo_stack)}).")

    def _apply_attachments(self):
        """Goc DA DINH -> dat lai theo guide/giao-diem hien tai (di theo khi keo line)."""
        if not self.attach:
            return
        touched = False
        for (bi, ci), att in list(self.attach.items()):
            if bi >= len(self.boxes):
                continue
            if att[0] == "L":                        # dinh vao 1 LINE (theo t doc line)
                _, gi, t = att
                if gi >= len(self.guides):
                    continue
                a1, a2 = self.guides[gi]
                newc = a1 + t * (a2 - a1)
            else:                                    # "X" dinh vao GIAO DIEM 2 line
                _, gA, gB = att
                if gA >= len(self.guides) or gB >= len(self.guides):
                    continue
                pt = self._line_intersect(np.array(self.guides[gA][0], float),
                                          np.array(self.guides[gA][1], float),
                                          np.array(self.guides[gB][0], float),
                                          np.array(self.guides[gB][1], float))
                if pt is None:
                    continue
                newc = pt
            newc = self._clamp_pt(newc)
            pts = self.boxes[bi]["pts"] * [self.w, self.h]
            pts[ci] = newc
            self.boxes[bi]["pts"] = pts / [self.w, self.h]
            touched = True
        if touched:
            self.dirty = True

    def _resync_parallel(self, master_gi):
        """Tinh lai duong ban = duong 'master' dich VUONG GOC dung 1px (luon song song, cach 1px)."""
        partner = self.guide_link.get(master_gi)
        if partner is None or partner >= len(self.guides):
            return
        a1, a2 = self.guides[master_gi]
        d = a2 - a1; nrm = np.linalg.norm(d)
        if nrm < 1e-6:
            return
        nvec = np.array([-d[1], d[0]]) / nrm            # phap tuyen don vi
        b_mid = (self.guides[partner][0] + self.guides[partner][1]) / 2
        a_mid = (a1 + a2) / 2
        if np.dot(b_mid - a_mid, nvec) < 0:             # giu partner o dung ben cu
            nvec = -nvec
        off = nvec * 0.75
        self.guides[partner][0] = self._clamp_pt(a1 + off)
        self.guides[partner][1] = self._clamp_pt(a2 + off)

    def _weld_group(self, ep):
        for grp in self.weld_groups:
            if ep in grp:
                return grp
        return None

    def _weld(self, a, b):
        ga, gb = self._weld_group(a), self._weld_group(b)
        if ga and gb:
            if ga is not gb:
                ga |= gb; self.weld_groups.remove(gb)
        elif ga:
            ga.add(b)
        elif gb:
            gb.add(a)
        else:
            self.weld_groups.append({a, b})

    def unweld(self):
        """Ctrl+H: tach dinh guide dang re chuot toi khoi cham chung."""
        if self.hover_guide is None:
            self.status.config(text="Rê vào 1 đỉnh guide rồi Ctrl+H để tách."); return
        ep = self.hover_guide
        grp = self._weld_group(ep)
        if grp:
            self._push_undo()
            grp.discard(ep)
            if len(grp) <= 1:
                self.weld_groups.remove(grp)
            self.status.config(text="✂ Đã tách đỉnh guide khỏi chấm chung.")
            self.refresh_all()
        else:
            self.status.config(text="Đỉnh này chưa hàn với đỉnh nào.")

    def _magnet_guide_endpoint(self, gi, pi, rad=10.0):
        """Tha dau guide gan ENDPOINT guide khac -> DINH + HAN (di chung); gan LINE -> chi dinh. Keo ra thi go."""
        if gi >= len(self.guides):
            return
        p = self.guides[gi][pi]
        partner = self.guide_link.get(gi)
        best, bd, best_ep = None, rad, None              # 1) endpoint guide khac -> han chung 1 cham
        for gj, g in enumerate(self.guides):
            if gj == gi or gj == partner:
                continue
            for pj in (0, 1):
                q = np.array(g[pj], float)
                d = float(np.linalg.norm(p - q))
                if d < bd:
                    bd, best, best_ep = d, q, (gj, pj)
        if best is not None:
            self.guides[gi][pi] = np.array(best, float)
            self._weld((gi, pi), best_ep)                # HAN 2 dinh -> sau keo di chung
            self._resync_parallel(gi); self._apply_attachments(); return
        best = None                                      # 2) line guide khac (chieu vuong goc, trong doan)
        if best is None:
            bd = rad
            for gj, g in enumerate(self.guides):
                if gj == gi or gj == partner:
                    continue
                a, b = np.array(g[0], float), np.array(g[1], float)
                ab2 = float(np.dot(b - a, b - a))
                if ab2 < 1e-9:
                    continue
                t = float(np.dot(p - a, b - a) / ab2)
                if t < -0.05 or t > 1.05:
                    continue
                foot = a + t * (b - a)
                d = float(np.linalg.norm(p - foot))
                if d < bd:
                    bd, best = d, foot
        if best is not None:
            self.guides[gi][pi] = self._clamp_pt(np.array(best, float))
            self._resync_parallel(gi); self._apply_attachments()

    def _place_add_point(self, xd, yd):
        self.add_pts.append(list(self._clamp_pt([xd, yd])))
        if len(self.add_pts) == 4:
            self._push_undo()
            self.boxes.append({"cid": "0", "pts": np.array(self.add_pts) / [self.w, self.h]})
            self.add_pts = []; self.dirty = True
            self.mode_var.set("edit"); self.mode = "edit"

    def on_press(self, e):
        if e.inaxes != self.ax or e.xdata is None:
            return
        if e.button == 3:
            self.panning = (e.x, e.y, self.ax.get_xlim(), self.ax.get_ylim()); return
        if e.button != 1:
            return
        # UU TIEN: re trung dau guide (co vong vang, chua khoa) -> KEO dau guide o moi che do
        if not self.guides_locked:
            ge = self._nearest_guide_endpoint(e.x, e.y, rad_px=16)
            if ge:
                self._pre = self._snapshot(); self._drag_moved = False
                self.drag = ("guide", ge[0], ge[1]); return
        if self.mode in ("guide", "add", "snapside", "genguide"):
            # CHAM/BAM (bam-nha tai cho) HAY DI ANH (bam-keo)? -> quyet dinh luc tha chuot
            self._click_cand = (self.mode, e.xdata, e.ydata, e.x, e.y)
            self._press_xy = (e.x, e.y)
            self.panning = (e.x, e.y, self.ax.get_xlim(), self.ax.get_ylim())
            return
        # che do SUA: trung goc -> keo goc; vung trong -> di anh
        nc = self.nearest_corner(e.x, e.y)
        if nc:
            self._pre = self._snapshot(); self._drag_moved = False
            self.drag = ("corner", nc[0], nc[1])
        else:
            self.panning = (e.x, e.y, self.ax.get_xlim(), self.ax.get_ylim())

    def on_motion(self, e):
        if e.xdata is not None:
            self._last_xy = (e.xdata, e.ydata)
        if self.panning and e.x is not None:
            x0, y0, (xa, xb), (ya, yb) = self.panning
            bb = self.ax.get_window_extent()
            dx = (e.x - x0) / bb.width * (xb - xa)
            dy = (e.y - y0) / bb.height * (yb - ya)
            self.ax.set_xlim(xa - dx, xb - dx); self.ax.set_ylim(ya - dy, yb - dy)
            self.canvas.draw_idle(); return
        if self.drag and e.xdata is not None:
            self._drag_moved = True
            if self.drag[0] == "guide":
                _, gi, pi = self.drag
                newp = self._clamp_pt([e.xdata, e.ydata])   # guide khong ra ngoai anh
                grp = self._weld_group((gi, pi))
                if grp:
                    for (gj, pj) in grp:                 # cac dinh da HAN -> di chung 1 cham
                        if gj < len(self.guides):
                            self.guides[gj][pj] = newp.copy(); self._resync_parallel(gj)
                else:
                    self.guides[gi][pi] = newp; self._resync_parallel(gi)
                self._apply_attachments()   # goc da dinh guide -> di theo
            else:
                _, bi, ci = self.drag
                self.attach.pop((bi, ci), None)   # keo tay -> tach khoi guide
                pts = self.boxes[bi]["pts"] * [self.w, self.h]
                pts[ci] = self._clamp_pt([e.xdata, e.ydata])   # khong cho ra ngoai anh
                self.boxes[bi]["pts"] = pts / [self.w, self.h]
            self.refresh_all(); return
        # khong keo / khong pan -> HOVER (vong vang cho guide + spotlight cho tam)
        hv = self._nearest_guide_endpoint(e.x, e.y, rad_px=16) \
            if (not self.guides_locked and e.x is not None) else None
        bh = None
        if hv is None and self.mode == "edit" and self.spotlight_var.get() and e.xdata is not None:
            bh = self._box_at(e.xdata, e.ydata)
        if hv != self.hover_guide or bh != self.hover_box:
            self.hover_guide = hv; self.hover_box = bh
            self._refresh_hover(); self._refresh_dim(); self.canvas.draw_idle()

    def on_release(self, e):
        self.panning = None                       # ket thuc di anh (trai/phai)
        if self.drag:
            kind = self.drag[0]
            if self._drag_moved and self._pre is not None:
                self._push_undo(self._pre)        # 1 lan keo = 1 buoc undo
            self._pre = None; self._drag_moved = False
            if kind == "corner":                  # tha goc gan guide -> tu dinh + GAN
                _, bi, ci = self.drag
                G, inters = self._guide_geoms()
                self._snap_corner(bi, ci, G, inters, self.thresh())
                self.dirty = True
            elif kind == "guide":                 # tha dau guide gan guide khac -> dinh vao (nam cham)
                _, gi, pi = self.drag
                self._magnet_guide_endpoint(gi, pi)
            self.drag = None
            self.refresh_all(); return
        # guide/add: neu bam-nha TAI CHO (khong keo) -> dat diem; neu da keo -> chi la di anh
        cc = self._click_cand; self._click_cand = None
        if cc is not None:
            moved = (e.x is not None and self._press_xy is not None and
                     (abs(e.x - self._press_xy[0]) + abs(e.y - self._press_xy[1])) > 5)
            if not moved:
                mode, xd, yd, ex, ey = cc
                if mode == "guide":
                    self._place_guide_point(xd, yd, ex, ey); self.refresh_all()
                elif mode == "add":
                    self._place_add_point(xd, yd); self.refresh_all()
                elif mode == "genguide":
                    self.gen_guides_for_strip(xd, yd)
                else:  # snapside -> bam de dinh goc vao line
                    self._snap_side(xd, yd)

    def zoom_center(self, f):
        """Zoom quanh tam khung nhin (dung cho nut +/-)."""
        x0, x1 = self.ax.get_xlim(); y0, y1 = self.ax.get_ylim()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.ax.set_xlim(cx - (cx - x0) * f, cx + (x1 - cx) * f)
        self.ax.set_ylim(cy - (cy - y0) * f, cy + (y1 - cy) * f)
        self.canvas.draw_idle()

    def on_scroll(self, e):
        if e.inaxes != self.ax or e.xdata is None:
            return
        up = (e.button == "up")
        key = e.key or ""
        x0, x1 = self.ax.get_xlim(); y0, y1 = self.ax.get_ylim()
        if "control" in key or "ctrl" in key:      # Ctrl + vuot = ZOOM quanh con tro
            f = (1 / 1.2) if up else 1.2
            cx, cy = e.xdata, e.ydata
            self.ax.set_xlim(cx - (cx - x0) * f, cx + (x1 - cx) * f)
            self.ax.set_ylim(cy - (cy - y0) * f, cy + (y1 - cy) * f)
        elif "shift" in key:                        # Shift + vuot = di chuyen NGANG
            dx = (x1 - x0) * (-0.18 if up else 0.18)
            self.ax.set_xlim(x0 - dx, x1 - dx)
        else:                                       # vuot 2 ngon = di chuyen DOC (pan)
            dy = (y1 - y0) * (-0.18 if up else 0.18)
            self.ax.set_ylim(y0 - dy, y1 - dy)
        self.canvas.draw_idle()

    def on_key(self, e):
        self._dispatch_key(e.key)

    def _tk_key(self, e):
        if e.state & 0x4:        # dang giu Control -> de bind Ctrl+... xu ly (khong chay phim don)
            return
        self._dispatch_key(e.keysym.lower())

    def _dispatch_key(self, k):
        if k in ("q", "escape"):
            self._on_close()
        elif k == "g":
            self.mode_var.set("guide"); self.mode = "guide"; self.guide_pending = []; self.refresh_all()
        elif k == "l":
            self.lock_var.set(not self.guides_locked); self.toggle_lock()
        elif k == "c":
            self.clear_guide()
        elif k == "f":
            self.mode_var.set("snapside"); self.mode = "snapside"; self.refresh_all()
        elif k == "a":
            self.mode_var.set("add"); self.mode = "add"; self.add_pts = []; self.refresh_all()
        elif k in ("d", "delete", "backspace"):
            if self.hover_guide is not None:      # dang re vao dau guide -> xoa RIENG guide do
                self._delete_guide(self.hover_guide[0])
            else:
                self.delete_box()
        elif k == "n":
            self.go(+1)
        elif k == "b":
            self.go(-1)
        elif k == "s":
            self.save()
        elif k == "r":
            self.reset_view()

    def _on_close(self):
        if self.dirty:
            if messagebox.askyesno("Lưu?", "Còn thay đổi chưa lưu. Lưu trước khi thoát?"):
                self.save()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description="GUI sua tay OBB + can thang theo guide.")
    ap.add_argument("--labels", default="")
    ap.add_argument("--images", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = tk.Tk()
    root.geometry("1500x950")
    App(root, labels=a.labels, images=a.images, out=a.out)
    root.mainloop()


if __name__ == "__main__":
    main()
