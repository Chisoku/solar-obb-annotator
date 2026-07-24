#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
snap_editor.py — Editor sua tay OBB tam pin + can thang theo DUONG GUIDE (matplotlib GUI).

Dung matplotlib (TkAgg) nen KHONG can cv2 co GUI — hop voi may cai opencv-python-headless.

Luong chinh (dung y ban):
  1. Nhan G roi cham 2 diem -> tao 1 DUONG THANG chuan (guide). Moi lan cham TU BAT vao goc
     pin gan nhat (neu co) de duong di qua dung 2 dinh pin.
  2. Keo 1 goc pin lai gan duong guide -> tha ra thi goc TU DINH (chieu vuong goc) vao duong.
  3. Phim F: snap MOT PHAT tat ca goc dang nam gan duong guide vao duong (can ca cot/hang nhanh).

Dieu khien:
  CHUOT
    - Trai keo:  keo goc pin gan nhat. Tha gan guide -> dinh vao guide.
    - Che do G:  2 lan chuot trai = 2 dau guide (tu bat vao goc gan nhat).
    - Che do A:  4 lan chuot trai = 1 box moi.
    - Phai keo:  di chuyen khung nhin (pan).       Con lan: phong to/thu nho quanh con tro.
  PHIM
    G = ve guide (2 cham)      C = xoa guide       F = snap tat ca goc gan guide
    A = them box (4 cham)      D = xoa box gan con tro
    [ / ] = giam / tang nguong bat guide (snap px)
    N = anh sau    B = anh truoc    (tu luu khi chuyen anh)
    S = luu    Z = undo (nap lai anh hien tai)    R = reset zoom    H = an/hien tro giup    Q/ESC = thoat

Chay:
  py snap_editor.py --labels train/labels --images train/images
  py snap_editor.py --labels train/labels_snapped --images train/images         # sua tiep ban auto-snap
  py snap_editor.py --labels train/labels --images train/images --out train/labels_fixed
"""
import os, sys, glob, argparse
import numpy as np
import cv2  # chi dung imread/cvtColor (khong can GUI)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

IMG_EXTS = (".JPG", ".jpg", ".jpeg", ".png", ".PNG", ".JPEG")

# tat cac phim tat mac dinh cua matplotlib de khong xung dot
for _k in list(matplotlib.rcParams):
    if _k.startswith("keymap."):
        matplotlib.rcParams[_k] = []


# ----------------------------- I/O + geometry (da test) -----------------------------
def find_image(images_dir, stem):
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def load_boxes(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 9:
                continue
            try:
                pts = np.array(list(map(float, p[1:9])), dtype=np.float64).reshape(4, 2)
            except ValueError:
                continue
            boxes.append({"cid": p[0], "pts": pts})
    return boxes


def save_boxes(path, boxes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for b in boxes:
            nums = " ".join(f"{v:.6f}" for v in b["pts"].reshape(-1))
            f.write(f"{b['cid']} {nums}\n")


def project_to_line(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        return a.copy()
    t = float((p - a) @ ab) / denom
    return a + t * ab


def dist_to_line(p, a, b):
    return float(np.linalg.norm(p - project_to_line(p, a, b)))


# ----------------------------- editor -----------------------------
class Editor:
    def __init__(self, files, images_dir, out_dir):
        self.files = files
        self.images_dir = images_dir
        self.out_dir = out_dir
        self.idx = 0
        self.boxes = []
        self.cxy = np.empty((0, 2))     # toa do px tat ca goc (N*4,2) - cache de bat nhanh
        self.img = None
        self.w = self.h = 0
        self.stem = ""
        self.guide = []
        self.guide_mode = False
        self.add_mode = False
        self.add_pts = []
        self.drag = None
        self.panning = None            # (x_px, y_px, xlim, ylim)
        self.snap_thresh = 14.0        # px
        self.show_help = True
        self.dirty = False
        # figure
        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        self.fig.canvas.manager.set_window_title("snap_editor")
        self.im = None
        (self.box_line,) = self.ax.plot([], [], "-", color="#00dd00", lw=1.0)
        self.corner_sc = self.ax.scatter([], [], s=14, c="#ff8c00", zorder=5)
        (self.guide_line,) = self.ax.plot([], [], "-", color="#ff50ff", lw=1.2)
        self.guide_sc = self.ax.scatter([], [], s=45, c="#ff50ff", marker="o", zorder=6)
        self.add_sc = self.ax.scatter([], [], s=40, c="#00ffff", marker="x", zorder=6)
        self.help_txt = self.fig.text(0.5, 0.01, "", ha="center", va="bottom",
                                      fontsize=8, family="monospace", color="#0a7a0a")
        self.ax.set_axis_off()

    # ---- path ----
    def out_path(self):
        base = os.path.basename(self.files[self.idx])
        return os.path.join(self.out_dir, base) if self.out_dir else self.files[self.idx]

    # ---- nap anh ----
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
        self.guide = []; self.add_pts = []; self.drag = None; self.dirty = False
        if self.im is None:
            self.im = self.ax.imshow(self.img)
        else:
            self.im.set_data(self.img)
        self.ax.set_xlim(0, self.w); self.ax.set_ylim(self.h, 0)
        self.refresh_boxes(); self.refresh_guide(); self.refresh_add(); self.update_title()
        self.fig.canvas.draw_idle()

    # ---- cache goc px ----
    def rebuild_cxy(self):
        if self.boxes:
            self.cxy = np.concatenate([b["pts"] * [self.w, self.h] for b in self.boxes], 0)
        else:
            self.cxy = np.empty((0, 2))

    # ---- ve boxes ----
    def refresh_boxes(self):
        self.rebuild_cxy()
        xs, ys = [], []
        for b in self.boxes:
            p = b["pts"] * [self.w, self.h]
            for k in (0, 1, 2, 3, 0):
                xs.append(p[k, 0]); ys.append(p[k, 1])
            xs.append(np.nan); ys.append(np.nan)
        self.box_line.set_data(xs, ys)
        self.corner_sc.set_offsets(self.cxy if len(self.cxy) else np.empty((0, 2)))

    def refresh_guide(self):
        if self.guide:
            g = np.array(self.guide)
            self.guide_sc.set_offsets(g)
        else:
            self.guide_sc.set_offsets(np.empty((0, 2)))
        if len(self.guide) == 2:
            a, b = np.array(self.guide[0]), np.array(self.guide[1])
            d = b - a; n = np.linalg.norm(d)
            if n > 1:
                d = d / n * max(self.w, self.h)
                p1, p2 = a - d, b + d
                self.guide_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            else:
                self.guide_line.set_data([], [])
        else:
            self.guide_line.set_data([], [])

    def refresh_add(self):
        if self.add_pts:
            self.add_sc.set_offsets(np.array(self.add_pts))
        else:
            self.add_sc.set_offsets(np.empty((0, 2)))

    def update_title(self):
        mode = "GUIDE (cham 2 diem)" if self.guide_mode else \
               ("ADD-BOX (cham 4 diem)" if self.add_mode else "EDIT")
        star = " *chua luu" if self.dirty else ""
        self.ax.set_title(f"[{self.idx+1}/{len(self.files)}] {self.stem[:50]}{star}   "
                          f"boxes={len(self.boxes)}   mode={mode}   snap={self.snap_thresh:.0f}px",
                          fontsize=10)
        if self.show_help:
            self.help_txt.set_text(
                "Trai-keo goc (tha gan guide=dinh) | Phai-keo=pan | Wheel=zoom || "
                "G guide  C xoaguide  F snap-all  A them  D xoa  [ ] nguong  "
                "N sau  B truoc  S luu  Z undo  R reset  H help  Q thoat")
        else:
            self.help_txt.set_text("")

    # ---- bat goc gan con tro (theo pixel man hinh) ----
    def nearest_corner(self, ex, ey, rad_px=12):
        if not len(self.cxy):
            return None
        disp = self.ax.transData.transform(self.cxy)
        d = np.hypot(disp[:, 0] - ex, disp[:, 1] - ey)
        j = int(d.argmin())
        if d[j] <= rad_px:
            return (j // 4, j % 4)
        return None

    def nearest_box(self, xd, yd):
        if not self.boxes:
            return None
        centers = np.array([b["pts"].mean(0) * [self.w, self.h] for b in self.boxes])
        d = np.hypot(centers[:, 0] - xd, centers[:, 1] - yd)
        return int(d.argmin())

    def snap_pt_to_corner(self, xd, yd, ex, ey):
        nc = self.nearest_corner(ex, ey, rad_px=16)
        if nc:
            return self.boxes[nc[0]]["pts"][nc[1]] * [self.w, self.h]
        return np.array([xd, yd])

    def snap_all_to_guide(self):
        if len(self.guide) < 2:
            return 0
        a, b = np.array(self.guide[0]), np.array(self.guide[1])
        n = 0
        for box in self.boxes:
            pts = box["pts"] * [self.w, self.h]
            ch = False
            for ci in range(4):
                if dist_to_line(pts[ci], a, b) <= self.snap_thresh:
                    pts[ci] = project_to_line(pts[ci], a, b); ch = True; n += 1
            if ch:
                box["pts"] = pts / [self.w, self.h]; self.dirty = True
        return n

    # ---- events ----
    def on_press(self, e):
        if e.inaxes != self.ax or e.xdata is None:
            return
        if e.button == 3:  # phai -> pan
            self.panning = (e.x, e.y, self.ax.get_xlim(), self.ax.get_ylim()); return
        if e.button != 1:
            return
        if self.guide_mode:
            self.guide.append(list(self.snap_pt_to_corner(e.xdata, e.ydata, e.x, e.y)))
            if len(self.guide) >= 2:
                self.guide = self.guide[:2]; self.guide_mode = False
            self.refresh_guide(); self.update_title(); self.fig.canvas.draw_idle(); return
        if self.add_mode:
            self.add_pts.append([e.xdata, e.ydata])
            if len(self.add_pts) == 4:
                pts = np.array(self.add_pts) / [self.w, self.h]
                self.boxes.append({"cid": "0", "pts": pts})
                self.add_pts = []; self.add_mode = False; self.dirty = True
                self.refresh_boxes()
            self.refresh_add(); self.update_title(); self.fig.canvas.draw_idle(); return
        nc = self.nearest_corner(e.x, e.y)
        if nc:
            self.drag = nc

    def on_motion(self, e):
        if self.panning and e.x is not None:
            x0px, y0px, (xa, xb), (ya, yb) = self.panning
            bbox = self.ax.get_window_extent()
            dx = (e.x - x0px) / bbox.width * (xb - xa)
            dy = (e.y - y0px) / bbox.height * (yb - ya)
            self.ax.set_xlim(xa - dx, xb - dx); self.ax.set_ylim(ya - dy, yb - dy)
            self.fig.canvas.draw_idle(); return
        if self.drag and e.xdata is not None:
            bi, ci = self.drag
            pts = self.boxes[bi]["pts"] * [self.w, self.h]
            pts[ci] = [e.xdata, e.ydata]
            self.boxes[bi]["pts"] = pts / [self.w, self.h]
            self.refresh_boxes(); self.fig.canvas.draw_idle()

    def on_release(self, e):
        if e.button == 3:
            self.panning = None; return
        if self.drag:
            bi, ci = self.drag
            pts = self.boxes[bi]["pts"] * [self.w, self.h]
            if len(self.guide) == 2:
                a, b = np.array(self.guide[0]), np.array(self.guide[1])
                if dist_to_line(pts[ci], a, b) <= self.snap_thresh:
                    pts[ci] = project_to_line(pts[ci], a, b)
            self.boxes[bi]["pts"] = pts / [self.w, self.h]
            self.drag = None; self.dirty = True
            self.refresh_boxes(); self.update_title(); self.fig.canvas.draw_idle()

    def on_scroll(self, e):
        if e.inaxes != self.ax or e.xdata is None:
            return
        f = (1 / 1.2) if e.button == "up" else 1.2
        cx, cy = e.xdata, e.ydata
        x0, x1 = self.ax.get_xlim(); y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(cx - (cx - x0) * f, cx + (x1 - cx) * f)
        self.ax.set_ylim(cy - (cy - y0) * f, cy + (y1 - cy) * f)
        self.fig.canvas.draw_idle()

    def on_key(self, e):
        k = e.key
        if k in ("q", "escape"):
            if self.dirty:
                self.save()
            plt.close(self.fig); return
        elif k == "g":
            self.guide_mode = not self.guide_mode; self.guide = []; self.add_mode = False
            self.refresh_guide()
        elif k == "c":
            self.guide = []; self.refresh_guide()
        elif k == "f":
            n = self.snap_all_to_guide(); print(f"[F] snap {n} goc vao guide")
            self.refresh_boxes()
        elif k == "a":
            self.add_mode = not self.add_mode; self.add_pts = []; self.guide_mode = False
            self.refresh_add()
        elif k == "d":
            nb = self.nearest_box(*self._mouse_data())
            if nb is not None:
                self.boxes.pop(nb); self.dirty = True; self.refresh_boxes()
        elif k in ("[",):
            self.snap_thresh = max(2.0, self.snap_thresh - 2)
        elif k in ("]",):
            self.snap_thresh = min(80.0, self.snap_thresh + 2)
        elif k == "n":
            self.go(+1); return
        elif k == "b":
            self.go(-1); return
        elif k == "s":
            self.save()
        elif k == "z":
            self.load_current(); return
        elif k == "r":
            self.ax.set_xlim(0, self.w); self.ax.set_ylim(self.h, 0)
        elif k == "h":
            self.show_help = not self.show_help
        self.update_title(); self.fig.canvas.draw_idle()

    def _mouse_data(self):
        # vi tri chuot hien tai (data coords) tu event cuoi
        return getattr(self, "_last_xy", (self.w / 2, self.h / 2))

    def on_move_track(self, e):
        if e.xdata is not None:
            self._last_xy = (e.xdata, e.ydata)

    # ---- luu / dieu huong ----
    def save(self):
        save_boxes(self.out_path(), self.boxes); self.dirty = False
        print(f"[luu] {self.out_path()}  ({len(self.boxes)} boxes)")
        self.update_title(); self.fig.canvas.draw_idle()

    def go(self, delta):
        if self.dirty:
            self.save()
        self.idx = int(np.clip(self.idx + delta, 0, len(self.files) - 1))
        self.load_current()

    def run(self):
        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("motion_notify_event", self.on_motion)
        c.mpl_connect("motion_notify_event", self.on_move_track)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("scroll_event", self.on_scroll)
        c.mpl_connect("key_press_event", self.on_key)
        self.load_current()
        print("San sang. H an/hien tro giup, Q thoat.")
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Editor sua tay OBB + can thang theo duong guide.")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default=None, help="thu muc luu (mac dinh: ghi de --labels)")
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.labels, "*.txt")))
    if not files:
        print("Khong thay .txt trong", a.labels); sys.exit(1)
    ed = Editor(files, a.images, a.out)
    ed.idx = int(np.clip(a.start, 0, len(files) - 1))
    ed.run()


if __name__ == "__main__":
    main()
