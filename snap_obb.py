#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
snap_obb.py — Can chinh (snap) annotation OBB tam pin mat troi cho thang theo tung "dai" (day).

Boi canh data that (da khao sat trong solar-segment-obb.yolov8-obb):
  - Anh drone top-down; moi "dai" pin la mot DAI DOC rong ~vai tam x hang chuc tam (LUOI 2D),
    KHONG phai cot don 1 tam. => Can align theo ca COT (canh trai/phai) LAN HANG (canh tren/duoi).
  - Thu tu 4 goc trong file KHONG co dinh (2 quy uoc lan lon) => xac dinh canh bang HINH HOC.
  - Co box nhieu (label tren bui cay/loi di) => de nguyen, khong snap.
  - Moi dai co goc nghieng rieng => uoc luong goc theo tung dai (khong ep chung 1 goc).

Trong tam thiet ke: SNAP NHE (gentle) — chi khu jitter vai px, khong dung lai luoi cung toan cuc.
  1. Doc label (normalize) + kich thuoc anh -> toa do pixel.
  2. Uoc luong goc nghieng luoi theta_g (circular mean qua 4*angle, |theta|<=45deg).
  3. Tach panel thanh cac DAI DOC: cluster theo TAM-x (da xoay), khe rong = loi di -> tach dai;
     cot thua/le khong tao cau noi (chong box nhieu noi 2 dai).
  4. Voi moi dai: uoc luong goc rieng, xoay ve truc, tinh 4 canh (trai/phai/tren/duoi).
     - Cluster theo TAM panel (tach bach, khong bi chain) de gan cot & hang.
     - Moi cot: canh trai/phai <- median cua cac panel cung cot.
     - Moi hang: canh tren/duoi <- median cua cac panel cung hang.
     - (Tuy chon) cho 2 cot ke nhau dung chung vach bien -> luoi khit.
     - Gioi han dich chuyen (move cap): panel di chuyen bat thuong -> giu nguyen.
  5. Ghi ra thu muc moi cung format YOLOv8-OBB (normalize lai). Che do --debug ve truoc/sau.

Cach dung:
  # Chay thu vai file + xuat anh so sanh truoc/sau:
  py snap_obb.py --labels train/labels --images train/images --out train/labels_snapped \
                 --debug-dir preview --debug-files a.txt b.txt

  # Chay ca 1 split:
  py snap_obb.py --labels train/labels --images train/images --out train/labels_snapped

  # Chay ca dataset (train/valid/test -> <split>/labels_snapped):
  py snap_obb.py --dataset <root> --debug-dir preview --debug-n 3
"""
import os, sys, glob, math, argparse
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image

IMG_EXTS = (".JPG", ".jpg", ".jpeg", ".png", ".PNG", ".JPEG")


# ----------------------------- I/O -----------------------------
def parse_label(path):
    """Doc file YOLOv8-OBB. Tra ve list (class_id_str, poly(4,2) normalized)."""
    items = []
    with open(path, "r") as f:
        for line in f:
            p = line.split()
            if len(p) < 9:
                continue
            cid = p[0]
            try:
                coords = np.array(list(map(float, p[1:9])), dtype=np.float64).reshape(4, 2)
            except ValueError:
                continue
            items.append((cid, coords))
    return items


def write_label(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for cid, coords in items:
            nums = " ".join(f"{v:.6f}" for v in coords.reshape(-1))
            f.write(f"{cid} {nums}\n")


def find_image(images_dir, stem):
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def image_size(images_dir, stem):
    p = find_image(images_dir, stem)
    if p is None:
        return None
    with Image.open(p) as im:
        return im.size  # (w, h)


# ----------------------------- geom helpers -----------------------------
def estimate_theta(polys_px):
    """Goc nghieng luoi (radian, dau, |theta|<=45deg) qua circular-mean cua 4*edge_angle."""
    acc = 0j
    for poly in polys_px:
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            acc += np.exp(1j * 4 * math.atan2(e[1], e[0]))
    if acc == 0:
        return 0.0
    return math.atan2(acc.imag, acc.real) / 4.0


def rot_matrix(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def cluster_1d(values, tol):
    """Gom gia tri 1D thanh cum (single-linkage: gap < tol). Tra ve (centers, labels).

    LUU Y: chi dung cho gia tri TACH BACH (vd tam cot, tam hang) — KHONG dung cho
    tap top+bottom xen ke (se bi chain qua khe nho)."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    centers = []
    labels = np.empty(len(values), dtype=int)
    cur, cur_vals, last = [], [], None
    for idx in order:
        v = values[idx]
        if last is None or (v - last) <= tol:
            cur.append(idx); cur_vals.append(v)
        else:
            for j in cur:
                labels[j] = len(centers)
            centers.append(float(np.mean(cur_vals)))
            cur, cur_vals = [idx], [v]
        last = v
    if cur:
        for j in cur:
            labels[j] = len(centers)
        centers.append(float(np.mean(cur_vals)))
    return np.array(centers), labels


def corner_disp(new, orig):
    """Dich chuyen tung goc: khoang cach moi goc MOI toi goc GOC gan nhat (bat bien thu tu goc)."""
    d = np.linalg.norm(new[:, None, :] - orig[None, :, :], axis=2)  # 4x4
    return d.min(axis=1)  # 4 gia tri


def edges_in_frame(polys_px, theta, center):
    """Xoay -theta quanh center; tra ve cx,cy,L,R,T,B (mang) cho moi panel (he da xoay ve truc)."""
    R = rot_matrix(-theta)
    n = len(polys_px)
    cx = np.empty(n); cy = np.empty(n)
    L = np.empty(n); Rr = np.empty(n); T = np.empty(n); B = np.empty(n)
    for k, poly in enumerate(polys_px):
        r = (poly - center) @ R.T
        xs = np.sort(r[:, 0]); ys = np.sort(r[:, 1])
        L[k] = xs[:2].mean(); Rr[k] = xs[2:].mean()
        T[k] = ys[:2].mean(); B[k] = ys[2:].mean()
        cx[k] = r[:, 0].mean(); cy[k] = r[:, 1].mean()
    return cx, cy, L, Rr, T, B


# ----------------------------- strip grouping -----------------------------
def group_strips(polys_px, theta_g, aisle_frac=1.5, min_col=3):
    """Tach panel thanh cac DAI DOC (strip) theo tam-x (da xoay). Khe > aisle_frac*pw = loi di.
    Cot thua/le (it panel) khong tao cau noi giua 2 dai. Tra ve (strips, pw, ph)."""
    center = np.concatenate(polys_px, axis=0).mean(axis=0)
    cx, cy, L, Rr, T, B = edges_in_frame(polys_px, theta_g, center)
    pw = float(np.median(Rr - L)); ph = float(np.median(B - T))

    xcenters, xlab = cluster_1d(cx, 0.5 * pw)
    ncol = len(xcenters)
    col_members = [np.where(xlab == c)[0] for c in range(ncol)]
    col_sizes = np.array([len(m) for m in col_members])
    med_size = np.median(col_sizes[col_sizes > 0]) if (col_sizes > 0).any() else 0
    solid = col_sizes >= max(min_col, 0.3 * med_size)  # cot "chac" (loai cot nhieu le)

    order = np.argsort(xcenters)
    strips, cur, prev_x = [], [], None
    for c in order:
        if not solid[c]:
            continue
        x = xcenters[c]
        if prev_x is not None and (x - prev_x) > aisle_frac * pw:
            if cur:
                strips.append(cur); cur = []
        cur.extend(col_members[c].tolist())
        prev_x = x
    if cur:
        strips.append(cur)
    return strips, pw, ph


# ----------------------------- snapping (gentle) -----------------------------
def snap_strip(polys_px, idxs, min_members=3, move_cap_frac=0.6, tile_edges=False):
    """Snap 1 dai: moi cot -> canh trai/phai ve median cung cot; moi hang -> tren/duoi ve median cung hang.
    Uoc luong goc RIENG cho dai. Gioi han dich chuyen. Tra ve dict idx->poly_px moi."""
    sub = [polys_px[i] for i in idxs]
    m = len(sub)
    theta_t = estimate_theta(sub)
    center = np.concatenate(sub, axis=0).mean(axis=0)
    cx, cy, L, Rr, T, B = edges_in_frame(sub, theta_t, center)
    pw = float(np.median(Rr - L)); ph = float(np.median(B - T))

    _, col_lab = cluster_1d(cx, 0.5 * pw)  # cot theo tam-x
    _, row_lab = cluster_1d(cy, 0.5 * ph)  # hang theo tam-y

    newL, newR, newT, newB = L.copy(), Rr.copy(), T.copy(), B.copy()
    for c in np.unique(col_lab):           # snap canh doc theo cot
        mem = np.where(col_lab == c)[0]
        if len(mem) >= min_members:
            newL[mem] = np.median(L[mem]); newR[mem] = np.median(Rr[mem])
    for rr in np.unique(row_lab):          # snap canh ngang theo hang
        mem = np.where(row_lab == rr)[0]
        if len(mem) >= min_members:
            newT[mem] = np.median(T[mem]); newB[mem] = np.median(B[mem])

    if tile_edges:                         # 2 cot ke nhau dung chung vach bien -> luoi khit
        col_ids = np.unique(col_lab)
        col_cx = {c: np.median(cx[col_lab == c]) for c in col_ids}
        col_order = sorted(col_ids, key=lambda c: col_cx[c])
        for a, b in zip(col_order[:-1], col_order[1:]):
            ma = np.where(col_lab == a)[0]; mb = np.where(col_lab == b)[0]
            if len(ma) < min_members or len(mb) < min_members:
                continue
            ra = np.median(newR[ma]); lb = np.median(newL[mb])
            if 0 <= (lb - ra) < 0.5 * pw:  # that su ke nhau (khong qua loi di)
                mid = 0.5 * (ra + lb)
                newR[ma] = mid; newL[mb] = mid

    Rinv = rot_matrix(theta_t)
    cap = move_cap_frac * min(pw, ph)
    out = {}
    for k in range(m):
        Lk, Rk, Tk, Bk = newL[k], newR[k], newT[k], newB[k]
        if Rk < Lk: Lk, Rk = Rk, Lk
        if Bk < Tk: Tk, Bk = Bk, Tk
        rect = np.array([[Lk, Tk], [Rk, Tk], [Rk, Bk], [Lk, Bk]], dtype=np.float64)
        newpoly = rect @ Rinv.T + center
        if corner_disp(newpoly, sub[k]).max() > cap:
            continue  # dich chuyen bat thuong -> giu nguyen (an toan)
        out[idxs[k]] = newpoly
    return out


def snap_image(items, w, h, min_panels=5, aisle_frac=1.5,
               min_members=3, move_cap_frac=0.6, tile_edges=False):
    """Snap toan bo 1 anh. items: list (cid, poly_norm). Tra ve (items_moi, stats)."""
    empty = {"n": 0, "snapped": 0, "strips": 0, "mean_disp_px": 0.0, "max_disp_px": 0.0}
    if not items:
        return items, empty
    polys_px = [c.copy() * np.array([w, h]) for _, c in items]
    theta_g = estimate_theta(polys_px)
    strips, pw, ph = group_strips(polys_px, theta_g, aisle_frac=aisle_frac)

    new_polys, n_strips = {}, 0
    for idxs in strips:
        if len(idxs) < min_panels:
            continue  # dai qua nho / box le / nhieu -> de nguyen
        n_strips += 1
        new_polys.update(snap_strip(polys_px, idxs, min_members=min_members,
                                    move_cap_frac=move_cap_frac, tile_edges=tile_edges))

    out_items, disps = [], []
    for i, (cid, coords) in enumerate(items):
        if i in new_polys:
            np_px = new_polys[i]
            disps.append(corner_disp(np_px, polys_px[i]).mean())
            out_items.append((cid, np_px / np.array([w, h])))
        else:
            out_items.append((cid, coords))
    stats = {
        "n": len(items), "snapped": len(new_polys), "strips": n_strips,
        "mean_disp_px": float(np.mean(disps)) if disps else 0.0,
        "max_disp_px": float(np.max(disps)) if disps else 0.0,
    }
    return out_items, stats


# ----------------------------- debug viz -----------------------------
def draw_compare(images_dir, stem, before_items, after_items, out_path, scale=2):
    if cv2 is None:
        return
    ip = find_image(images_dir, stem)
    if ip is None:
        return
    img = cv2.imread(ip)
    if img is None:
        pil = Image.open(ip).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    for cid, c in before_items:
        pts = (c * [w * scale, h * scale]).astype(np.int32)
        cv2.polylines(big, [pts], True, (0, 0, 255), 1, cv2.LINE_AA)   # do = truoc
    for cid, c in after_items:
        pts = (c * [w * scale, h * scale]).astype(np.int32)
        cv2.polylines(big, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)   # xanh la = sau
    cv2.putText(big, "RED=before  GREEN=after", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, big)


# ----------------------------- driver -----------------------------
def process_dir(labels_dir, images_dir, out_dir, limit=None, debug_dir=None,
                debug_files=None, debug_n=0, **kw):
    files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    if limit:
        files = files[:limit]
    debug_set = set(debug_files or [])
    n_debug_done = 0
    agg = {"files": 0, "boxes": 0, "snapped": 0, "strips": 0}
    for lf in files:
        stem = os.path.splitext(os.path.basename(lf))[0]
        sz = image_size(images_dir, stem)
        if sz is None:
            print(f"  [skip] khong thay anh cho {stem}")
            continue
        w, h = sz
        items = parse_label(lf)
        out_items, stats = snap_image(items, w, h, **kw)
        if out_dir:
            write_label(os.path.join(out_dir, os.path.basename(lf)), out_items)
        agg["files"] += 1
        agg["boxes"] += stats["n"]
        agg["snapped"] += stats["snapped"]
        agg["strips"] += stats["strips"]
        want_debug = (os.path.basename(lf) in debug_set) or (n_debug_done < debug_n)
        if debug_dir and want_debug:
            draw_compare(images_dir, stem, items, out_items,
                         os.path.join(debug_dir, stem + "__cmp.png"))
            n_debug_done += 1
            print(f"  [debug] {stem[:48]:48s} n={stats['n']:3d} strips={stats['strips']:2d} "
                  f"snapped={stats['snapped']:3d} mean_disp={stats['mean_disp_px']:.2f}px "
                  f"max_disp={stats['max_disp_px']:.2f}px")
    print(f"[done] {labels_dir}: files={agg['files']} boxes={agg['boxes']} "
          f"strips={agg['strips']} snapped={agg['snapped']} "
          f"({100*agg['snapped']/max(1,agg['boxes']):.1f}%)")
    return agg


def main():
    ap = argparse.ArgumentParser(description="Snap OBB solar-panel annotations thang theo tung dai.")
    ap.add_argument("--labels")
    ap.add_argument("--images")
    ap.add_argument("--out")
    ap.add_argument("--dataset", help="root chua train/valid/test; ghi ra <split>/labels_snapped")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--debug-dir")
    ap.add_argument("--debug-files", nargs="*", default=None)
    ap.add_argument("--debug-n", type=int, default=0)
    ap.add_argument("--min-panels", type=int, default=5)
    ap.add_argument("--aisle-frac", type=float, default=1.5)
    ap.add_argument("--min-members", type=int, default=3)
    ap.add_argument("--move-cap-frac", type=float, default=0.6)
    ap.add_argument("--tile-edges", action="store_true",
                    help="ep 2 cot ke nhau dung chung vach bien (mac dinh TAT)")
    a = ap.parse_args()

    kw = dict(min_panels=a.min_panels, aisle_frac=a.aisle_frac, min_members=a.min_members,
              move_cap_frac=a.move_cap_frac, tile_edges=a.tile_edges)

    if a.dataset:
        for split in ("train", "valid", "test"):
            ld = os.path.join(a.dataset, split, "labels")
            idr = os.path.join(a.dataset, split, "images")
            if not os.path.isdir(ld):
                continue
            od = os.path.join(a.dataset, split, "labels_snapped")
            print(f"== {split} ==")
            process_dir(ld, idr, od, limit=a.limit, debug_dir=a.debug_dir,
                        debug_files=a.debug_files, debug_n=a.debug_n, **kw)
    else:
        if not (a.labels and a.images):
            ap.error("can --labels va --images (hoac --dataset)")
        process_dir(a.labels, a.images, a.out, limit=a.limit, debug_dir=a.debug_dir,
                    debug_files=a.debug_files, debug_n=a.debug_n, **kw)


if __name__ == "__main__":
    main()
