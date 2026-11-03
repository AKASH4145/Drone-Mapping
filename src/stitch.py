"""Stitch overlapping top-down (nadir) drone photos into one map, and the
segmentation predictions into one labelled map using the same alignment.

How it works:
1. Every photo is shrunk to a working size and SIFT features are found.
2. Each pair of photos is matched. A similarity transform (shift, rotate,
   scale) is fitted with RANSAC, and pairs with enough inliers are kept.
3. All transforms are solved together with one linear least-squares problem,
   so errors do not pile up the way they do when photos are chained one by one.
4. Each map pixel is taken from the photo whose centre is closest to it.
   Centres have the least lens distortion and the best predictions.
5. Class labels are warped with nearest-neighbour interpolation, so class ids
   are never blended into wrong values.

Usage (from the project folder):
    python src/stitch.py --input data/nadir/sheffield_park_3
    python src/stitch.py --input data/nadir/sheffield_park_3 --no-seg
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NODATA = 255


def load_images(folder, work_px):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    imgs, full_shapes = [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print('Could not read', p)
            continue
        full_shapes.append(img.shape[:2])
        s = work_px / max(img.shape[:2])
        imgs.append((p, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)))
    return imgs, full_shapes


def find_features(imgs):
    sift = cv2.SIFT_create(nfeatures=4000)
    feats = []
    for _, img in imgs:
        kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        feats.append((np.float32([k.pt for k in kp]), des))
    return feats


def match_pairs(feats, min_inliers, ratio=0.75):
    """Return {(i, j): (pts_i, pts_j)} with RANSAC inlier matches for each good pair."""
    matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    pairs = {}
    n = len(feats)
    for i in range(n):
        for j in range(i + 1, n):
            (pi, di), (pj, dj) = feats[i], feats[j]
            if di is None or dj is None or len(di) < 2 or len(dj) < 2:
                continue
            knn = matcher.knnMatch(di, dj, k=2)
            good = [m for m, *rest in knn if rest and m.distance < ratio * rest[0].distance]
            if len(good) < min_inliers:
                continue
            a = pi[[m.queryIdx for m in good]]
            b = pj[[m.trainIdx for m in good]]
            A, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=4.0)
            if A is None:
                continue
            inl = mask.ravel().astype(bool)
            if inl.sum() >= min_inliers:
                pairs[(i, j)] = (a[inl], b[inl])
    return pairs


def largest_component(n, pairs):
    adj = {i: set() for i in range(n)}
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)
    seen, best = set(), []
    for s in range(n):
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u] - seen:
                seen.add(v)
                stack.append(v)
        if len(comp) > len(best):
            best = comp
    return sorted(best), adj


def solve_global(nodes, pairs, adj, max_pts=100):
    """Solve one similarity transform per photo (x' = a*x - b*y + tx, y' = b*x + a*y + ty).

    For every matched point pair we want T_i(p_i) == T_j(p_j). This is linear in
    (a, b, tx, ty), so all photos are solved at once with least squares.
    The photo with the most connections is fixed as the reference (identity).
    """
    ref = max(nodes, key=lambda k: len(adj[k]))
    col = {k: 4 * c for c, k in enumerate(n for n in nodes if n != ref)}
    nvar = 4 * (len(nodes) - 1)
    rows, rhs = [], []
    rng = np.random.default_rng(0)

    def coeffs(k, x, y, sign, rx, ry):
        """Add +/- T_k(x, y) to the x-row (rx) and y-row (ry); return constant if k is ref."""
        if k == ref:
            return sign * x, sign * y
        c = col[k]
        rx[c:c + 4] += sign * np.array([x, -y, 1, 0])
        ry[c:c + 4] += sign * np.array([y, x, 0, 1])
        return 0.0, 0.0

    for (i, j), (pi, pj) in pairs.items():
        if i not in col and i != ref or j not in col and j != ref:
            continue
        idx = rng.choice(len(pi), min(max_pts, len(pi)), replace=False)
        for (xi, yi), (xj, yj) in zip(pi[idx], pj[idx]):
            rx, ry = np.zeros(nvar), np.zeros(nvar)
            cx1, cy1 = coeffs(i, xi, yi, 1, rx, ry)
            cx2, cy2 = coeffs(j, xj, yj, -1, rx, ry)
            rows += [rx, ry]
            rhs += [-(cx1 + cx2), -(cy1 + cy2)]

    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    T = {ref: np.eye(3)}
    for k, c in col.items():
        a, b, tx, ty = sol[c:c + 4]
        T[k] = np.array([[a, -b, tx], [b, a, ty], [0, 0, 1]])

    # Report alignment error in working pixels
    errs = []
    for (i, j), (pi, pj) in pairs.items():
        if i in T and j in T:
            qi = pi @ T[i][:2, :2].T + T[i][:2, 2]
            qj = pj @ T[j][:2, :2].T + T[j][:2, 2]
            errs.append(np.linalg.norm(qi - qj, axis=1))
    errs = np.concatenate(errs)
    return T, ref, float(np.median(errs)), float(np.percentile(errs, 95))


def segment_frames(imgs, nodes, args):
    """Run the U-Net on each full-size photo and cache the class-id masks."""
    from predict import load_model, predict_full  # torch is only needed in this mode
    import torch

    cache = Path(args.out) / 'frames'
    cache.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = None
    labels = {}
    for k in nodes:
        p, small = imgs[k]
        f = cache / f'{p.stem}_labels.png'
        if f.exists():
            lab = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        else:
            if model is None:
                model = load_model(args.weights, device)
                print(f'Segmenting on {device} (cached in {cache})')
            t0 = time.time()
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            lab = predict_full(model, rgb, device, scale=args.seg_scale)
            cv2.imwrite(str(f), lab)
            print(f'  {p.name}: {time.time() - t0:.1f}s')
        labels[k] = cv2.resize(lab, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    return labels


def composite(imgs, nodes, T, labels=None, max_side=12000):
    """Warp every photo onto one canvas, each pixel coming from the nearest photo centre."""
    corners = []
    for k in nodes:
        h, w = imgs[k][1].shape[:2]
        c = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
        corners.append((T[k] @ c)[:2].T)
    corners = np.vstack(corners)
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    W, H = x1 - x0, y1 - y0
    if max(W, H) > max_side:
        raise SystemExit(f'Canvas would be {W}x{H}, alignment probably failed. '
                         'Try a higher --min-inliers or check the photos overlap.')

    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    rgb = np.zeros((H, W, 3), np.uint8)
    lab = np.full((H, W), NODATA, np.uint8)
    best = np.full((H, W), np.inf, np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    for k in nodes:
        img = imgs[k][1]
        h, w = img.shape[:2]
        M = (shift @ T[k])[:2]
        valid = cv2.warpAffine(np.ones((h, w), np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8)) > 0   # drop black warp edges
        cx, cy = M @ np.array([w / 2, h / 2, 1])
        d = (xx - cx) ** 2 + (yy - cy) ** 2
        take = valid & (d < best)
        best[take] = d[take]
        rgb[take] = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)[take]
        if labels is not None:
            lab[take] = cv2.warpAffine(labels[k], M, (W, H), flags=cv2.INTER_NEAREST,
                                       borderValue=NODATA)[take]
    return rgb, lab, np.isfinite(best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Folder of overlapping nadir photos')
    parser.add_argument('--out', default=str(ROOT / 'outputs' / 'stitch'))
    parser.add_argument('--weights', default=str(ROOT / 'models' / 'unet_r34_best.pth'))
    parser.add_argument('--work', type=int, default=1200, help='Long side of each photo for stitching (px)')
    parser.add_argument('--seg-scale', type=float, default=0.5, help='Downscale before segmentation')
    parser.add_argument('--min-inliers', type=int, default=40, help='Matches needed to link two photos')
    parser.add_argument('--no-seg', action='store_true', help='Only stitch the RGB map')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    imgs, _ = load_images(args.input, args.work)
    if len(imgs) < 2:
        raise SystemExit(f'Need at least 2 photos in {args.input}')
    print(f'{len(imgs)} photos loaded')

    feats = find_features(imgs)
    pairs = match_pairs(feats, args.min_inliers)
    nodes, adj = largest_component(len(imgs), pairs)
    print(f'{len(pairs)} overlapping pairs | {len(nodes)} of {len(imgs)} photos connected')
    dropped = [imgs[k][0].name for k in range(len(imgs)) if k not in nodes]
    if dropped:
        print('  Not connected, left out:', ', '.join(dropped))

    T, ref, med_err, p95_err = solve_global(nodes, pairs, adj)
    print(f'Reference photo: {imgs[ref][0].name} | alignment error median {med_err:.1f}px, '
          f'95th percentile {p95_err:.1f}px')

    labels = None if args.no_seg else segment_frames(imgs, nodes, args)
    rgb, lab, covered = composite(imgs, nodes, T, labels)

    cv2.imwrite(str(out / 'mosaic_rgb.jpg'), rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
    summary = {
        'photos_total': len(imgs), 'photos_used': len(nodes), 'pairs': len(pairs),
        'reference': imgs[ref][0].name, 'align_err_median_px': med_err,
        'align_err_p95_px': p95_err, 'mosaic_size': [rgb.shape[1], rgb.shape[0]],
        'transforms': {imgs[k][0].name: T[k][:2].tolist() for k in nodes},
    }

    if labels is not None:
        from predict import CLASSES, PALETTE
        cv2.imwrite(str(out / 'mosaic_labels.png'), lab)   # class ids, 255 = no data (for change detection)
        colour = np.zeros_like(rgb)
        has = lab != NODATA
        colour[has] = PALETTE[lab[has]][:, ::-1]           # PALETTE is RGB, OpenCV writes BGR
        cv2.imwrite(str(out / 'mosaic_mask.png'), colour)
        overlay = rgb.copy()
        overlay[has] = cv2.addWeighted(rgb, 0.55, colour, 0.45, 0)[has]
        cv2.imwrite(str(out / 'mosaic_overlay.jpg'), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

        counts = np.bincount(lab[has], minlength=len(CLASSES))
        print('\nArea share of the map:')
        for c, n in zip(CLASSES, counts):
            print(f'  {c:10s} {100 * n / counts.sum():5.1f}%')
        summary['class_share'] = {c: float(n / counts.sum()) for c, n in zip(CLASSES, counts)}

    with open(out / 'stitch_info.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'\nMosaic {rgb.shape[1]}x{rgb.shape[0]} saved to {out} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
