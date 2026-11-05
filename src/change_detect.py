"""Find land-cover change between two images of the same area, using the Siamese U-Net trained on SECOND.

For every pair it saves a picture (before, after, predicted classes before and after, plus the true labels
if they exist) and a JSON file with how much changed and the most common changes. If label1 / label2 folders
exist, it also scores the predictions with the SECOND metrics (SeK, Fscd, mIoU, change F1).

Usage (from the project folder):
    python src/change_detect.py --before path/to/before.png --after path/to/after.png
    python src/change_detect.py --pairs data/second/SECOND_total_test --limit 50
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CLASSES = ['Water', 'Ground', 'LowVeg', 'Tree', 'Building', 'Playground']
NAMES = ['Unchanged'] + CLASSES
SECOND_RGB = [(255, 255, 255), (0, 0, 255), (128, 128, 128), (0, 128, 0), (0, 255, 0), (128, 0, 0), (255, 0, 0)]
PALETTE = np.array(SECOND_RGB, np.uint8)       # 0 = unchanged (white), 1..6 = classes
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IGNORE = 255
IMG_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


# ---------- reading ----------

def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_label(path):
    """SECOND label: 0 = unchanged, 1..6 = class, 255 = unknown colour."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        rgb = cv2.cvtColor(m[..., :3], cv2.COLOR_BGR2RGB)
        out = np.full(rgb.shape[:2], IGNORE, np.uint8)
        for i, c in enumerate(SECOND_RGB):
            out[np.all(rgb == c, axis=-1)] = i
        return out
    out = m.astype(np.uint8).copy()
    out[out > len(CLASSES)] = IGNORE
    return out


def find_pairs(folder):
    """Every im1 folder under `folder`, paired with im2 (and label1 / label2 if they exist) by file name."""
    pairs = []
    for im1 in sorted(Path(folder).rglob('im1')):
        if not im1.is_dir():
            continue
        im2, l1, l2 = im1.parent / 'im2', im1.parent / 'label1', im1.parent / 'label2'
        for a in sorted(p for p in im1.iterdir() if p.suffix.lower() in IMG_EXT):
            b = im2 / a.name
            if b.exists():
                pairs.append({'name': f'{im1.parent.name}_{a.stem}', 'before': a, 'after': b,
                              'lab1': l1 / a.name if (l1 / a.name).exists() else None,
                              'lab2': l2 / a.name if (l2 / a.name).exists() else None})
    return pairs


# ---------- prediction ----------

def starts(total, tile, stride):
    s = list(range(0, max(total - tile, 0) + 1, stride))
    if s[-1] != max(total - tile, 0):
        s.append(total - tile)
    return s


def predict_pair(run_tile, a, b, tile=512, overlap=0.25, scale=1.0):
    """Sliding-window prediction over two aligned images of any size. Overlapping tiles are averaged.
    Returns change probability (H, W) and land-cover class 0..5 for each date, at the input size."""
    if a.shape != b.shape:
        raise ValueError(f'Before and after must be the same size, got {a.shape} and {b.shape}')
    H0, W0 = a.shape[:2]
    if scale != 1.0:
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = a.shape[:2]
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        a = cv2.copyMakeBorder(a, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        b = cv2.copyMakeBorder(b, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    H, W = a.shape[:2]
    na = (a.astype(np.float32) / 255.0 - MEAN) / STD
    nb = (b.astype(np.float32) / 255.0 - MEAN) / STD

    k = len(CLASSES)
    chg = np.zeros((H, W), np.float32)
    pa = np.zeros((k, H, W), np.float32)
    pb = np.zeros((k, H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    stride = max(1, int(tile * (1 - overlap)))
    for y in starts(H, tile, stride):
        for x in starts(W, tile, stride):
            c, sa, sb = run_tile(na[y:y + tile, x:x + tile], nb[y:y + tile, x:x + tile])
            chg[y:y + tile, x:x + tile] += c
            pa[:, y:y + tile, x:x + tile] += sa
            pb[:, y:y + tile, x:x + tile] += sb
            cnt[y:y + tile, x:x + tile] += 1
    chg = (chg / cnt)[:h, :w]
    ca, cb = pa.argmax(0)[:h, :w].astype(np.uint8), pb.argmax(0)[:h, :w].astype(np.uint8)
    if scale != 1.0:
        chg = cv2.resize(chg, (W0, H0), interpolation=cv2.INTER_LINEAR)
        ca = cv2.resize(ca, (W0, H0), interpolation=cv2.INTER_NEAREST)
        cb = cv2.resize(cb, (W0, H0), interpolation=cv2.INTER_NEAREST)
    return chg, ca, cb


def scd_maps(chg, ca, cb, thr=0.5, drop_same=False):
    """0 where unchanged, otherwise land-cover class + 1 (same coding as SECOND labels).
    drop_same: a change always goes from one class to a different one, so a pixel called changed
    but given the same class on both dates is treated as unchanged."""
    changed = chg > thr
    if drop_same:
        changed &= ca != cb
    return np.where(changed, ca + 1, 0).astype(np.uint8), np.where(changed, cb + 1, 0).astype(np.uint8)


# ---------- scoring (same maths as the notebook and the SECOND authors' code) ----------

def add_to_hist(hist, pred, label):
    """pred and label coded 0 = unchanged, 1..6 = class; label 255 is skipped."""
    K = hist.shape[0]
    m = label <= len(CLASSES)
    hist += np.bincount(pred[m].astype(np.int64) * K + label[m], minlength=K * K).reshape(K, K)


def cal_kappa(hist):
    if hist.sum() == 0:
        return 0.0
    po = np.diag(hist).sum() / hist.sum()
    pe = (hist.sum(1) * hist.sum(0)).sum() / hist.sum() ** 2
    return 0.0 if pe == 1 else (po - pe) / (1 - pe)


def scd_scores(hist):
    c2 = np.array([[hist[0, 0], hist[0, 1:].sum()], [hist[1:, 0].sum(), hist[1:, 1:].sum()]], float)
    iu = np.diag(c2) / np.maximum(c2.sum(1) + c2.sum(0) - np.diag(c2), 1)
    h0 = hist.astype(float).copy()
    h0[0, 0] = 0
    sek = cal_kappa(h0) * np.exp(iu[1]) / np.e
    pred_ch, lab_ch = hist[1:, :].sum(), hist[:, 1:].sum()
    sc_tp = np.diag(hist[1:, 1:]).sum()
    p, r = sc_tp / max(pred_ch, 1), sc_tp / max(lab_ch, 1)
    tp, fp, fn = c2[1, 1], c2[1, 0], c2[0, 1]
    bp, br = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {'SeK': float(sek), 'Fscd': float(2 * p * r / max(p + r, 1e-9)), 'mIoU': float(iu.mean()),
            'OA': float(np.diag(hist).sum() / max(hist.sum(), 1)),
            'change_precision': float(bp), 'change_recall': float(br),
            'change_F1': float(2 * bp * br / max(bp + br, 1e-9)), 'change_IoU': float(iu[1])}


# ---------- outputs ----------

def panel(images, titles, height=320):
    tiles = []
    for im, t in zip(images, titles):
        s = height / im.shape[0]
        im = cv2.resize(im, (max(1, round(im.shape[1] * s)), height), interpolation=cv2.INTER_NEAREST)
        bar = np.full((28, im.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, im]))
        tiles.append(np.full((height + 28, 6, 3), 255, np.uint8))
    return np.hstack(tiles[:-1])


def transitions(pa, pb, pixel_m=None, top=5):
    m = pa > 0
    K = len(NAMES)
    t = np.bincount(pa[m].astype(np.int64) * K + pb[m], minlength=K * K).reshape(K, K)
    order = np.dstack(np.unravel_index(np.argsort(-t, axis=None), t.shape))[0]
    out = []
    for i, j in order[:top]:
        if t[i, j] == 0:
            break
        item = {'from': NAMES[i], 'to': NAMES[j], 'pixels': int(t[i, j])}
        if pixel_m:
            item['area_m2'] = round(float(t[i, j]) * pixel_m ** 2, 1)
        out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', help='Image of the area at the first date')
    parser.add_argument('--after', help='Image of the same area at the second date, aligned with --before')
    parser.add_argument('--pairs', help='Folder containing im1 / im2 (and optional label1 / label2) folders')
    parser.add_argument('--limit', type=int, default=0, help='Only process the first N pairs (0 = all)')
    parser.add_argument('--weights', default=str(ROOT / 'models' / 'second_siamese_best.pth'))
    parser.add_argument('--out', default=str(ROOT / 'outputs' / 'change'))
    parser.add_argument('--scale', type=float, default=1.0, help='Resize before predicting (SECOND tiles are 512 px)')
    parser.add_argument('--overlap', type=float, default=0.25, help='Tile overlap for images larger than 512 px')
    parser.add_argument('--threshold', type=float, default=0.5, help='Change probability needed to call a pixel changed')
    parser.add_argument('--drop-same-class', action='store_true',
                        help='Treat pixels predicted as changed but with the same class on both dates as unchanged')
    parser.add_argument('--pixel-size', type=float, help='Ground size of one pixel in metres, to report areas in m2')
    parser.add_argument('--save-images', type=int, default=20, help='Save result pictures for the first N pairs')
    args = parser.parse_args()

    if args.pairs:
        pairs = find_pairs(args.pairs)
    elif args.before and args.after:
        pairs = [{'name': Path(args.before).stem, 'before': Path(args.before), 'after': Path(args.after),
                  'lab1': None, 'lab2': None}]
    else:
        parser.error('Give --before and --after, or --pairs')
    if not pairs:
        raise SystemExit(f'No image pairs found in {args.pairs} (expected im1 and im2 folders)')
    if args.limit:
        pairs = pairs[:args.limit]

    import torch                                      # loaded here so the rest can be tested without torch
    from siamese import load_model, make_tile_runner
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_tile = make_tile_runner(load_model(args.weights, device), device)
    print(f'Running on {device} | {len(pairs)} pair(s)')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hist = np.zeros((len(NAMES), len(NAMES)), np.int64)
    summary = []
    t0 = time.time()
    for i, p in enumerate(pairs):
        a, b = read_image(p['before']), read_image(p['after'])
        chg, ca, cb = predict_pair(run_tile, a, b, overlap=args.overlap, scale=args.scale)
        pa, pb = scd_maps(chg, ca, cb, args.threshold, args.drop_same_class)
        info = {'name': p['name'], 'changed_share': round(float((pa > 0).mean()), 4),
                'top_changes': transitions(pa, pb, args.pixel_size)}

        labelled = p['lab1'] is not None and p['lab2'] is not None
        if labelled:
            l1, l2 = read_label(p['lab1']), read_label(p['lab2'])
            add_to_hist(hist, pa, l1)
            add_to_hist(hist, pb, l2)

        if i < args.save_images:
            imgs = [a, b, PALETTE[pa], PALETTE[pb]]
            titles = ['before', 'after', 'predicted before', 'predicted after']
            if labelled:
                imgs += [PALETTE[l1 % 7], PALETTE[l2 % 7]]
                titles += ['true before', 'true after']
            cv2.imwrite(str(out / f'{p["name"]}_result.jpg'),
                        cv2.cvtColor(panel(imgs, titles), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        summary.append(info)

        if i == 0:
            print(f'First pair took {time.time() - t0:.1f}s')
        if len(pairs) == 1:
            print(f'Changed: {100 * info["changed_share"]:.1f}% of pixels')
            for t in info['top_changes']:
                area = f' ({t["area_m2"]} m2)' if 'area_m2' in t else ''
                print(f'  {t["from"]:10s} -> {t["to"]:10s} {t["pixels"]:,} px{area}')

    result = {'pairs': len(pairs), 'threshold': args.threshold, 'drop_same_class': args.drop_same_class,
              'per_pair': summary}
    if hist.sum():
        scores = scd_scores(hist)
        result['scores'] = scores
        print(f'\nScores on {len(pairs)} labelled pair(s):')
        for k, v in scores.items():
            print(f'  {k:17s} {v:.3f}')
    with open(out / 'change_results.json', 'w') as f:
        json.dump(result, f, indent=1)
    print(f'\nSaved to {out} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
