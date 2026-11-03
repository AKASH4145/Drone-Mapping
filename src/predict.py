"""Run the trained U-Net on full-size drone frames (works on CPU).

Usage (from the project folder):
    python src/predict.py --input data/raw/samples/test/Images
    python src/predict.py --input data/raw/samples/val/Images/000000.png
    python src/predict.py --input path/to/odm_orthophoto.tif --scale 1.0 --overlap 0.5
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

ROOT = Path(__file__).resolve().parents[1]

CLASSES = ['Clutter', 'Building', 'Road', 'Tree', 'LowVeg', 'MovingCar', 'StaticCar', 'Human']
PALETTE = np.array([(0, 0, 0), (128, 0, 0), (128, 64, 128), (0, 128, 0),
                    (128, 128, 0), (64, 0, 128), (192, 0, 192), (64, 64, 0)], np.uint8)  # RGB
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def starts(total, tile, stride=None):
    stride = stride or tile
    s = list(range(0, total - tile + 1, stride))
    if s[-1] != total - tile:
        s.append(total - tile)
    return s


def load_model(weights, device):
    model = smp.Unet('resnet34', encoder_weights=None, classes=len(CLASSES))
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


@torch.no_grad()
def predict_full(model, img_rgb, device, scale=0.5, tile=512, overlap=0.0):
    """Downscale, predict tile by tile, average overlaps, upscale back to full size.

    overlap=0.5 moves tiles by half a tile, so every pixel is predicted about 4 times
    and averaged. This removes blocky seams at tile borders but takes ~4x longer.
    """
    small = cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    H, W = small.shape[:2]
    norm = (small.astype(np.float32) / 255.0 - MEAN) / STD
    probs = torch.zeros(len(CLASSES), H, W)
    count = torch.zeros(1, H, W)
    stride = max(1, int(tile * (1 - overlap)))
    for y in starts(H, tile, stride):
        for x in starts(W, tile, stride):
            t = torch.from_numpy(np.ascontiguousarray(norm[y:y + tile, x:x + tile].transpose(2, 0, 1)))
            out = model(t.unsqueeze(0).to(device))[0].float().softmax(0).cpu()
            probs[:, y:y + tile, x:x + tile] += out
            count[:, y:y + tile, x:x + tile] += 1
    pred = (probs / count).argmax(0).numpy().astype(np.uint8)
    return cv2.resize(pred, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)


def rgb_label_to_id(lab_rgb):
    out = np.full(lab_rgb.shape[:2], 255, np.uint8)
    for cid, c in enumerate(PALETTE):
        out[np.all(lab_rgb == c, axis=-1)] = cid
    return out


def image_miou(pred, gt):
    """Mean IoU over classes present in either prediction or ground truth."""
    ious = []
    valid = gt != 255
    for c in range(len(CLASSES)):
        p, g = (pred == c) & valid, (gt == c) & valid
        union = (p | g).sum()
        if union > 0:
            ious.append((p & g).sum() / union)
    return float(np.mean(ious))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Image file or folder of images')
    parser.add_argument('--weights', default=str(ROOT / 'models' / 'unet_r34_best.pth'))
    parser.add_argument('--out', default=str(ROOT / 'outputs' / 'predictions'))
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Downscale before predicting. 0.5 matches training on UAVid 4K frames')
    parser.add_argument('--overlap', type=float, default=0.0,
                        help='Tile overlap from 0 to 0.75. 0.5 removes tile seams, ~4x slower')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inp = Path(args.input)
    if inp.is_dir():
        paths = sorted(p for p in inp.rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
                       and p.parent.name != 'Labels')
    else:
        paths = [inp]
    if not paths:
        print('No images found in', inp)
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.weights, device)
    print(f'Running on {device} | {len(paths)} image(s)')

    scores = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print('Could not read', p)
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        pred = predict_full(model, rgb, device, scale=args.scale, overlap=args.overlap)
        dt = time.time() - t0

        colour = PALETTE[pred]
        overlay = cv2.addWeighted(rgb, 0.55, colour, 0.45, 0)
        stem = f'{p.parent.parent.name}_{p.stem}' if p.parent.name == 'Images' else p.stem
        cv2.imwrite(str(out_dir / f'{stem}_overlay.jpg'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f'{stem}_mask.png'), cv2.cvtColor(colour, cv2.COLOR_RGB2BGR))

        msg = f'{p.name}: {dt:.1f}s'
        label_path = p.parent.parent / 'Labels' / p.name
        if label_path.exists():
            gt = rgb_label_to_id(cv2.cvtColor(cv2.imread(str(label_path)), cv2.COLOR_BGR2RGB))
            score = image_miou(pred, gt)
            scores.append(score)
            msg += f' | mIoU {score:.3f}'
        print(msg)

    if scores:
        print(f'\nAverage per-image mIoU on {len(scores)} labelled image(s): {np.mean(scores):.3f}')
    print('Saved to', out_dir)


if __name__ == '__main__':
    main()