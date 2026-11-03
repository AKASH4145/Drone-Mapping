"""Turn an ODM orthophoto into a georeferenced land-cover map.

Reads the orthophoto GeoTIFF, runs the U-Net on it, blanks out the empty border,
and saves:
    <name>_labels.tif   class ids with the same coordinates as the orthophoto
                        (255 = no data). Opens in QGIS with class colours.
    <name>_mask.png     colour mask
    <name>_overlay.jpg  mask blended over the photo
    <name>_stats.json   area of each class in square metres

Usage (from the project folder):
    python src/ortho_map.py --input data/odm/sheffield_park_3/odm_orthophoto/odm_orthophoto.tif
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch

from predict import CLASSES, PALETTE, load_model, predict_full

ROOT = Path(__file__).resolve().parents[1]
NODATA = 255


def read_orthophoto(path):
    with rasterio.open(path) as src:
        if src.count < 3:
            raise SystemExit(f'Expected an RGB orthophoto, got {src.count} band(s)')
        rgb = src.read([1, 2, 3])
        valid = src.dataset_mask() > 0   # uses the alpha band or nodata value
        geo = {'crs': src.crs, 'transform': src.transform, 'res': src.res}
    if rgb.dtype != np.uint8:
        raise SystemExit(f'Expected 8-bit colour, got {rgb.dtype}')
    return np.ascontiguousarray(rgb.transpose(1, 2, 0)), valid, geo


def write_labels(path, labels, geo):
    profile = {
        'driver': 'GTiff', 'height': labels.shape[0], 'width': labels.shape[1],
        'count': 1, 'dtype': 'uint8', 'crs': geo['crs'], 'transform': geo['transform'],
        'nodata': NODATA, 'compress': 'deflate', 'tiled': True,
        'blockxsize': 512, 'blockysize': 512,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(labels, 1)
        dst.write_colormap(1, {i: tuple(int(v) for v in c) + (255,) for i, c in enumerate(PALETTE)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='ODM orthophoto GeoTIFF')
    parser.add_argument('--name', help='Name for the outputs (default: ODM project folder name)')
    parser.add_argument('--weights', default=str(ROOT / 'models' / 'unet_r34_best.pth'))
    parser.add_argument('--out', default=str(ROOT / 'outputs' / 'ortho_map'))
    parser.add_argument('--scale', type=float, default=1.0, help='1.0 worked best at 5 cm/pixel')
    parser.add_argument('--overlap', type=float, default=0.5)
    args = parser.parse_args()

    inp = Path(args.input)
    # ODM layout: <project>/odm_orthophoto/odm_orthophoto.tif, so use <project> as the name
    name = args.name or (inp.parents[1].name if inp.parent.name == 'odm_orthophoto' else inp.stem)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rgb, valid, geo = read_orthophoto(inp)
    print(f'{inp.name}: {rgb.shape[1]}x{rgb.shape[0]} px | '
          f'{abs(geo["res"][0]) * 100:.1f} cm/pixel | CRS {geo["crs"]}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.weights, device)
    t0 = time.time()
    labels = predict_full(model, rgb, device, scale=args.scale, overlap=args.overlap)
    labels[~valid] = NODATA
    print(f'Segmented on {device} in {time.time() - t0:.0f}s')

    write_labels(out / f'{name}_labels.tif', labels, geo)

    colour = np.zeros_like(rgb)
    colour[valid] = PALETTE[labels[valid]]
    overlay = rgb.copy()
    overlay[valid] = cv2.addWeighted(rgb, 0.55, colour, 0.45, 0)[valid]
    cv2.imwrite(str(out / f'{name}_mask.png'), cv2.cvtColor(colour, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / f'{name}_overlay.jpg'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Area per class. Only meaningful when the CRS is in metres (ODM uses UTM).
    counts = np.bincount(labels[valid], minlength=len(CLASSES))
    px_area = abs(geo['res'][0] * geo['res'][1])
    in_metres = geo['crs'] is not None and geo['crs'].is_projected
    stats = {'input': str(inp), 'crs': str(geo['crs']), 'pixel_size_m': abs(geo['res'][0]),
             'scale': args.scale, 'overlap': args.overlap, 'classes': {}}
    print(f'\n{"Class":10s} {"Share":>7s} {"Area (m2)":>11s}')
    for c, n in zip(CLASSES, counts):
        share = n / counts.sum()
        area = float(n * px_area) if in_metres else None
        stats['classes'][c] = {'share': float(share), 'area_m2': area}
        print(f'{c:10s} {100 * share:6.1f}% {area if area is not None else float("nan"):11.0f}')
    if in_metres:
        stats['total_area_m2'] = float(counts.sum() * px_area)
        print(f'{"Total":10s} {"":7s} {stats["total_area_m2"]:11.0f}')

    with open(out / f'{name}_stats.json', 'w') as f:
        json.dump(stats, f, indent=1)
    print('\nSaved to', out)


if __name__ == '__main__':
    main()
