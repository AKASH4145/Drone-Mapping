# Drone Mapping with Deep Learning

A drone mapping pipeline: a segmentation model labels every pixel of an aerial image as building, road, tree, low vegetation, car, human or clutter, and those labels are turned into a georeferenced land-cover map of a whole area.

![Test frame prediction](assets/test_000000_overlay.jpg)

## Motive

Drones can survey an area much faster than ground teams, but raw footage is hard to use on its own. A map that shows where the buildings, roads, vegetation and vehicles are is far more useful for urban planning, disaster response and surveillance. I wanted to build that pipeline end to end, from training the model to producing a map with real-world coordinates.

## Objective

1. Train a segmentation model on drone images (done)
2. Turn predictions into a single labelled map of an area (done, first version)
3. Compare maps of the same area over time to detect changes (planned)

## Tech stack

Python, PyTorch, segmentation-models-pytorch, OpenCV, NumPy, rasterio, OpenDroneMap. Training on Google Colab (T4 GPU), everything else runs on CPU.

## Dataset

[UAVid](https://uavid.nl): 4K oblique drone images of urban scenes with 8 classes. I used the Kaggle mirror.

- 200 labelled training images and 70 validation images. The mirror also includes 400 pre-flipped and shifted copies of the training images. I left these out and used random augmentation during training instead.
- The test set has no public labels, so test frames are used only for visual results.
- Each image is downscaled to 1920x1080 and cut into 512x512 tiles, giving 2,400 training tiles and 840 validation tiles.

For mapping I used the Sheffield Park 3 sample set from [ODMdata](https://github.com/OpenDroneMap/ODMdata): 32 top-down photos of a park with GPS tags.

## Model and training

- U-Net with a ResNet34 encoder pretrained on ImageNet
- Loss: Cross-Entropy + Dice (Dice helps with rare classes like Human)
- AdamW, learning rate 3e-4, cosine schedule, 30 epochs, batch size 8, mixed precision
- Augmentation: random flips, 90 degree rotations, brightness and contrast
- About an hour on a Colab T4

## Results

### Segmentation

Validation set (70 images, one video sequence), best epoch 27:

| Class | IoU |
|---|---|
| Building | 0.909 |
| Road | 0.787 |
| Tree | 0.769 |
| MovingCar | 0.761 |
| StaticCar | 0.682 |
| LowVeg | 0.679 |
| Clutter | 0.652 |
| Human | 0.399 |
| **Mean IoU** | **0.705** |

Pixel accuracy: 0.873

![Validation comparison](assets/val_comparison.png)

Notes on these numbers:

- The validation set comes from a single sequence, and the best epoch was picked on it, so the score is slightly optimistic. It is not comparable to the official UAVid test benchmark.
- Human is the weakest class. Humans make up only 0.12% of training pixels and become very small after downscaling.
- The most common error is confusing low vegetation with trees and clutter, which look similar from above.

### Land-cover map

The 32 photos were turned into an orthophoto with OpenDroneMap (about 8 minutes on CPU, 5 cm per pixel, UTM zone 17N). The model was then run on the orthophoto with 50% tile overlap, which removed the blocky seams that appeared at tile borders without overlap.

![Land-cover map of Sheffield Park](assets/sheffield_park_3_overlay.jpg)

Estimated area of each class (about 1.5 minutes on CPU):

| Class | Share | Area (m²) |
|---|---|---|
| Tree | 43.4% | 12,285 |
| Clutter | 25.3% | 7,172 |
| LowVeg | 24.6% | 6,969 |
| Building | 5.3% | 1,488 |
| Road | 1.3% | 380 |
| Cars and Human | 0.0% | 4 |
| **Total** | | **28,298** |

There is no ground truth for this area, so these are the model's estimates, not measured accuracy. Looking at the map:

- Trees and roofs are labelled well.
- Building is overestimated. The houses cover a few hundred square metres, but the baseball diamond is also labelled Building, because UAVid has no class for bare ground.
- Most dry brown grass ends up as Clutter. UAVid grass is green, so the model does not recognise brown grass as low vegetation.
- No cars were found. UAVid shows cars at an angle, and they look different from straight above.

These come from training on oblique city footage and testing on top-down park photos. Fine-tuning on a top-down dataset would fix most of them.

More predictions on UAVid:

![Test frame 400](assets/test_000400_overlay.jpg)
![Validation frame](assets/val_000000_overlay.jpg)

## Project structure

```
drone-mapping/
  assets/       images used in this README
  notebooks/
    01_data_setup.ipynb           download, inspect and tile UAVid (Colab)
    02_train_segmentation.ipynb   training and evaluation (Colab)
  src/
    predict.py      run the model on full-size frames
    ortho_map.py    land-cover map from an ODM orthophoto, with coordinates and areas
    stitch.py       lightweight stitching of nadir photos without ODM
  models/       trained weights (not in the repo)
  data/         datasets and ODM projects (not in the repo)
  outputs/      predictions and maps (not in the repo)
```

## Setup

```
git clone https://github.com/AKASH4145/drone-mapping.git
cd drone-mapping
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The trained weights are not stored in the repo. Train your own with notebook 02 and place `unet_r34_best.pth` in `models/`.

For maps, install [OpenDroneMap](https://github.com/OpenDroneMap/ODM) (the native Windows installer works).

## Usage

### Predict on images

```
python src/predict.py --input path/to/images
```

Saves a colour mask and an overlay for each image to `outputs/predictions/`. If a matching `Labels` folder exists next to the images, it also prints the mIoU for each image. Takes about 6 seconds per 4K frame on CPU. Add `--overlap 0.5` to remove tile seams (about 4x slower).

### Land-cover map

1. Put the photos in `data/odm/<project>/images/` and run ODM from the ODM Console:
   ```
   run "full\path\to\data\odm\<project>" --fast-orthophoto --orthophoto-resolution 5
   ```
2. Segment the orthophoto:
   ```
   python src/ortho_map.py --input data/odm/<project>/odm_orthophoto/odm_orthophoto.tif
   ```

Outputs in `outputs/ortho_map/`: a label GeoTIFF with the same coordinates as the orthophoto (opens in QGIS with class colours), a colour mask, an overlay and a JSON file with the area of each class.

### Stitching without ODM

```
python src/stitch.py --input data/nadir/sheffield_park_3
```

A simpler alternative I wrote to understand how stitching works. It matches SIFT features between every pair of photos, fits a similarity transform to each pair with RANSAC, then solves all transforms together with one least-squares problem so errors do not build up. Predictions are warped with nearest-neighbour interpolation so class ids are never blended. On Sheffield Park 3 the median alignment error is 2.3 px (95th percentile 7.1 px). It assumes flat ground and has no real-world coordinates, which is why the main pipeline uses ODM.

## Future plans

- Change detection between two maps of the same area captured on different dates
- Fine-tune the model on top-down drone images
- Pad image borders before predicting, so edge pixels are not mislabelled as Building

## Credits

UAVid dataset: Lyu et al., "UAVid: A semantic segmentation dataset for UAV imagery", ISPRS Journal of Photogrammetry and Remote Sensing, 2020.

Sheffield Park 3 images from the OpenDroneMap sample datasets.

## Author

Akash G S | [GitHub](https://github.com/AKASH4145) | [LinkedIn](https://www.linkedin.com/in/akash-g-s-32aa15394/)