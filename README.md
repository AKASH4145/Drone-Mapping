# Drone Mapping with Deep Learning

Semantic segmentation of drone imagery, built as the first stage of a drone mapping pipeline. The model labels every pixel of an aerial frame as building, road, tree, low vegetation, car, human or clutter.

![Test frame prediction](assets/test_000000_overlay.jpg)

## Motive

Drones can survey an area much faster than ground teams, but raw footage is hard to use on its own. A map that shows where the buildings, roads, vegetation and vehicles are is far more useful for urban planning, disaster response and surveillance. I wanted to build that pipeline end to end, from training the model to running it on full 4K frames.

## Objective

1. Train a segmentation model on drone images (done)
2. Stitch predictions into a single labelled map of an area (in progress)
3. Compare maps of the same area over time to detect changes (planned)

## Tech stack

Python, PyTorch, segmentation-models-pytorch, OpenCV, NumPy, Matplotlib. Training on Google Colab (T4 GPU), inference runs on CPU.

## Dataset

[UAVid](https://uavid.nl): 4K oblique drone images of urban scenes with 8 classes. I used the Kaggle mirror.

- 200 labelled training images and 70 validation images. The mirror also includes 400 pre-flipped and shifted copies of the training images. I left these out and used random augmentation during training instead.
- The test set has no public labels, so test frames are used only for visual results.
- Each image is downscaled to 1920x1080 and cut into 512x512 tiles, giving 2,400 training tiles and 840 validation tiles.

## Model and training

- U-Net with a ResNet34 encoder pretrained on ImageNet
- Loss: Cross-Entropy + Dice (Dice helps with rare classes like Human)
- AdamW, learning rate 3e-4, cosine schedule, 30 epochs, batch size 8, mixed precision
- Augmentation: random flips, 90 degree rotations, brightness and contrast
- About an hour on a Colab T4

## Results

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

More predictions:

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
    predict.py    run the model on full-size frames
  models/       trained weights (not in the repo)
  data/         dataset (not in the repo)
  outputs/      predictions (not in the repo)
```

## Setup

```
git clone https://github.com/AKASH4145/drone-mapping.git
cd drone-mapping
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The trained weights are not stored in the repo. Train your own with notebook 02, or download them [here](LINK_TO_WEIGHTS) and place `unet_r34_best.pth` in `models/`.

## Usage

```
python src/predict.py --input path/to/images
```

Saves a colour mask and an overlay for each image to `outputs/predictions/`. If a matching `Labels` folder exists next to the images, it also prints the mIoU for each image. Takes about 6 seconds per 4K frame on CPU.

## Future plans

- Stitch predictions from nadir (top-down) drone images into one georeferenced map
- Change detection between two maps of the same area
- Train at full resolution to improve small classes like Human

## Credits

UAVid dataset: Lyu et al., "UAVid: A semantic segmentation dataset for UAV imagery", ISPRS Journal of Photogrammetry and Remote Sensing, 2020.
