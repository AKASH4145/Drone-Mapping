"""Siamese U-Net for semantic change detection. Same model as notebooks/04_second_change_detection.ipynb,
so weights trained there (second_siamese_best.pth) load directly."""
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 6


class SiameseUNet(nn.Module):
    def __init__(self, n_classes, mid=64, encoder_weights=None):
        super().__init__()
        self.seg = smp.Unet('resnet34', encoder_weights=encoder_weights, classes=n_classes)
        ch = self.seg.encoder.out_channels[1:]              # skip the raw image level
        self.lateral = nn.ModuleList([nn.Sequential(nn.Conv2d(c, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
                                      for c in ch])
        self.change_head = nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                                         nn.Conv2d(mid, 1, 1))

    def decode(self, feats):
        try:
            d = self.seg.decoder(*feats)     # older segmentation-models-pytorch
        except TypeError:
            d = self.seg.decoder(feats)      # newer versions take a list
        return self.seg.segmentation_head(d)

    def forward(self, a, b):
        fa, fb = self.seg.encoder(a), self.seg.encoder(b)
        size = fa[1].shape[-2:]
        # Difference of the two dates' features at every scale, merged at half resolution
        diff = sum(F.interpolate(lat(torch.abs(xa - xb)), size=size, mode='bilinear', align_corners=False)
                   for lat, xa, xb in zip(self.lateral, fa[1:], fb[1:]))
        change = F.interpolate(self.change_head(diff), size=a.shape[-2:], mode='bilinear', align_corners=False)
        return change[:, 0], self.decode(fa), self.decode(fb)


def load_model(weights, device='cpu'):
    model = SiameseUNet(NUM_CLASSES)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


def make_tile_runner(model, device):
    """Returns run_tile(a, b): normalised HxWx3 float arrays in, numpy probabilities out:
    change (H, W) and land cover for each date (6, H, W)."""
    @torch.no_grad()
    def run_tile(a, b):
        ta = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None].to(device)
        tb = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1)))[None].to(device)
        c, sa, sb = model(ta, tb)
        return (torch.sigmoid(c[0]).cpu().numpy(), sa[0].softmax(0).cpu().numpy(), sb[0].softmax(0).cpu().numpy())
    return run_tile
