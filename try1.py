import os
from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure

from src.loftr import LoFTR, default_cfg


def main():
    _default_cfg = deepcopy(default_cfg)
    _default_cfg["coarse"][
        "temp_bug_fix"
    ] = True  # set to False when using the old ckpt
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("weights/indoor_ds_new.ckpt")["state_dict"])
    # matcher = matcher.eval().cuda()
    matcher = matcher.eval()

    # Load example images
    img0_pth = "assets/scannet_sample_images/scene0711_00_frame-001680.jpg"
    img1_pth = "assets/scannet_sample_images/scene0711_00_frame-001995.jpg"
    img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    img0 = torch.from_numpy(img0_raw)[None][None] / 255.0
    img1 = torch.from_numpy(img1_raw)[None][None] / 255.0
    batch = {"image0": img0, "image1": img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch["mkpts0_f"].cpu().numpy()
        mkpts1 = batch["mkpts1_f"].cpu().numpy()
        mconf = batch["mconf"].cpu().numpy()

    print(mkpts0.shape)

    return


if __name__ == "__main__":
    main()
