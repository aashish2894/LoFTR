import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat




class CustomFinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config["fine_concat_coarse_feat"]
        self.W = self.config["fine_window_size"]

        d_model_f = self.config["fine"]["d_model"]
        self.d_model_f = d_model_f

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, data):
        # input feat_f0 -> [N, C, H, W] same shape for feat_f1
        # data variable should have keypoint locations
        # data[refKpts] --> [M,2]
        # data[coarseObsKpts] --> [M,2] these would be coarse predictions coming from say IMU
        # returns feature maps with shape [M, W*W, C] -> W is the window size, C is the number of channels (same as input)

        # TODO
        # 1. sample WxW feature maps at given locations
        # support drawing sample at both integer and floating locations. for floating use BiLinear interpolation
        # 2. Write a function to do this and verify that the sample drawn is correct
        # 3. Write unit test to verify that the sample being drawn is the correct sample
        # 4. This might become a very slow operation, so later think of ways of speeding up this approach


        W = self.W
        # stride = data["hw0_f"][0] // data["hw0_c"][0]
        stride = W
        data.update({"W": W})
        assert data["refKpts"].shape[0] != 0
        assert data["refKpts"].shape[0] == data["coarseObsKpts"].shape[0]

        # 1. Extract WxW patches at refKpts in feat_f0, and at coarseObsKpts in feat_f1

        # 2. unfold(crop) all local windows
        feat_f0_unfold = F.unfold(
            feat_f0, kernel_size=(W, W), stride=stride, padding=W // 2
        )
        feat_f0_unfold = rearrange(feat_f0_unfold, "n (c ww) l -> n l ww c", ww=W**2)
        feat_f1_unfold = F.unfold(
            feat_f1, kernel_size=(W, W), stride=stride, padding=W // 2
        )
        feat_f1_unfold = rearrange(feat_f1_unfold, "n (c ww) l -> n l ww c", ww=W**2)

        return feat_f0_unfold, feat_f1_unfold

    def extract_subpixel_patches(self, feature_map, subpixel_locations: list):
        B, C, H, W = 1, 3, 100, 100  # Example feature map
        feature_map = torch.rand(B, C, H, W)
        
        # Sub-pixel locations to extract patches from (batch size 1, 2D locations)
        subpixel_locations = [[15, 20], [7, 9], [5.2, 5.8], [7.1, 21.1]]
        subpixel_locations_tensor = torch.tensor([[15, 20], [7, 9], [5.2, 5.8], [7.1, 21.1]])
        
        # Extract 5x5 patches from the feature map
        patch_size = 5
        
        B, C, H, W = feature_map.size()
        
        # Normalize grid coordinates to be in the range [-1, 1]
        x = (subpixel_locations_tensor[:, 0] / (W - 1)) * 2 - 1  # Normalize x to [-1, 1]
        y = (subpixel_locations_tensor[:, 1] / (H - 1)) * 2 - 1  # Normalize y to [-1, 1]
        grid = torch.stack([x, y], dim=-1)
        grid = grid.unsqueeze(0)
        numKpts = grid.shape[1]
        assert numKpts % 2 == 0
        grid = rearrange(grid, "n (v b) c -> n v b c", v=numKpts // 2, b=numKpts // 2)
        patches = F.grid_sample(
            feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        # this is how we need to maodify patches
        temp1 = torch.rand(1, 128, 2 * 5, 2 * 5)
        temp2 = F.unfold(temp1, kernel_size=(5, 5), stride=5)
        temp3 = rearrange(temp2, "n (c ww) l -> n l ww c", ww=5**2)

        return
