import torch
import torch.nn.functional as F


def extract_subpixel_patch(feature_map, subpixel_locations, patch_size):
    """
    Extract patches from feature_map at sub-pixel locations.

    Args:
    - feature_map: [B, C, H, W] tensor representing the feature map.
    - subpixel_locations: [B, 2] tensor representing the (x, y) sub-pixel locations to extract from.
    - patch_size: Integer representing the size of the patch (e.g., 3 for a 3x3 patch).

    Returns:
    - patches: [B, C, patch_size, patch_size] tensor of extracted patches.
    """
    B, C, H, W = feature_map.size()

    # Normalize grid coordinates to be in the range [-1, 1]
    x = (subpixel_locations[:, 0] / (W - 1)) * 2 - 1  # Normalize x to [-1, 1]
    y = (subpixel_locations[:, 1] / (H - 1)) * 2 - 1  # Normalize y to [-1, 1]

    # Generate a relative grid of coordinates in the range [-1, 1]
    lin_coords = torch.linspace(-1, 1, patch_size, device=feature_map.device)
    grid_y, grid_x = torch.meshgrid(lin_coords, lin_coords)

    # Stack and expand grid to match the batch size
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [patch_size, patch_size, 2]
    base_grid = base_grid.unsqueeze(0).repeat(
        B, 1, 1, 1
    )  # [B, patch_size, patch_size, 2]

    # Add the subpixel locations (expand to match the grid size)
    grid = base_grid + torch.stack([x, y], dim=-1).view(
        B, 1, 1, 2
    )  # [B, patch_size, patch_size, 2]

    # grid_sample expects grid to be in the range [-1, 1], so the normalization is correct
    # Use grid_sample to extract patches
    patches = F.grid_sample(
        feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    return patches


def main():
    # Example usage:
    B, C, H, W = 1, 3, 10, 10  # Example feature map
    feature_map = torch.rand(B, C, H, W)

    # Sub-pixel locations to extract patches from (batch size 1, 2D locations)
    subpixel_locations = torch.tensor([[5.2, 5.8]])

    # Extract 3x3 patches from the feature map
    patch_size = 3
    patches = extract_subpixel_patch(feature_map, subpixel_locations, patch_size)

    print(patches.size())  # Output: [1, 3, 3, 3] (1 patch, 3 channels, 3x3 patch size)

if __name__ == '__main__':
    main()