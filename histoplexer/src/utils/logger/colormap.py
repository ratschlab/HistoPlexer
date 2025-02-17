import matplotlib.pyplot as plt
import torch

def colormap(gs_image: torch.Tensor, 
             vmin: list = None, 
             vmax: list = None) -> torch.Tensor:
    """
    Applies a 'viridis' colormap to a batch of grayscale images, with optional per-image normalization ranges.

    Args:
        gs_image (torch.Tensor): A tensor of grayscale images with shape [B, 1, H, W].
        vmin (list, optional): A list of minimum values for normalization, one per image in the batch. 
                                    Defaults to all zeros if not provided.
        vmax (list, optional): A list of maximum values for normalization, one per image in the batch. 
                                    Defaults to all ones if not provided.

    Returns:
        torch.Tensor: A tensor of colorized images with shape [B, 3, H, W].
    """
    # Use matplotlib's 'viridis' colormap
    cmap = plt.cm.viridis
    
    # Set default vmin and vmax values if not provided
    batch_size = gs_image.shape[0]
    if vmin is None:
        vmin = [0] * batch_size
    if vmax is None:
        vmax = [1] * batch_size

    # Normalize and apply colormap to each image in the batch
    rgb_img = []
    for i, image in enumerate(gs_image):
        # Use different variable names for the loop variables
        vmin_i = vmin[i]
        vmax_i = vmax[i]

        # Clamp values and normalize to the range [vmin, vmax]
        image = torch.clamp(image, vmin_i, vmax_i)
        image = (image - vmin_i) / (vmax_i - vmin_i)

        # Apply colormap
        image_np = image.squeeze(0).cpu().numpy()  # Remove channel dimension and convert to numpy
        colored_image = cmap(image_np)[..., :3]  # Apply colormap and remove alpha channel

        # Convert back to tensor and add batch dimension
        colored_image_tensor = torch.from_numpy(colored_image).float().unsqueeze(0).permute(0, 3, 1, 2)
        rgb_img.append(colored_image_tensor)

    # Combine into a single tensor
    rgb_img = torch.cat(rgb_img, dim=0)

    return rgb_img
