import cv2
import torch
import os

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer

# -------------------------------------------------------------------------
# 1. Define the neural network architecture for the RealESRGAN model
#    - RRDBNet is a deep residual network widely used for super-resolution.
#    - We configure the architecture to match the pretrained RealESRGAN_x4plus model.
# -------------------------------------------------------------------------
model = RRDBNet(
    num_in_ch=3,        # Number of input image channels (RGB = 3)
    num_out_ch=3,       # Number of output channels
    num_feat=64,        # Base number of feature maps
    num_block=23,       # Number of residual-in-residual dense blocks
    num_grow_ch=32,     # Growth channels for dense blocks
    scale=4             # Upscaling factor (×4 model)
)

# -------------------------------------------------------------------------
# 2. Select computation device
#    - Use GPU (CUDA) if available for faster inference.
#    - Fall back to CPU if no GPU is detected.
# -------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------------------------------------------------
# 3. Initialize the RealESRGAN upsampler
#    - Loads the pretrained model weights (RealESRGAN_x4plus.pth)
#    - Handles automatic tiling for large images (disabled here: tile=0)
#    - Configures padding and precision settings
# -------------------------------------------------------------------------
upsampler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth',  # Path to pretrained weights
    model=model,                          # RRDBNet architecture defined above
    tile=0,                               # Tile size (0 = no tiling)
    tile_pad=10,                          # Padding around tiles
    pre_pad=0,                            # Extra padding before processing
    half=False,                           # Use FP16 if True (only works on CUDA)
    device=device
)

# -------------------------------------------------------------------------
# 4. Load input image
#    - The image must exist in the working directory under the name 'low.jpg'
#    - cv2.imread returns None if the file cannot be found
# -------------------------------------------------------------------------
img = cv2.imread('low.jpg')

# Optional safety check (commented out if not needed):
# if img is None:
#     raise FileNotFoundError("Input image 'low.jpg' not found.")

# -------------------------------------------------------------------------
# 5. Run the super-resolution enhancement
#    - The 'enhance' function returns the upscaled image and face enhancement output (unused)
#    - outscale=4 ensures ×4 upscaling in case internal scaling is adjusted
# -------------------------------------------------------------------------
output, _ = upsampler.enhance(img, outscale=4)

# -------------------------------------------------------------------------
# 6. Save the processed output image
#    - Writes the upscaled result to 'output.png' in the working directory
# -------------------------------------------------------------------------
cv2.imwrite('output.png', output)
print('완료!')  # "Done!" in Korean
