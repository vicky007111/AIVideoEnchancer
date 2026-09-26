import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

class TileProcessor:
    """Handle tiling for VRAM-efficient processing of large images."""

    def __init__(self, tile_size=256, tile_pad=10, Half_precision=True, device='cuda'):
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.half_precision = Half_precision
        self.device = device

    def process(self, img: np.ndarray, model: nn.Module, scale: int = 4):
        """
        Process an image tile by tile.

        Args:
            img: Input image as numpy array (H, W, C), uint8 (0-255)
            model: The model to process with
            scale: Upscaling factor

        Returns:
            Output image as numpy array (H*scale, W*scale, C), uint8
        """
        if self.tile_size == 0:
            return self._process_full(img, model, scale)

        return self._process_tiled(img, model, scale)

    def _process_full(self, img: np.ndarray, model: nn.Module, scale: int):
        model.eval()
        # Shape: (1, C, H, W), float in [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(self.device)

        if self.half_precision:
            img_tensor = img_tensor.half()
            model = model.half()
        else:
            img_tensor = img_tensor.float()
            model = model.float()

        with torch.no_grad():
            output = model(img_tensor)

        output_np = output.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).mul(255.0).round().byte().cpu().numpy()
        return output_np

    def _process_tiled(self, img: np.ndarray, model: nn.Module, scale: int):
        model.eval()
        # Shape: (1, C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(self.device)

        if self.half_precision:
            img_tensor = img_tensor.half()
            model = model.half()
        else:
            img_tensor = img_tensor.float()
            model = model.float()

        batch, channel, height, width = img_tensor.shape
        output_height = height * scale
        output_width = width * scale
        output_shape = (batch, channel, output_height, output_width)

        output = torch.zeros(output_shape, dtype=img_tensor.dtype, device=self.device)

        tiles_x = int(np.ceil(width / self.tile_size))
        tiles_y = int(np.ceil(height / self.tile_size))

        with torch.no_grad():
            for y in range(tiles_y):
                for x in range(tiles_x):
                    # Tile coordinates with padding
                    ofs_x = x * self.tile_size
                    ofs_y = y * self.tile_size

                    # Input tile range
                    input_start_x = max(0, ofs_x - self.tile_pad)
                    input_end_x = min(width, ofs_x + self.tile_size + self.tile_pad)
                    input_start_y = max(0, ofs_y - self.tile_pad)
                    input_end_y = min(height, ofs_y + self.tile_size + self.tile_pad)

                    # Crop input tile
                    input_tile = img_tensor[:, :, input_start_y:input_end_y, input_start_x:input_end_x]

                    # Process tile
                    output_tile = model(input_tile)

                    # Output crop coordinates
                    output_start_x = ofs_x * scale
                    output_end_x = min(ofs_x + self.tile_size, width) * scale
                    output_start_y = ofs_y * scale
                    output_end_y = min(ofs_y + self.tile_size, height) * scale

                    # Offset within output tile
                    out_tile_start_x = (ofs_x - input_start_x) * scale
                    out_tile_end_x = out_tile_start_x + (output_end_x - output_start_x)
                    out_tile_start_y = (ofs_y - input_start_y) * scale
                    out_tile_end_y = out_tile_start_y + (output_end_y - output_start_y)

                    # Put into output tensor
                    output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                        :, :, out_tile_start_y:out_tile_end_y, out_tile_start_x:out_tile_end_x
                    ]

        output_np = output.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).mul(255.0).round().byte().cpu().numpy()
        return output_np