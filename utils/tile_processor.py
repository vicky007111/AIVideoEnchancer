import torch
from torch import nn as nn
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
            model: The model to process with (already on device, already in fp16/fp32)
            scale: Desired upscaling factor

        Returns:
            Output image as numpy array (H*scale, W*scale, C), uint8
        """
        # Determine internal model scale factor
        model_scale = getattr(model, 'scale', getattr(model, 'upscale', scale))

        if self.tile_size == 0:
            output = self._process_full(img, model, model_scale)
        else:
            output = self._process_tiled(img, model, model_scale)

        # If user requested a different scale than native model scale, resize output
        if model_scale != scale:
            import cv2
            h, w = img.shape[:2]
            output = cv2.resize(output, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

        return output

    def _to_tensor(self, img):
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(self.device)
        # ponytail: cast tensor to match model dtype instead of re-casting the model every frame
        if self.half_precision:
            t = t.half()
        return t

    def _to_numpy(self, tensor):
        return tensor.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).mul(255.0).round().byte().cpu().numpy()

    def _process_full(self, img: np.ndarray, model: nn.Module, scale: int):
        img_tensor = self._to_tensor(img)
        with torch.no_grad():
            output = model(img_tensor)
        return self._to_numpy(output)

    def _process_tiled(self, img: np.ndarray, model: nn.Module, scale: int):
        img_tensor = self._to_tensor(img)

        _, channel, height, width = img_tensor.shape
        output_height = height * scale
        output_width = width * scale

        output = torch.zeros((1, channel, output_height, output_width), dtype=img_tensor.dtype, device=self.device)

        tiles_x = int(np.ceil(width / self.tile_size))
        tiles_y = int(np.ceil(height / self.tile_size))

        with torch.no_grad():
            for y in range(tiles_y):
                for x in range(tiles_x):
                    ofs_x = x * self.tile_size
                    ofs_y = y * self.tile_size

                    input_start_x = max(0, ofs_x - self.tile_pad)
                    input_end_x = min(width, ofs_x + self.tile_size + self.tile_pad)
                    input_start_y = max(0, ofs_y - self.tile_pad)
                    input_end_y = min(height, ofs_y + self.tile_size + self.tile_pad)

                    input_tile = img_tensor[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    output_tile = model(input_tile)

                    output_start_x = ofs_x * scale
                    output_end_x = min(ofs_x + self.tile_size, width) * scale
                    output_start_y = ofs_y * scale
                    output_end_y = min(ofs_y + self.tile_size, height) * scale

                    out_tile_start_x = (ofs_x - input_start_x) * scale
                    out_tile_end_x = out_tile_start_x + (output_end_x - output_start_x)
                    out_tile_start_y = (ofs_y - input_start_y) * scale
                    out_tile_end_y = out_tile_start_y + (output_end_y - output_start_y)

                    output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                        :, :, out_tile_start_y:out_tile_end_y, out_tile_start_x:out_tile_end_x
                    ]

        return self._to_numpy(output)
