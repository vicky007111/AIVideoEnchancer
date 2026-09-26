import sys
sys.path.insert(0, '/var/home/vicky/video enchancer')

import numpy as np
import torch
from utils.tile_processor import TileProcessor
from utils.rrdbnet import RRDBNet

def test_tile_processor():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=6, num_grow_ch=32).to(device)
    processor = TileProcessor(tile_size=64, tile_pad=10, Half_precision=False, device=device)

    # 128x128 image with 3 channels
    dummy_img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)

    output = processor.process(dummy_img, model, scale=4)

    assert output.shape == (512, 512, 3), f"Expected shape (512, 512, 3), got {output.shape}"
    assert output.dtype == np.uint8

if __name__ == '__main__':
    test_tile_processor()
    print("TileProcessor tests passed")