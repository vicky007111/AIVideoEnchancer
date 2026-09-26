import sys
import os
sys.path.insert(0, '/var/home/vicky/video enchancer')

import tempfile
import torch
from utils.model_loader import ensure_models_dir, load_realesrgan_weights
from utils.rrdbnet import RRDBNet


def test_ensure_models_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "test_models")
        result_dir = ensure_models_dir(models_dir)
        assert os.path.exists(result_dir)
        assert result_dir == models_dir


def test_rrdbnet_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32).to(device)

    # Create dummy input
    dummy_input = torch.randn(1, 3, 64, 64).to(device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 3, 256, 256), f"Expected shape (1, 3, 256, 256), got {output.shape}"

    print(f"RRDBNet model test passed on {device}")


if __name__ == "__main__":
    test_ensure_models_dir()
    test_rrdbnet_model()
    print("Model loader tests passed")