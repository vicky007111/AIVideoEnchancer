import os
import torch
import requests
from tqdm import tqdm
from utils.rrdbnet import RRDBNet

def download_file(url, dest_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Error downloading file")

def ensure_models_dir(models_dir="./models"):
    """Ensure the models directory exists."""
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def load_realesrgan_weights(model, model_path, device='cuda'):
    """Load weights into Real-ESRGAN model."""
    loadnet = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    elif 'params' in loadnet:
        keyname = 'params'
    else:
        keyname = None

    if keyname is not None:
        model.load_state_dict(loadnet[keyname], strict=True)
    else:
        model.load_state_dict(loadnet, strict=True)

    model.eval()
    return model.to(device)

def get_realesrgan_model(scale=4, model_name='RealESRGAN_x4plus', models_dir="./models", device='cuda'):
    """Get or download Real-ESRGAN model."""
    ensure_models_dir(models_dir)
    model_path = os.path.join(models_dir, f"{model_name}.pth")

    if not os.path.exists(model_path):
        print(f"Downloading Real-ESRGAN model {model_name}...")
        url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth"
        download_file(url, model_path)

    # Define the model architecture
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=64, num_block=23, num_grow_ch=32)
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=64, num_block=6, num_grow_ch=32)
    elif model_name == 'realesr-animevideov3':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=6, num_grow_ch=32)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=64, num_block=23, num_grow_ch=32)

    return load_realesrgan_weights(model, model_path, device=device)