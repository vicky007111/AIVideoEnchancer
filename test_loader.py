import torch
import os
def check_file(path):
    try:
        sd = torch.load(path, map_location='cpu', weights_only=True)
        print(path, "is valid, keys:", sd.keys() if isinstance(sd, dict) else "no keys")
        return True
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return False

for p in os.listdir("models"):
    if p.endswith(".pth"):
        check_file(os.path.join("models", p))
