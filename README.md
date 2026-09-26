# AIVideoEnhancer 🚀

A lightweight, local-first AI video enhancement tool designed to drastically improve video resolution, reduce blur, eliminate compression artifacts, and upscale footage using state-of-the-art deep learning models (Real-ESRGAN).

Optimized for consumer GPUs with **6GB VRAM** (such as NVIDIA GeForce RTX 4050 Laptop / RTX 4060) using streaming FFmpeg pipes and intelligent spatial tiling.

---

## ✨ Features

- **Drastic Quality Upscaling**: 2x and 4x super-resolution with fine detail synthesis.
- **Blur & Artifact Reduction**: Removes video noise and JPEG/H.264 compression artifacts.
- **Low VRAM Consumption**: Runs smoothly on 6GB VRAM (or even 4GB) via configurable tile sizes (`--tile 256`).
- **Zero Memory Leaks**: Uses streaming FFmpeg raw video pipes (does not extract thousands of frames to disk).
- **Audio Preservation**: Copies the original audio streams without transcoding.
- **FP16 Half-Precision**: Cuts GPU VRAM usage in half and boosts inference speed.

---

## 🛠️ Installation

### 1. Requirements
- Python 3.10+
- PyTorch 2.0+ with CUDA support
- FFmpeg and FFprobe installed on your system

### 2. Using `uv` (Recommended)
```bash
# Sync dependencies using uv
uv sync
```

### 3. Or using standard pip
```bash
pip install -r requirements.txt
```

---

## 🎬 Quick Start

### Basic 4x Upscaling
```bash
python enhance.py -i input.mp4 -o enhanced_output.mp4
```

### 2x Upscaling (Faster)
```bash
python enhance.py -i input.mp4 -o enhanced_output.mp4 -s 2
```

### Anime / Animated Content
```bash
python enhance.py -i anime.mp4 -o anime_enhanced.mp4 -m RealESRGAN_x4plus_anime_6B
```

---

## ⚙️ Command Line Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `-i`, `--input` | *Required* | Path to the input video file. |
| `-o`, `--output` | *Required* | Path to save the enhanced output video. |
| `-s`, `--scale` | `4` | Upscale factor (`2` or `4`). |
| `-m`, `--model` | `RealESRGAN_x4plus` | Model: `RealESRGAN_x4plus`, `RealESRGAN_x4plus_anime_6B`, `realesr-animevideov3`. |
| `--tile` | `256` | Tile size in pixels for VRAM management (`0` = no tiling). |
| `--tile_pad` | `10` | Overlap padding between tiles for seamless stitching. |
| `--fp32` | `False` | Use FP32 precision instead of default FP16 half-precision. |
| `--cpu` | `False` | Force CPU processing instead of GPU. |
| `--model_dir` | `./models` | Directory where pre-trained model weights are stored. |

---

## 🧪 Running Tests

Run the test suite:
```bash
python -m pytest tests/ -v
```

---

## 📄 License

MIT License.
