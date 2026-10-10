# Video Enhancer Performance Fixes

## Issues Fixed

### 1. ❌ "Could not find MARK" Error (2x Scale)
**Root Cause:** Missing `RealESRGAN_x2plus.pth` model file.

**Fix:** Downloaded the missing model (67MB) from GitHub releases.

### 2. 🐌 Extremely Slow Processing (30000+ seconds for 1min video)
**Root Cause:** Multiple performance bottlenecks:
- Calling `model.half()` and `model.float()` **on every tile** instead of once
- Model not using optimized data types
- Pixel unshuffle missing for 2x models

**Fixes Applied:**
1. **Removed per-tile dtype conversions** - Cast model once at load time, cast tensors per-frame instead
2. **Added pixel unshuffle support for RealESRGAN_x2plus** - The 2x model uses pixel_unshuffle (3ch→12ch) before processing
3. **Fixed model architecture mismatches**:
   - `RealESRGAN_x2plus`: Added pixel_unshuffle, 2x upsampling via 2 conv layers
   - `realesr-animevideov3`: Corrected to use upscale=4 (checkpoint has 48 output channels)
4. **Smart scale adaptation** - If model's native scale differs from user's requested scale, output is resized automatically
5. **Removed torch.compile** - Avoided 3-minute startup compilation and edge-tile recompilation storms

## Performance Improvement
Expected speedup: **50-100x faster** for typical videos on RTX 4050.

## Files Modified
- `utils/model_loader.py` - Fixed model loading, added pixel unshuffle support, removed torch.compile
- `utils/tile_processor.py` - Removed per-tile dtype conversions, added smart scale adaptation
- `utils/rrdbnet.py` - Added pixel_unshuffle for scale=2 models

## Models Working
✅ RealESRGAN_x4plus (4x, general)
✅ RealESRGAN_x2plus (2x, general) 
✅ RealESRGAN_x4plus_anime_6B (4x, anime)
✅ realesr-animevideov3 (4x, anime - auto-resizes to 2x if requested)

## Testing
Run smoke tests:
```bash
python3 -c "
import numpy as np
from utils.model_loader import get_realesrgan_model
from utils.tile_processor import TileProcessor

processor = TileProcessor(tile_size=256, tile_pad=10, Half_precision=True, device='cuda')
dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

m2 = get_realesrgan_model(scale=2, model_name='RealESRGAN_x2plus', device='cuda').half()
out2 = processor.process(dummy_img, m2, scale=2)
print('2x model output shape:', out2.shape)
"
```

## Usage
```bash
# CLI
python enhance.py -i input.mp4 -o output.mp4 --scale 2 --model RealESRGAN_x2plus

# Web UI
python app.py
# Then open http://localhost:7860
```
