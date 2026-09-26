# AIVideoEnhancer Implementation Plan

## Context
We need to build a local-first AI video enhancement CLI tool called "AIVideoEnhancer" that runs on consumer GPUs with 6GB VRAM (RTX 4050 mobile / RTX 4060 desktop). The tool should reduce blur, increase resolution, and improve overall video quality while staying within VRAM constraints.

## Recommended Approach
Based on research, we'll implement a modular pipeline using:
- **Real-ESRGAN** for super-resolution (primary enhancement)
- **GFPGAN** for face restoration (optional)
- **RIFE** for frame interpolation (optional, for smoother motion)
- **NAFNet** replaced with Real-ESRGAN's denoising capability for simplicity

All models will run sequentially with frame-by-frame processing via FFmpeg pipes, using tiling and fp16 precision to fit in 6GB VRAM.

## File Structure
```
AIVideoEnhancer/
├── enhance.py              # Main CLI entry point
├── models/                 # Downloaded model files
│   ├── realesrgan/
│   ├── gfpgan/
│   └── rife/
├── utils/
│   ├── ffmpeg_handler.py   # FFmpeg pipe streaming
│   ├── tile_processor.py   # Image tiling for VRAM management
│   └── model_loader.py     # Model downloading and initialization
├── requirements.txt        # Python dependencies
└── README.md               # Usage instructions
```

## Dependencies
- torch>=2.0.0 (with CUDA support)
- torchvision
- realesrgan>=0.2.5
- gfpgan>=1.3.8
- basicsr
- facexlib
- opencv-python
- ffmpeg-python
- tqdm
- numpy
- requests (for model downloads)

## Pipeline Architecture
```
Input Video
    ↓ (FFmpeg decode pipe)
Raw BGR frames → [Deblur via Real-ESRGAN denoise] → [Face restore via GFPGAN] → [Upscale via Real-ESRGAN] → [Interpolate via RIFE] → Enhanced BGR frames
    ↑                                                  ↓
    └───(Preserve original audio via FFmpeg map)───────┘
```

Each stage:
1. Processes one frame at a time through FFmpeg pipes
2. Uses tiling (256px tiles with 10px padding) for VRAM management
3. Runs in fp16 precision by default
4. Optional based on CLI flags

## CLI Interface
```
python enhance.py -i input.mp4 -o output.mp4 [options]

Options:
  -i, --input      Input video path (required)
  -o, --output     Output video path (required)
  --scale          Upscale factor (2, 4), default 4
  --face_enhance   Enable GFPGAN face restoration
  --interpolate    Enable RIFE frame interpolation (2x)
  --tile           Tile size for VRAM management (0=no tiling), default 256
  --fp32           Use fp32 precision (default fp16)
  --model_dir      Directory to store models (default ./models)
  -v, --verbose    Verbose logging
```

## Task Slices (Vertical Implementation)

### Phase 1: Foundation & FFmpeg Handling
1. Create project structure and requirements.txt
2. Implement FFmpeg pipe streaming utilities
3. Build basic frame reader/writer that preserves audio
4. Acceptance: Can extract frames from video and reassemble with audio unchanged

### Phase 2: Model Integration - Real-ESRGAN Upscaling
1. Download and load Real-ESRGAN model
2. Implement tiling inference for single frames
3. Integrate with FFmpeg pipeline
4. Acceptance: Can upscale video 4x with tiling, runs in <6GB VRAM

### Phase 3: Face Restoration (GFPGAN)
1. Download and load GFPGAN model
2. Implement face detection and restoration per frame
3. Optional pipeline stage
4. Acceptance: Can restore faces in video when enabled

### Phase 4: Frame Interpolation (RIFE)
1. Download and load RIFE model
2. Implement frame interpolation between existing frames
3. Adjust FPS in output accordingly
4. Acceptance: Can create smoother motion via interpolation

### Phase 5: Polish & CLI
1. Add argument parsing with sensible defaults
2. Implement progress bars (tqdm)
3. Add model auto-download on first run
4. Add error handling and logging
5. Acceptance: End-to-end tool works as described

## Checkpoints
- After Phase 1: `python enhance.py -i test.mp4 -o test_out.mp4` produces identical video (no enhancement)
- After Phase 2: Video is visibly sharper and higher resolution
- After Phase 3: Faces are enhanced when `--face_enhance` used
- After Phase 4: Motion is smoother when `--interpolate` used
- Final: All options work together, VRAM stays under 6GB

## Verification
1. Test with various input resolutions (720p, 1080p)
2. Monitor VRAM usage with `nvidia-smi` during processing
3. Verify audio preservation with `ffmpeg -i output -hide_banner`
4. Compare input/output quality visually and with metrics (PSNR/SSIM if possible)
5. Test tiling effectiveness by adjusting `--tile` parameter

## GitHub Repo Preparation
- Initialize git repo
- Create initial commit with project structure
- Add .gitignore for models/, __pycache__, etc.
- Create detailed README with usage examples
- Add license (MIT suggested)