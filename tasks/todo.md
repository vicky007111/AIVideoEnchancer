# AIVideoEnhancer Task List

## Phase 1: Foundation & FFmpeg Handling
- [x] Create project directory structure
- [x] Write requirements.txt with all dependencies
- [x] Implement ffmpeg_handler.py for pipe streaming
- [x] Create basic frame reader/writer that preserves audio
- [x] Test: extract and reassemble video without changes

## Phase 2: Real-ESRGAN Integration
- [x] Implement model_loader.py for Real-ESRGAN weights
- [x] Create RealESRGAN enhancement class with tiling support
- [x] Integrate with FFmpeg pipeline
- [x] Test: upscale video 4x with tiling, monitor VRAM <6GB

## Phase 3: Face Restoration & Interpolation (Optional modules)
- [x] Model loader hooks for GFPGAN and RIFE
- [x] Integrated pipeline flags

## Phase 4: CLI & Polish
- [x] Implement argument parsing in enhance.py
- [x] Add progress bars with tqdm
- [x] Add model auto-download on first run
- [x] Implement error handling and logging
- [x] Test end-to-end with all options combined
- [x] Create README.md with usage examples
- [x] Add .gitignore and initialize git repo