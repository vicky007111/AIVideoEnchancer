# AIVideoEnhancer Task List

## Phase 1: Foundation & FFmpeg Handling
- [x] Create project directory structure
- [x] Write requirements.txt with all dependencies
- [x] Implement ffmpeg_handler.py for pipe streaming
- [x] Create basic frame reader/writer that preserves audio
- [x] Test: extract and reassemble video without changes

## Phase 2: Real-ESRGAN Integration
- [ ] Implement model_downloader.py for Real-ESRGAN weights
- [ ] Create RealESRGAN enhancement class with tiling support
- [ ] Integrate with FFmpeg pipeline
- [ ] Test: upscale video 4x with tiling, monitor VRAM <6GB

## Phase 3: Face Restoration (GFPGAN)
- [ ] Implement GFPGAN model downloader
- [ ] Create face restoration class with detection and enhancement
- [ ] Add as optional pipeline stage
- [ ] Test: face enhancement works with --face_enhance flag

## Phase 4: Frame Interpolation (RIFE)
- [ ] Implement RIFE model downloader
- [ ] Create frame interpolation class
- [ ] Adjust output FPS and frame timing
- [ ] Test: motion smoothing with --interpolate flag

## Phase 5: CLI & Polish
- [ ] Implement argument parsing in enhance.py
- [ ] Add progress bars with tqdm
- [ ] Add model auto-download on first run
- [ ] Implement error handling and logging
- [ ] Test end-to-end with all options combined
- [ ] Create README.md with usage examples
- [ ] Add .gitignore and initialize git repo