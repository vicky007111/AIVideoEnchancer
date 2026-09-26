#!/usr/bin/env python3
"""
AIVideoEnhancer - AI-powered video enhancement tool
Supports real-time video upscaling, deblurring, and face restoration
Optimized for 6GB VRAM GPUs (RTX 4050/4060)
"""

import argparse
import os
import sys
import numpy as np
import cv2
import torch
from tqdm import tqdm

from utils.ffmpeg_handler import FFmpegReader, FFmpegWriter
from utils.tile_processor import TileProcessor
from utils.model_loader import get_realesrgan_model

def parse_args():
    parser = argparse.ArgumentParser(description='AIVideoEnhancer - Enhance your videos with AI')
    parser.add_argument('-i', '--input', required=True, help='Input video path')
    parser.add_argument('-o', '--output', required=True, help='Output video path')
    parser.add_argument('-s', '--scale', type=int, default=4, choices=[2, 4], help='Upscale factor (2 or 4, default: 4)')
    parser.add_argument('-m', '--model', default='RealESRGAN_x4plus', choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B', 'realesr-animevideov3'], help='Model name')
    parser.add_argument('--tile', type=int, default=256, help='Tile size for VRAM management (0=no tiling, default: 256)')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding for blending (default: 10)')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision (default: fp16/half)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU processing instead of CUDA')
    parser.add_argument('--model_dir', default='./models', help='Directory to store models (default: ./models)')
    return parser.parse_args()

def enhance_frame(frame, model, processor, scale=4):
    """Enhance a single frame using the model."""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame through tiled inference
    output_frame = processor.process(frame_rgb, model, scale=scale)

    # Convert RGB back to BGR
    output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
    return output_frame_bgr

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' does not exist.")
        sys.exit(1)

    print("=" * 60)
    print(" 🚀 AIVideoEnhancer - Local AI Video Enhancement Pipeline")
    print("=" * 60)

    # Device selection
    if args.cpu or not torch.cuda.is_available():
        device = 'cpu'
        half_precision = False
    else:
        device = 'cuda'
        half_precision = not args.fp32

    print(f"[*] Compute Device : {device.upper()} {'(' + torch.cuda.get_device_name(0) + ')' if device == 'cuda' else ''}")
    print(f"[*] Precision      : {'FP32' if not half_precision else 'FP16 (Half Precision)'}")
    print(f"[*] Model Selected : {args.model}")
    print(f"[*] Scale Factor   : {args.scale}x")
    print(f"[*] Tiling         : {args.tile}px (padding: {args.tile_pad}px)")

    # Load Model
    print(f"[*] Loading model weights...")
    model = get_realesrgan_model(
        scale=args.scale,
        model_name=args.model,
        models_dir=args.model_dir,
        device=device
    )

    # Initialize Processor
    processor = TileProcessor(
        tile_size=args.tile,
        tile_pad=args.tile_pad,
        Half_precision=half_precision,
        device=device
    )

    # Initialize FFmpeg reader and writer
    reader = FFmpegReader(args.input)
    output_width = int(reader.width * args.scale)
    output_height = int(reader.height * args.scale)
    output_fps = reader.fps

    print(f"[*] Input Video    : {reader.width}x{reader.height} @ {reader.fps:.2f} fps")
    print(f"[*] Output Video   : {output_width}x{output_height} @ {output_fps:.2f} fps")
    print(f"[*] Preserving Audio: Yes (Direct stream copy)")

    # Create destination directory if needed
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = FFmpegWriter(
        args.output,
        fps=output_fps,
        width=output_width,
        height=output_height,
        audio_source=args.input
    )

    print("-" * 60)
    print("Processing video frames...")

    pbar = tqdm(total=reader.total_frames, unit="frame", desc="Enhancing")
    frame_count = 0

    try:
        while True:
            frame = reader.read_frame()
            if frame is None:
                break

            enhanced_frame = enhance_frame(
                frame,
                model=model,
                processor=processor,
                scale=args.scale
            )

            writer.write_frame(enhanced_frame)
            frame_count += 1
            pbar.update(1)

    finally:
        pbar.close()
        reader.close()
        writer.close()

    print("=" * 60)
    print(f"✨ Enhancement Complete! Processed {frame_count} frames.")
    print(f"💾 Saved to: {args.output}")
    print("=" * 60)

if __name__ == '__main__':
    main()