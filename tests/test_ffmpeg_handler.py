import sys
sys.path.insert(0, '/var/home/vicky/video enchancer')

import numpy as np
import subprocess
import os
import tempfile
from utils.ffmpeg_handler import FFmpegReader, FFmpegWriter

def test_ffmpeg_pipe():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        output_path = os.path.join(tmpdir, "output.mp4")

        # Create a test video: 10 frames of 64x64 RGB color bars
        # Using ffmpeg to generate a test video
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', 'testsrc=duration=0.33:size=64x64:rate=30',
            '-pix_fmt', 'bgr24',
            '-y',
            input_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Now use our FFmpegReader and FFmpegWriter to copy the video
        reader = FFmpegReader(input_path)
        writer = FFmpegWriter(output_path, fps=30, width=64, height=64)

        try:
            while True:
                frame = reader.read_frame()
                if frame is None:
                    break
                writer.write_frame(frame)
        finally:
            reader.close()
            writer.close()

        # Check that the output video exists and has the same properties
        # We can check by reading the first frame and comparing
        cmd_check = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,pix_fmt,nb_frames',
            '-of', 'csv=p=0',
            input_path
        ]
        result_in = subprocess.run(cmd_check, capture_output=True, text=True, check=True)
        width_in, height_in, pix_fmt_in, nb_frames_in = result_in.stdout.strip().split(',')

        cmd_check = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,pix_fmt,nb_frames',
            '-of', 'csv=p=0',
            output_path
        ]
        result_out = subprocess.run(cmd_check, capture_output=True, text=True, check=True)
        width_out, height_out, pix_fmt_out, nb_frames_out = result_out.stdout.strip().split(',')

        assert width_in == width_out, f"Width mismatch: {width_in} vs {width_out}"
        assert height_in == height_out, f"Height mismatch: {height_in} vs {height_out}"
        # We force yuv420p in the writer for compatibility with libx264, so we expect the output to be yuv420p
        assert pix_fmt_out == 'yuv420p', f"Expected output pixel format yuv420p, got {pix_fmt_out}"
        assert nb_frames_in == nb_frames_out, f"Frame count mismatch: {nb_frames_in} vs {nb_frames_out}"

        # Optionally, we can compare the frames by decoding both to raw video and comparing
        # But for now, we trust the properties.

if __name__ == "__main__":
    test_ffmpeg_pipe()
    print("Test passed")