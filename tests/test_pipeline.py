import subprocess
import os
import tempfile
import torch

def test_full_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        output_path = os.path.join(tmpdir, "output.mp4")

        # Generate a small 3-frame 32x32 test video
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', 'testsrc=duration=0.1:size=32x32:rate=30',
            '-pix_fmt', 'bgr24',
            '-y',
            input_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Run enhance.py --scale 4 --tile 32 --fp32
        # Use CPU if CUDA not working or test on whatever is available
        enhance_cmd = [
            'python',
            'enhance.py',
            '-i', input_path,
            '-o', output_path,
            '--scale', '4',
            '--tile', '32',
            '--fp32'
        ]
        result = subprocess.run(enhance_cmd, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        assert result.returncode == 0, f"Enhance failed with code {result.returncode}"
        assert os.path.exists(output_path), "Output file was not created"

        # Check output dimensions
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            output_path
        ]
        res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        w, h = res.stdout.strip().split(',')
        assert int(w) == 128, f"Expected width 128, got {w}"
        assert int(h) == 128, f"Expected height 128, got {h}"

if __name__ == '__main__':
    test_full_pipeline()
    print("Full pipeline test passed!")