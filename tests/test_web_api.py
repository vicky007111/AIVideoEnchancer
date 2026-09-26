import pytest
from fastapi.testclient import TestClient
import subprocess
import os
import tempfile
from app import app

client = TestClient(app)

def test_homepage():
    response = client.get("/")
    assert response.status_code == 200
    assert "AIVideoEnhancer" in response.text

def test_system_info():
    response = client.get("/api/system_info")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    assert "name" in data

def test_models_list():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 3
    assert any(m["id"] == "RealESRGAN_x4plus" for m in models)

def test_upload_and_enhance():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test.mp4")
        # Generate dummy 2-frame video
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', 'testsrc=duration=0.1:size=32x32:rate=30',
            '-pix_fmt', 'bgr24',
            '-y',
            input_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        with open(input_path, "rb") as f:
            upload_res = client.post("/api/upload", files={"file": ("test.mp4", f, "video/mp4")})

        assert upload_res.status_code == 200
        upload_data = upload_res.json()
        assert "filename" in upload_data

        # Start enhancement
        enhance_res = client.post("/api/enhance", data={
            "filename": upload_data["filename"],
            "model_name": "RealESRGAN_x4plus",
            "scale": 4,
            "tile_size": 32,
            "use_fp16": False
        })
        assert enhance_res.status_code == 200
        enhance_data = enhance_res.json()
        assert "task_id" in enhance_data
