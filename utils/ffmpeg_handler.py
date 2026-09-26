import subprocess
import numpy as np

class FFmpegReader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.pipe = None
        self.width = None
        self.height = None
        self.fps = None
        self.total_frames = None
        self._open()

    def _open(self):
        # Probe the video to get width, height, fps, duration/frames
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames:format=duration',
            '-of', 'csv=p=0',
            self.video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        lines = [line for line in probe_result.stdout.strip().split('\n') if line]

        parts = lines[0].split(',')
        self.width = int(parts[0])
        self.height = int(parts[1])
        r_frame_rate = parts[2]
        num, den = map(int, r_frame_rate.split('/'))
        self.fps = num / den

        if len(parts) > 3 and parts[3].isdigit():
            self.total_frames = int(parts[3])
        elif len(lines) > 1:
            try:
                duration = float(lines[1])
                self.total_frames = int(duration * self.fps)
            except ValueError:
                self.total_frames = None
        else:
            self.total_frames = None

        # Open pipe for reading raw video
        self.pipe = subprocess.Popen([
            'ffmpeg',
            '-i', self.video_path,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-'
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    def read_frame(self):
        # Read one frame: width * height * 3 bytes
        frame_size = self.width * self.height * 3
        raw_frame = self.pipe.stdout.read(frame_size)
        if not raw_frame or len(raw_frame) < frame_size:
            return None
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame

    def close(self):
        if self.pipe:
            self.pipe.terminate()
            self.pipe.wait()

class FFmpegWriter:
    def __init__(self, video_path, fps, width, height, audio_source=None):
        self.video_path = video_path
        self.fps = fps
        self.width = width
        self.height = height
        self.audio_source = audio_source
        self.pipe = None
        self._open()

    def _open(self):
        cmd = [
            'ffmpeg',
            '-y',  # overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # Input 0: raw frames from stdin
        ]

        if self.audio_source:
            cmd.extend([
                '-i', self.audio_source,  # Input 1: audio source
                '-map', '0:v:0',
                '-map', '1:a?',  # Copy audio if present
                '-c:a', 'copy',
            ])

        cmd.extend([
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High visual quality
            '-preset', 'fast',
            self.video_path
        ])

        self.pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    def write_frame(self, frame: np.ndarray):
        # frame is expected to be (H, W, 3) uint8
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame shape {frame.shape} does not match expected ({self.height}, {self.width}, 3)")
        self.pipe.stdin.write(frame.tobytes())

    def close(self):
        if self.pipe:
            self.pipe.stdin.close()
            self.pipe.wait()
