from __future__ import annotations

import subprocess
import math
from pathlib import Path

from .common import FPS, WIDTH, HEIGHT

FFMPEG = "/data/bin/ffmpeg"
MAX_IMAGE_CROP = 0.05  # Total removed from one source dimension, split equally at both edges.


def normalize_image(source, output):
    """Center-crop at most 5% of one axis, then resize to fill without padding."""
    from PIL import Image, ImageOps
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        width, height = image.size
        crop_width, crop_height = float(width), float(height)
        if width * HEIGHT > height * WIDTH:
            crop_width = max(height * WIDTH / HEIGHT, width * (1 - MAX_IMAGE_CROP))
        elif width * HEIGHT < height * WIDTH:
            crop_height = max(width * HEIGHT / WIDTH, height * (1 - MAX_IMAGE_CROP))
        left, top = (width - crop_width) / 2, (height - crop_height) / 2
        image = image.resize(
            (WIDTH, HEIGHT), resample=Image.Resampling.LANCZOS,
            box=(left, top, left + crop_width, top + crop_height),
        )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")
    return str(output)


def inspect_video(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video: {path}")
    data = {"width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}
    capture.release()
    if (not math.isfinite(data["fps"]) or data["fps"] <= 0 or data["frames"] <= 0 or
            min(data["width"], data["height"]) <= 0):
        raise ValueError(f"Invalid video stream: {path}")
    data["duration"] = data["frames"] / data["fps"]
    return data


def check_video(path, frames=None):
    import cv2
    info = inspect_video(path)
    if (info["width"], info["height"]) != (WIDTH, HEIGHT) or abs(info["fps"] - FPS) > 0.001:
        raise ValueError(f"Expected {WIDTH}x{HEIGHT}@{FPS}, got {info}: {path}")
    if frames is not None and info["frames"] != frames:
        raise ValueError(f"Expected {frames} frames, got {info['frames']}: {path}")
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image.shape[:2] != (HEIGHT, WIDTH):
                raise ValueError(f"Video dimensions change at frame {decoded}: {path}")
            decoded += 1
    finally:
        capture.release()
    if decoded != info["frames"]:
        raise ValueError(f"Expected {info['frames']} decodable frames, got {decoded}: {path}")
    return info


def image_frame(video, index, output):
    import cv2
    from PIL import Image
    info = inspect_video(video)
    if index < 0:
        index += info["frames"]
    if not 0 <= index < info["frames"]:
        raise ValueError(f"Frame {index} outside video")
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Cannot decode frame {index}: {video}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(output)
    return str(Path(output).resolve())


def video_frames(path):
    import cv2
    from PIL import Image
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded: {path}")
    return frames


def encode_frames(directory, output, semantic=False):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    # RGB lossless proxy avoids chroma subsampling changing label colors.
    codec = ["-c:v", "libx264rgb", "-crf", "0", "-pix_fmt", "rgb24"] if semantic else [
        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"]
    subprocess.run([FFMPEG, "-v", "error", "-y", "-framerate", str(FPS),
                    "-i", str(Path(directory) / "%06d.png"), *codec, str(output)], check=True)


def restore_palette(directory, palette):
    """Decode unlit label pixels into exact palette RGB after framebuffer quantization."""
    import numpy as np
    from PIL import Image
    colors = np.asarray([item["color"] for item in palette.values()], dtype=np.int32)
    for path in sorted(Path(directory).glob("*.png")):
        pixels = np.asarray(Image.open(path).convert("RGB"))
        packed = pixels.astype(np.uint32)
        packed = (packed[..., 0] << 16) | (packed[..., 1] << 8) | packed[..., 2]
        values, inverse = np.unique(packed, return_inverse=True)
        observed = np.stack([(values >> 16) & 255, (values >> 8) & 255, values & 255], axis=-1).astype(np.int32)
        distances = ((observed[:, None, :] - colors[None, :, :]) ** 2).sum(axis=-1)
        nearest = distances.argmin(axis=1)
        if np.any(distances[np.arange(len(values)), nearest] > 8**2):
            raise ValueError(f"Unexpected semantic colors in {path}; check lights, AA, fog and palette.")
        corrected = colors[nearest][inverse].reshape(pixels.shape).astype(np.uint8)
        Image.fromarray(corrected).save(path)
