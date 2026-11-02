"""Deterministic media checks; no content-based acceptance or retiming."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import subprocess
import shutil

from .common import FPS, WIDTH, HEIGHT, sha256, immutable_json, read_json, lock

FFMPEG = "/data/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/usr/bin/ffprobe"


def probe(path, decode=True):
    args = [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format"]
    if decode:
        args += ["-count_frames", "-show_frames", "-show_entries",
                 "frame=width,height,best_effort_timestamp_time:stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames,duration:format=duration"]
    out = subprocess.run([*args, "-of", "json", str(path)], check=True, capture_output=True, text=True)
    # ffprobe can return exit=0 after a partial decode; reject reported decode errors.
    if out.stderr.strip():
        raise ValueError(f"Video decode errors: {out.stderr[:500]}")
    obj = json.loads(out.stdout)
    if len(obj.get("streams", [])) != 1:
        raise ValueError("Video stream missing")
    stream = obj["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    result = {"width": stream["width"], "height": stream["height"], "fps": fps,
              "duration": float(stream.get("duration", obj["format"].get("duration", 0)))}
    if decode:
        frames = obj["frames"]
        result["frames"] = len(frames)
        if not frames or any((f["width"], f["height"]) != (result["width"], result["height"]) for f in frames):
            raise ValueError("Empty video or changing dimensions")
        times = [float(f["best_effort_timestamp_time"]) for f in frames]
        result["uniform"] = all(abs(b - a - 1 / fps) < 1e-4 for a, b in zip(times, times[1:]))
        result["start_time"] = times[0]
    return result


def check_video(path, frames, width=WIDTH, height=HEIGHT):
    media = probe(path)
    if (media["width"], media["height"], media["frames"]) != (width, height, frames):
        raise ValueError(f"Media shape mismatch: {media}; expected {width}x{height}/{frames} frames")
    if abs(media["fps"] - FPS) > 1e-6 or not media["uniform"] or abs(media["start_time"]) > 1e-4:
        raise ValueError("Video must begin at 0 with uniform 24 fps")
    if abs(media["duration"] - frames / FPS) > 0.002:
        raise ValueError("Video duration mismatch")
    return media


def normalize(source, target, frames):
    """Fixed prefix and center crop with playback-speed-preserving resampling."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-i", str(source), "-map", "0:v:0",
                    "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1,setpts=PTS-STARTPTS",
                    "-frames:v", str(frames), "-r", str(FPS), "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(target)], check=True)
    return check_video(target, frames)


def lock_output(task_dir, raw_video, task):
    task_dir = Path(task_dir)
    with lock(task_dir / ".output.lock"):
        record_path = task_dir / "output.lock.json"
        if record_path.exists():
            record = verify_locked(task_dir)
            if record.get("raw_sha256") != sha256(raw_video):
                raise ValueError("Cannot replace a locked task output")
            return record
        raw_media = check_video(raw_video, task["generation_frames"])
        output = task_dir / "output.mp4"
        if Path(raw_video).resolve() == output.resolve():
            raise ValueError("Raw archive and final output must be different files")
        # Recovery after a crash before the lock: this deterministic prefix may be rebuilt.
        if output.exists():
            output.unlink()
        subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-i", str(raw_video),
                        "-map", "0:v:0", "-map", "0:a?", "-frames:v", str(task["valid_frames"]),
                        "-t", str(task["valid_duration"]), "-c:v", "libx264", "-crf", "18",
                        "-r", str(FPS), "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(output)], check=True)
        media = check_video(output, task["valid_frames"])
        record = {"task_id": task["task_id"], "status": "valid", "task_config": task,
                  "raw_path": str(Path(raw_video).resolve()), "raw_sha256": sha256(raw_video),
                  "raw_media": raw_media, "path": str(output.resolve()), "sha256": sha256(output), "media": media}
        return immutable_json(record_path, record)


def lock_failure(task_dir, task, reason, infrastructure=False):
    if not reason:
        raise ValueError("A terminal failure requires a reason")
    return immutable_json(Path(task_dir) / "output.lock.json", {
        "task_id": task["task_id"], "task_config": task, "status": "invalid",
        "reason": reason, "infrastructure_failure": bool(infrastructure)})


def verify_locked(task_dir):
    record = read_json(Path(task_dir) / "output.lock.json")
    if record["status"] == "valid" and sha256(record["path"]) != record["sha256"]:
        raise ValueError("Locked output was modified")
    return record
