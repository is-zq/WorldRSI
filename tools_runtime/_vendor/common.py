from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
import fcntl

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
FPS, WIDTH, HEIGHT = 24, 832, 480


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and ".godot" not in p.parts
    )
    for file in files:
        if path.is_dir():
            digest.update(str(file.relative_to(path)).encode())
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        # This shared filesystem can return EAGAIN even for blocking flock.
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.25)
        yield


def frame_count(seconds):
    count = round(float(seconds) * FPS)
    if not 24 <= count <= 336:
        raise ValueError("Chunk duration must be between 1 and 14 seconds (24–336 frames).")
    return count


def aligned_frames(count):
    # H3 reference video encoder requires at least one temporal group.
    count = max(22, int(count))
    return count + (5 - count) % 17
