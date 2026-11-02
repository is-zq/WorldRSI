from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
FPS, WIDTH, HEIGHT = 24, 832, 480
MIN_GENERATION_FRAMES, MAX_GENERATION_FRAMES = 96, 336
MODALITIES = ("rgb", "semantic", "depth", "normal", "instance")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree(path):
    path = Path(path)
    return {p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob("*"))
            if p.is_file() and not p.name.endswith((".lock", ".pyc")) and "__pycache__" not in p.parts}


def write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, values):
    write_text(path, "".join(canonical(v) + "\n" for v in values))


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def immutable_json(path, value):
    path = Path(path)
    with lock(path.with_suffix(path.suffix + ".lock")):
        if path.exists():
            if read_json(path) != value:
                raise ValueError(f"Immutable record already exists: {path}")
        else:
            write_json(path, value)
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def inside(root, relative, suffix=None):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or not relative.parts or any(p in {"..", "."} for p in relative.parts):
        raise ValueError("Expected a relative path without traversal")
    target = root / relative
    if any((root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
        raise ValueError("Symbolic links are not accepted")
    if not target.resolve().is_relative_to(root):
        raise ValueError("Path escapes its root")
    if suffix and target.suffix.lower() != suffix:
        raise ValueError(f"Expected {suffix}")
    return target


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value
