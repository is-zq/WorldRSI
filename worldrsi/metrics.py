"""Official Physics-IQ component implementation and mask generation.

"""
from __future__ import annotations

import contextlib
import importlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import FPS, HEIGHT, ROOT, WIDTH, MAX_GENERATION_FRAMES, digest, immutable_json, read_json, sha256
from .media import check_video, normalize, verify_locked
from .records import metered

BENCHMARK_ROOT = Path("/data/physics-iq-verified-benchmark")
COMPONENTS = ("spatial_iou", "spatiotemporal_iou", "weighted_spatial_iou", "mse")
MASK_PARAMETERS = {
    "threshold": 10, "alpha": 0.3, "grayscale_blur": [5, 5],
    "morphology_kernel": [5, 5], "operations": ["open", "close"], "first_frame": "zero",
}


def physics_modules(root=BENCHMARK_ROOT, *, scoring=False):
    """Import the local authoritative implementation, rejecting package collisions."""
    root = Path(root).resolve()
    names = ["calculate_and_write_metrics_to_csv", "binary_mask_generator"]
    if scoring:
        names += ["calculate_iq_score_stable"]
    if not all((root / "physiq" / (name + ".py")).is_file() for name in names):
        raise FileNotFoundError(f"Physics-IQ source is incomplete: {root}")
    sys.path.insert(0, str(root))
    try:
        modules = [importlib.import_module("physiq." + name) for name in names]
    finally:
        sys.path.pop(0)
    for name, module in zip(names, modules):
        if Path(module.__file__).resolve() != root / "physiq" / (name + ".py"):
            raise ValueError("A different physiq package is already imported")
    return modules


def source_evidence(root=BENCHMARK_ROOT):
    """Record executable source bytes as well as the repository revision."""
    import cv2
    import numpy as np
    root = Path(root).resolve()
    names = ("binary_mask_generator.py", "calculate_and_write_metrics_to_csv.py",
             "calculate_iq_score.py", "calculate_iq_score_stable.py")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True)
    return {"root": str(root), "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "files": {"physiq/" + n: sha256(root / "physiq" / n) for n in names},
            "numpy": np.__version__, "opencv": cv2.__version__, "python": sys.version.split()[0]}


def generate_mask(source, directory, *, benchmark_root=BENCHMARK_ROOT):
    """Call the original generator with its exact naming and compression rules."""
    _, module = physics_modules(benchmark_root)
    source, directory = Path(source), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parts = source.name.split("_")
    if len(parts) != 3:
        raise ValueError("Mask input must use <id>_<view>_<scenario>.mp4")
    target = directory / f"{parts[0]}_video-masks_{FPS}FPS_{parts[1]}_take-1_{parts[2]}"
    if not target.exists():
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            module.generate_mask(str(source), str(directory / source.name), False,
                                 threshold_value=MASK_PARAMETERS["threshold"])
        (directory / (source.stem + ".log")).write_text(log.getvalue(), encoding="utf-8")
    if not target.is_file():
        raise ValueError("The authoritative mask generator did not produce a mask")
    return target


