from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from tools_runtime._vendor import depth
from tools_runtime._vendor.common import fingerprint, write_json
from tools_runtime._vendor.media import normalize_image


def fake_estimate(items, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    values = np.tile(np.linspace(1, 10, 64, dtype=np.float32), (32, 1))
    confidence = np.ones_like(values) * 2
    intrinsic = np.array([[40, 0, 32], [0, 40, 16], [0, 0, 1]], dtype=np.float32)
    results = {}
    for index, item in items:
        normalize_image(item["path"], directory / f"{index:03d}_input.png")
        results[index] = depth.export_prediction(directory, index, {**item, "method": "vggt"}, values, confidence, intrinsic)
    return results


def fake_generate(request, directory):
    request = {**request, "source_evidence": request.get("source_evidence", [
        {"path": path, "sha256": fingerprint(path)} for path in request.get("sources", [])])}
    with patch.object(depth, "estimate_images", side_effect=fake_estimate):
        return depth.generate(request, directory)


def fake_native(directory, rgb, world=None, trajectory=None, frames=None, source_frames=None):
    directory = Path(directory)
    frames = frames or [rgb]
    items = fake_estimate([(i, {"path": str(path), "role": "Native fixture"})
                          for i, path in enumerate(frames)], directory)
    for index, item in items.items():
        shutil.copy2(frames[index], directory / item["input"])
        item.update(path=item["input"], frame=index, method="godot")
        item.update(camera_transform=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], far=100)
        item["source_sha256"] = fingerprint(directory / item["input"])
    write_json(directory / "manifest.json", {"method": "godot", "items": list(items.values()),
        **({"source_frames": source_frames} if source_frames is not None else {}),
        "frames": len(frames), "rgb_sha256": fingerprint(rgb),
        "world_state_sha256": fingerprint(world) if world else None,
        "trajectory_sha256": fingerprint(trajectory) if trajectory else None})
    return str(directory)
