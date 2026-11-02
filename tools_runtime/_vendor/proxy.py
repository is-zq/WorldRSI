"""Synchronized, viewable Godot condition videos and their frozen provenance."""
from pathlib import Path
import hashlib
import math
import shutil

import numpy as np
from PIL import Image

from .common import FPS, WIDTH, HEIGHT, fingerprint, read_json, write_json
from .media import check_video, encode_frames


MODES = ("rgb", "semantic", "depth", "normal", "instance")
ROLES = {
    "rgb": "RGB rendered from the authored scene.",
    "semantic": "Task-authored categorical RGB codes from palette.json.",
    "depth": "Camera-Z depth in scene units; near-bright, far-dark, black background, fixed clip-wide scale.",
    "normal": "World-space XYZ normals mapped from [-1, 1] to RGB [0, 255]; black background.",
    "instance": "Stable per-object RGB codes; black background.",
}


def validate_modes(modes):
    if (not isinstance(modes, list) or not modes or
            any(not isinstance(mode, str) or mode not in MODES for mode in modes) or
            len(set(modes)) != len(modes)):
        raise ValueError(f"proxy modes must be a nonempty list of distinct names from {MODES}.")
    return modes


def validate_configuration(value):
    """Allow template-owned visualization choices within the fixed modality semantics."""
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int or
            value["schema_version"] != 1 or set(value) - {"schema_version", "depth", "instance"}):
        raise ValueError("Proxy configuration requires schema_version=1, depth and/or instance settings.")
    depth = value.get("depth", {})
    if not isinstance(depth, dict) or set(depth) - {"range", "near", "far", "gamma"}:
        raise ValueError("Unsupported depth mapping settings.")
    depth = {"range": "clip", "gamma": 1.0, **depth}
    gamma = depth["gamma"]
    if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("Depth gamma must be positive and finite.")
    if depth["range"] == "fixed":
        if (any(type(depth.get(key)) not in (int, float) or not math.isfinite(depth[key])
                for key in ("near", "far")) or not 0 <= depth["near"] < depth["far"]):
            raise ValueError("Fixed depth mapping requires finite 0 <= near < far in scene units.")
    elif depth["range"] != "clip" or "near" in depth or "far" in depth:
        raise ValueError("Depth range must be clip, or fixed with near/far bounds.")
    instance = value.get("instance", {})
    if not isinstance(instance, dict) or set(instance) - {"min_channel", "max_channel", "salt"}:
        raise ValueError("Unsupported instance color settings.")
    instance = {"min_channel": 64, "max_channel": 255, "salt": "", **instance}
    if (any(type(instance[key]) is not int for key in ("min_channel", "max_channel")) or
            not 1 <= instance["min_channel"] < instance["max_channel"] <= 255 or
            not isinstance(instance["salt"], str)):
        raise ValueError("Instance colors require 1 <= min_channel < max_channel <= 255 and a string salt.")
    return {"schema_version": 1, "depth": depth, "instance": instance}


def configuration(project=None):
    path = Path(project) / "proxy.json" if project else None
    return validate_configuration(read_json(path) if path and path.is_file() else {"schema_version": 1})


def allocate_instance_colors(identities, previous=None, *, settings=None):
    """Reserve unique colors without changing or recycling previously assigned ones."""
    settings = validate_configuration({"schema_version": 1, "instance": {} if settings is None else settings})["instance"]
    low, high = settings["min_channel"], settings["max_channel"]
    span = high - low + 1
    colors = dict(previous or {})
    occupied = set()
    for identity, color in colors.items():
        if (not isinstance(identity, str) or not isinstance(color, list) or len(color) != 3 or
                any(type(value) is not int or not low <= value <= high for value in color) or
                tuple(color) in occupied):
            raise ValueError("Invalid or duplicate persistent instance colors.")
        occupied.add(tuple(color))
    capacity = span ** 3
    stride = 104729 if math.gcd(104729, capacity) == 1 else 1
    identities = set(identities)
    if any(not isinstance(identity, str) for identity in identities):
        raise ValueError("Instance identities must be strings.")
    # Reserve the whole timeline in sorted order, independent of raster codes or traversal.
    for identity in sorted(identities - colors.keys()):
        if len(occupied) == capacity:
            raise ValueError("Instance color palette exhausted.")
        seed = settings["salt"] + "\0" + identity if settings["salt"] else identity
        digits = [value % span for value in hashlib.sha256(seed.encode()).digest()[:3]]
        code = (digits[0] * span + digits[1]) * span + digits[2]
        while True:
            color = [low + code // (span * span), low + code // span % span, low + code % span]
            if tuple(color) not in occupied:
                break
            # Visit every color without resolving ordinary collisions to a one-byte neighbor.
            code = (code + stride) % capacity
        colors[identity] = color
        occupied.add(tuple(color))
    return colors


def reserve_instance_colors(rendered):
    """Retain stable instance colors when a frozen project is copied for an intervention."""
    path = Path(rendered["project"]) / "instance_colors.json" if rendered.get("project") else None
    settings = configuration(rendered.get("project"))["instance"]
    previous = {}
    retired = set()
    if path and path.exists():
        saved = read_json(path)
        if saved.get("version") != 1 or not isinstance(saved.get("colors"), dict):
            raise ValueError("Unsupported persistent instance color palette.")
        previous = saved["colors"]
        old_settings = validate_configuration({"schema_version": 1, "instance": saved["settings"]})["instance"]
        allocate_instance_colors([], previous, settings=old_settings)
        if old_settings != settings:
            # An explicit mapping intervention reassigns the whole reservation set together.
            retired, previous = set(previous), {}
    manifest = read_json(Path(rendered["godot_depth"]) / "manifest.json")
    identities = {surface["instance_id"] for item in manifest["items"] for surface in item.get("instances", {}).values()}
    colors = allocate_instance_colors(identities | retired, previous, settings=settings)
    if path:
        write_json(path, {"version": 1, "settings": settings, "colors": colors})
    return colors


def export(directory, rendered, modes, colors=None):
    """RGB/semantic are already encoded; derive viewable videos from native arrays."""
    directory = Path(directory)
    native = Path(rendered["godot_depth"])
    manifest = read_json(native / "manifest.json")
    config = configuration(rendered.get("project"))
    depth_config = config["depth"]
    derived = [mode for mode in modes if mode in {"depth", "normal", "instance"}]
    near, far = float("inf"), 0.0
    if depth_config["range"] == "fixed":
        near, far = depth_config["near"], depth_config["far"]
    elif "depth" in derived:
        for item in manifest["items"]:
            with np.load(native / item["data"]) as arrays:
                depth = arrays["depth"]
                values = depth[arrays["valid"] & (depth > 0)]
                if values.size:
                    near, far = min(near, float(values.min())), max(far, float(values.max()))
        if not np.isfinite(near):
            near, far = 0.0, 1.0
    for mode in derived:
        (directory / mode).mkdir()
    if colors is None:
        colors = reserve_instance_colors(rendered) if "instance" in derived else {}
    else:
        allocate_instance_colors([], colors, settings=config["instance"])
    for index, item in enumerate(manifest["items"] if derived else []):
        with np.load(native / item["data"]) as arrays:
            depth = arrays["depth"]
            valid = arrays["valid"] & (depth > 0)
            if "depth" in derived:
                # Leave zero for invalid sky, and keep equal distances equal across time.
                pixels = np.zeros(valid.shape, dtype=np.uint8)
                pixels[valid] = np.rint(1 + 254 * np.clip(
                    (far - depth[valid]) / (far - near), 0, 1) ** depth_config["gamma"]).astype(np.uint8) if far > near else 255
                Image.fromarray(pixels).save(directory / "depth" / f"{index:06d}.png")
            if "normal" in derived:
                pixels = np.rint((np.clip(arrays["normal"], -1, 1) + 1) * 127.5).astype(np.uint8)
                pixels[~valid] = 0
                Image.fromarray(pixels).save(directory / "normal" / f"{index:06d}.png")
            if "instance" in derived:
                lookup = np.zeros((max(map(int, item["instances"]), default=0) + 1, 3), dtype=np.uint8)
                for code, surface in item["instances"].items():
                    identity = surface["instance_id"]
                    lookup[int(code)] = colors[identity]
                pixels = lookup[arrays["instance"]]
                pixels[~valid] = 0
                Image.fromarray(pixels).save(directory / "instance" / f"{index:06d}.png")
    for mode in derived:
        encode_frames(directory / mode, directory / f"{mode}.mp4", semantic=True)
        shutil.rmtree(directory / mode)
    items = []
    for mode in modes:
        path = directory / f"{mode}.mp4"
        check_video(path, manifest["frames"])
        item = {"mode": mode, "path": path.name, "sha256": fingerprint(path), "role": ROLES[mode]}
        if mode == "depth":
            item["encoding"] = {"near": near, "far": far, "units": "scene_units",
                                "range": depth_config["range"], "gamma": depth_config["gamma"],
                                "axis": "camera_z", "near_value": 255, "far_value": 1, "background": 0}
        if mode == "instance":
            item["colors"] = colors
        items.append(item)
    write_json(directory / "manifest.json", {
        "version": 1, "frames": manifest["frames"], "fps": FPS, "width": WIDTH, "height": HEIGHT,
        "configuration": config,
        "trajectory_sha256": manifest["trajectory_sha256"],
        "world_state_sha256": manifest["world_state_sha256"], "items": items})
    return str(directory)


def available(rendered, frames=None, *, check_media=False):
    """Read and validate the explicitly exported condition bundle."""
    directory = Path(rendered["proxy_videos"])
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("version") != 1 or
            (manifest.get("width"), manifest.get("height"), manifest.get("fps")) != (WIDTH, HEIGHT, FPS) or
            (frames is not None and manifest["frames"] != frames)):
        raise ValueError("Invalid proxy video bundle format or frame count.")
    modes = validate_modes([item["mode"] for item in manifest["items"]])
    if "semantic" not in modes:
        raise ValueError("Proxy video bundle requires its semantic anchor.")
    config = validate_configuration(manifest["configuration"])
    if rendered.get("project") and config != configuration(rendered["project"]):
        raise ValueError("Proxy video configuration differs from its project.")
    for key, artifact in [("trajectory_sha256", "trajectory"), ("world_state_sha256", "world_end")]:
        if manifest[key] != fingerprint(rendered[artifact]):
            raise ValueError("Proxy videos do not match the accepted trajectory/world state.")
    result = {}
    for item in manifest["items"]:
        path = directory / item["path"]
        if not path.resolve().is_relative_to(directory.resolve()) or fingerprint(path) != item["sha256"]:
            raise ValueError("Proxy video changed or is outside its bundle.")
        if check_media:
            check_video(path, manifest["frames"])
        result[item["mode"]] = {**item, "path": str(path)}
    if fingerprint(result["semantic"]["path"]) != fingerprint(rendered["proxy"]):
        raise ValueError("Proxy video bundle uses a different semantic anchor.")
    return result


def select(rendered, modes, frames):
    videos = available(rendered, frames)
    validate_modes(modes)
    if any(mode not in videos for mode in modes):
        raise ValueError(f"Selected proxy_modes are unavailable; available modes: {list(videos)}")
    return [(mode, videos[mode]) for mode in modes]
