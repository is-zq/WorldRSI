from __future__ import annotations

import gc
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .common import FPS, WIDTH, HEIGHT, fingerprint, read_json, write_json
from .media import normalize_image

SOURCE = Path("/data/vggt-omega")
CHECKPOINT = SOURCE / "vggt_omega_1b_512.pt"
GUIDANCE = (
    "Godot depth is camera-Z in scene units; confidence is a validity mask. "
    "VGGT-Omega depth has arbitrary scale; its confidence is relative and uncalibrated. "
    "Previews are near-bright and far-dark; magenta marks invalid or low-confidence pixels."
)


def relative_depth(depth, confidence):
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    if valid.any():
        valid &= confidence >= np.percentile(confidence[valid], 20)
    relative = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        log_depth = np.log(np.maximum(depth, np.finfo(np.float32).tiny))
        low, high = np.percentile(log_depth[valid], [2, 98])
        if high - low > 1e-6:
            relative[valid] = np.clip((log_depth[valid] - low) / (high - low), 0, 1)
        else:
            relative[valid] = 0.5
    return relative, valid


def export_prediction(directory, index, item, depth, confidence, intrinsic):
    import cv2
    directory = Path(directory)
    h, w = depth.shape
    depth = cv2.resize(depth, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    confidence = cv2.resize(confidence, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
    intrinsic[0] *= WIDTH / w
    intrinsic[1] *= HEIGHT / h
    relative, valid = relative_depth(depth, confidence)
    stem = f"{index:03d}"
    np.savez_compressed(directory / f"{stem}.npz", depth=depth, confidence=confidence,
                        relative_depth=relative, valid=valid, intrinsic=intrinsic)
    gray = np.rint((1 - relative) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgb[~valid] = (255, 0, 255)
    Image.fromarray(rgb).save(directory / f"{stem}_depth.png")
    conf = np.zeros_like(confidence, dtype=np.float32)
    finite = np.isfinite(confidence)
    if finite.any():
        low, high = np.percentile(confidence[finite], [2, 98])
        conf[finite] = np.clip((confidence[finite] - low) / max(high - low, 1e-6), 0, 1)
    if item.get("method") == "godot":
        conf = valid.astype(np.float32)
    Image.fromarray(np.rint(conf * 255).astype(np.uint8)).save(directory / f"{stem}_confidence.png")
    return {**item, "source_sha256": fingerprint(item["path"]), "data": f"{stem}.npz",
            "preview": f"{stem}_depth.png", "confidence_preview": f"{stem}_confidence.png",
            "input": f"{stem}_input.png", "valid_fraction": float(valid.mean()),
            "vertical_fov_degrees": math.degrees(2 * math.atan(HEIGHT / (2 * float(intrinsic[1, 1]))))}


def estimate_images(items, directory):
    """Run VGGT only on the indexed photographic/generated RGB inputs."""
    if not items:
        return {}
    import torch
    if not torch.cuda.is_available():
        raise ValueError("VGGT-Omega inference requires CUDA.")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    sys.path.insert(0, str(SOURCE))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True))
    model = model.to("cuda")
    results = {}
    try:
        for index, item in items:
            source = str(Path(item["path"]).resolve(strict=True))
            prepared = normalize_image(source, directory / f"{index:03d}_input.png")
            images = load_and_preprocess_images([prepared], image_resolution=512).to("cuda")
            with torch.inference_mode():
                prediction = model(images)
                _, intrinsic = encoding_to_camera(prediction["pose_enc"], images.shape[-2:])
            results[index] = export_prediction(directory, index, {**item, "path": source, "method": "vggt"},
                prediction["depth"][0, 0, ..., 0].float().cpu().numpy(),
                prediction["depth_conf"][0, 0].float().cpu().numpy(),
                intrinsic[0, 0].float().cpu().numpy())
            del images, prediction, intrinsic
            print(f"VGGT-Omega depth {len(results)}/{len(items)}: {item['role']}", flush=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return results


def native_frame(directory, index, rgb=None):
    directory = Path(directory)
    manifest = validate_bundle(directory)
    if manifest.get("method") != "godot":
        raise ValueError("Expected native Godot depth bundle.")
    if rgb and fingerprint(rgb) != manifest["rgb_sha256"]:
        raise ValueError("Godot depth does not match the RGB render.")
    item = manifest["items"][index]
    if item["frame"] != index:
        raise ValueError("Godot depth frame index mismatch.")
    return str(directory / item["input"])


def generate(request, directory):
    items = request.get("images", [])
    if not items:
        raise ValueError("Depth preparation requires images.")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for index, item in enumerate(items):
        if not item.get("native_depth"):
            continue
        source = Path(item["native_depth"]["path"])
        frame = item["native_depth"]["frame"]
        manifest = validate_bundle(source)
        if manifest.get("method") != "godot":
            raise ValueError("Expected native Godot depth bundle.")
        native = manifest["items"][frame]
        if native["frame"] != frame or fingerprint(item["path"]) != native["source_sha256"]:
            raise ValueError("Godot depth does not match the RGB frame.")
        copied = {**native, **item, "path": str(Path(item["path"]).resolve()), "method": "godot"}
        for key in ["data", "preview", "confidence_preview", "input"]:
            name = f"{index:03d}" + native[key][len(f"{frame:03d}"):]
            shutil.copy2(source / native[key], directory / name)
            copied[key] = name
        results[index] = copied
    estimated = [(i, item) for i, item in enumerate(items) if not item.get("native_depth")]
    if estimated:
        results.update(estimate_images(estimated, directory))
    write_json(directory / "manifest.json", {"model": str(CHECKPOINT), "image_resolution": 512,
               "width": WIDTH, "height": HEIGHT, "independent_images": True,
               "source_evidence": request.get("source_evidence", []), "guidance": GUIDANCE,
               "items": [results[i] for i in range(len(items))]})
    return {"depth": str(directory)}


def validate_bundle(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if not manifest.get("items"):
        raise ValueError("Depth bundle has no images.")
    for source in manifest.get("source_evidence", []):
        if fingerprint(source["path"]) != source["sha256"]:
            raise ValueError("Depth source changed; regenerate depth estimates.")
    for item in manifest["items"]:
        if fingerprint(directory / item["path"]) != item["source_sha256"]:
            raise ValueError("Depth source changed; regenerate depth estimates.")
        for key in ["data", "preview", "confidence_preview", "input"]:
            if not (directory / item[key]).is_file():
                raise ValueError(f"Missing depth artifact: {item[key]}")
    return manifest
