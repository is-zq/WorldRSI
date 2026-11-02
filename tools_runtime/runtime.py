"""Self-contained execution tools; all paths and tool artifacts stay caller-owned."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import numpy as np

from ._vendor import depth, godot, media, physics, proxy, simulators
from ._vendor.common import FPS, HEIGHT, ROOT, WIDTH, fingerprint, read_json, write_json

CONTRACT = "worldrsi.tools"
REQUEST_FIELDS = frozenset({"project", "frames", "action", "state", "trajectory",
                            "recapture", "modes", "frame_indices"})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _request(request):
    if not isinstance(request, dict) or "project" not in request:
        raise ValueError("A tool request requires a project path.")
    unsupported = request.keys() - REQUEST_FIELDS
    if unsupported:
        raise ValueError(f"Unsupported WorldRSI tool fields: {sorted(unsupported)}")
    result = dict(request)
    count = result.setdefault("frames", 1)
    if type(count) is not int or not 1 <= count <= 336:
        raise ValueError("Tool frames must be an integer in 1..336; H3 inputs require at least 96.")
    result["project"] = str(Path(result["project"]).resolve(strict=True))
    for filename in ("project.godot", "scene.gd"):
        if not (Path(result["project"]) / filename).is_file():
            raise ValueError(f"project must contain authored {filename}.")
    if "modes" in result:
        proxy.validate_modes(result["modes"])
    if "recapture" in result and type(result["recapture"]) is not bool:
        raise ValueError("recapture must be a boolean.")
    if result.get("recapture") and not result.get("trajectory"):
        raise ValueError("recapture requires an authored trajectory.")
    indices = result.get("frame_indices")
    if indices is not None and (not isinstance(indices, list) or not indices or
            any(type(index) is not int for index in indices) or
            indices != sorted(set(indices)) or indices[0] < 0 or len(indices) != count):
        raise ValueError("frame_indices must be sorted distinct nonnegative indices, one per output frame.")
    return result


def _prepare(request, directory):
    request = _request(request)
    output = Path(directory).resolve()
    project = Path(request["project"])
    if output.is_relative_to(project) or project.is_relative_to(output):
        raise ValueError("Output and source project must be separate directory trees.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new output directory for each tool attempt: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sources = {"project": {"path": str(project), "sha256": fingerprint(project)}}
    inputs = output / "inputs"
    inputs.mkdir()
    for key in ("action", "state", "trajectory"):
        if request.get(key):
            source = Path(request[key]).resolve(strict=True)
            target = inputs / f"{key}.json"
            shutil.copy2(source, target)
            read_json(target)
            sources[key] = {"path": str(source), "snapshot": str(target.relative_to(output)),
                            "sha256": fingerprint(target)}
            request[key] = str(target)
    write_json(inputs / "request.json", request)
    return request, output, sources


def scaffold(destination, profile=None, *, templates=None, runtime_assets=None):
    """Create runtime plumbing, optionally applying a learned solver profile."""
    if profile is None:
        return godot.scaffold(destination, templates=templates, runtime_assets=runtime_assets)
    return simulators.scaffold(destination, profile, templates=templates, runtime_assets=runtime_assets)


def prepare_first_frame(source, directory, *, estimate_depth=False):
    """Normalize a public first frame; VGGT depth is explicitly opt-in."""
    output = Path(directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new first-frame output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    original = Path(source).resolve(strict=True)
    frame = media.normalize_image(original, output / "first_frame.png")
    result = {"first_frame": frame}
    if estimate_depth:
        with (output / "depth.log").open("w") as log, contextlib.redirect_stdout(log):
            result.update(depth.generate({"images": [{"path": frame, "role": "Public task first frame"}]},
                                         output / "depth"))
    write_json(output / "first_frame.json", {"schema_version": 1, "artifacts": result,
        "source": {"path": str(original), "sha256": fingerprint(original)},
        "first_frame_sha256": fingerprint(frame), "depth_requested": bool(estimate_depth)})
    return result


def _timeline(path, source_frames=None):
    timeline = read_json(path)
    frames = timeline.get("frames")
    if timeline.get("fps") != FPS or not isinstance(frames, list) or not 1 <= len(frames) <= 336:
        raise ValueError("A frozen trajectory requires 1..336 frames at 24 fps.")
    if source_frames is None:
        source_frames = list(range(len(frames)))
    if len(source_frames) != len(frames):
        raise ValueError("Frozen trajectory sampling length mismatch.")
    for index, frame in zip(source_frames, frames):
        if (not isinstance(frame, dict) or frame.get("frame") != index or
                not isinstance(frame.get("time"), (int, float)) or
                not math.isfinite(frame["time"]) or abs(frame["time"] - index / FPS) > 1e-6 or
                not isinstance(frame.get("nodes"), dict) or not isinstance(frame.get("custom"), dict)):
            raise ValueError("Frozen trajectory requires frame/time, nodes and custom state.")
        physics.validate_finite(frame)
    return timeline


def _cameras(frames):
    cameras = []
    for frame in frames:
        current = [node for node in frame["nodes"].values()
                   if "camera_transform" in node and node.get("current")]
        if len(current) != 1:
            raise ValueError("Each frozen frame must have one current camera.")
        cameras.append(current[0])
    return cameras


def _inventory(artifacts):
    keys = ("project", "trajectory", "world_end", "godot_depth", "proxy_videos", "simulation")
    return {key: fingerprint(artifacts[key]) for key in keys if artifacts.get(key)}


def _load_render(rendered):
    if isinstance(rendered, (str, Path)):
        path = Path(rendered)
        if path.is_dir():
            path /= "render.json"
        saved = read_json(path)
        return saved["artifacts"], saved
    if not isinstance(rendered, dict):
        raise ValueError("Expected render artifacts or a render.json path.")
    if rendered.get("render_manifest"):
        saved = read_json(rendered["render_manifest"])
        if saved["artifacts"] != rendered:
            raise ValueError("Render artifacts do not match their saved manifest.")
        return rendered, saved
    return rendered, {}


def validate_render(rendered, *, check_media=True):
    """Check state, native buffers and common camera/time provenance, without quality scoring."""
    artifacts, saved = _load_render(rendered)
    native = Path(artifacts["godot_depth"])
    manifest = depth.validate_bundle(native)
    trajectory = _timeline(artifacts["trajectory"], manifest.get("source_frames"))
    frames = trajectory["frames"]
    if (manifest.get("method"), manifest.get("width"), manifest.get("height"), manifest.get("fps"),
            manifest.get("frames")) != ("godot", WIDTH, HEIGHT, FPS, len(frames)):
        raise ValueError("Native buffer format, fps or frame count mismatch.")
    if len(manifest["items"]) != len(frames):
        raise ValueError("Native buffers and frozen trajectory have different lengths.")
    for key, artifact in (("trajectory_sha256", "trajectory"), ("world_state_sha256", "world_end")):
        if manifest.get(key) != fingerprint(artifacts[artifact]):
            raise ValueError("Native buffers do not match the frozen trajectory/world state.")
    if read_json(artifacts["world_end"]) != frames[-1]:
        raise ValueError("Ending state differs from the frozen trajectory.")
    cameras = _cameras(frames)
    for index, (item, frame, camera) in enumerate(zip(manifest["items"], frames, cameras)):
        if item.get("frame") != index or abs(item.get("time", -1) - frame["time"]) > 1e-6:
            raise ValueError("Native buffers do not share the frozen timeline.")
        if not np.allclose(item["camera_transform"], camera["camera_transform"], rtol=1e-6, atol=1e-6):
            raise ValueError("Native buffers do not share the frozen camera trajectory.")
        with np.load(native / item["data"], allow_pickle=False) as data:
            for key in ("depth", "confidence", "relative_depth", "valid", "instance"):
                if data[key].shape != (HEIGHT, WIDTH) or not np.isfinite(data[key]).all():
                    raise ValueError(f"Invalid native {key} buffer.")
            if data["normal"].shape != (HEIGHT, WIDTH, 3) or not np.isfinite(data["normal"]).all():
                raise ValueError("Invalid native normal buffer.")
            intrinsic = data["intrinsic"]
            if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0:
                raise ValueError("Invalid native camera intrinsic matrix.")
            if not np.allclose(intrinsic, item["intrinsic"], rtol=1e-6, atol=1e-6):
                raise ValueError("Native camera intrinsics changed.")
            valid = data["valid"]
            observed = set(map(int, np.unique(data["instance"][valid]))) - {0}
            if not observed <= set(map(int, item["instances"])):
                raise ValueError("Native instance codes have no stable object identity.")
    videos = proxy.available(artifacts, len(frames), check_media=check_media) if artifacts.get("proxy_videos") else {}
    if len(frames) > 1 and "source_frames" not in manifest and not videos:
        raise ValueError("A complete render requires synchronized condition videos.")
    if videos and manifest["rgb_sha256"] != fingerprint(artifacts["proxy"]):
        raise ValueError("Native buffers and condition videos have different semantic anchors.")
    if saved.get("hashes") and _inventory(artifacts) != saved["hashes"]:
        raise ValueError("Frozen render artifact hash changed.")
    if artifacts.get("render_manifest"):
        output = Path(artifacts["render_manifest"]).parent
        for source in saved.get("inputs", {}).values():
            if source.get("snapshot") and fingerprint(output / source["snapshot"]) != source["sha256"]:
                raise ValueError("Frozen render input snapshot changed.")
    simulation = trajectory.get("simulation")
    if simulation and simulation.get("contract") == 1:
        physics.checkpoint(Path(artifacts["project"]), frames[-1], simulation)
    elif frames[-1]["custom"].get("physics"):
        physics.checkpoint(Path(artifacts["project"]), frames[-1], frames[-1]["custom"]["physics"])
    result = {"schema_version": 1, "contract": CONTRACT, "passed": True,
        "frames": len(frames), "fps": FPS, "width": WIDTH, "height": HEIGHT,
        "modes": list(videos), "trajectory_sha256": fingerprint(artifacts["trajectory"]),
        "world_state_sha256": fingerprint(artifacts["world_end"]),
        "camera_sha256": _digest(cameras),
        "timeline_sha256": _digest([{key: frame[key] for key in ("frame", "time")} for frame in frames])}
    for key in ("camera_sha256", "timeline_sha256"):
        if saved.get(key) and saved[key] != result[key]:
            raise ValueError(f"Frozen render {key} changed.")
    return result


def render(request, directory):
    """Compute or replay one trajectory and export synchronized condition modalities."""
    proxy.validate_modes(request.get("modes"))
    request, output, sources = _prepare(request, directory)
    started = time.monotonic()
    result = godot.render(request, output)
    checks = validate_render(result)
    result["checks"] = str(output / "checks.json")
    result["render_manifest"] = str(output / "render.json")
    write_json(result["checks"], checks)
    renderer = read_json(output / "renderer.json")
    write_json(result["render_manifest"], {"schema_version": 1, "contract": CONTRACT,
        "artifacts": result, "inputs": sources, "hashes": _inventory(result),
        "camera_sha256": checks["camera_sha256"], "timeline_sha256": checks["timeline_sha256"],
        "renderer": renderer, "elapsed_seconds": time.monotonic() - started})
    return result


def simulate(request, directory):
    """Compute physical states/checkpoints without invoking a graphical renderer."""
    request, output, sources = _prepare(request, directory)
    if request.get("trajectory") or request.get("frame_indices") or request.get("recapture"):
        raise ValueError("simulate computes new states; use render to replay an existing trajectory.")
    project = output / "project"
    shutil.copytree(request["project"], project, ignore=shutil.ignore_patterns(".godot"))
    # The project already contains its selected templates and task-local edits.
    # Replacing helpers here would silently undo template interventions.
    action = read_json(request["action"]) if request.get("action") else {}
    initial = read_json(request["state"]) if request.get("state") else {}
    config = physics.configuration(action, project=project)
    count = request["frames"]
    started = time.monotonic()
    if config["backend"] != "godot":
        trajectory, details = physics.simulate(config, project, output, count, action, initial)
        # External driver trajectories retain project-owned fields and checkpoint resources.
        data = read_json(trajectory)
    else:
        env = {**os.environ, "GODOT_SILENCE_ROOT_WARNING": "1"}
        with (output / "godot.log").open("w") as log:
            commands = [["godot", "--headless", "--editor", "--path", str(project), "--import"]]
            commands += [["godot", "--headless", "--path", str(project), "--check-only", "--script", f"res://{script}"]
                         for script in ("runner.gd", "scene.gd")]
            commands += [["godot", "--headless", "--path", str(project), "--audio-driver", "Dummy",
                "--fixed-fps", str(FPS * config["substeps"]), "--", "--mode=bake",
                f"--frames={count}", f"--substeps={config['substeps']}", f"--output={output}",
                f"--action={request.get('action', '')}", f"--state={request.get('state', '')}"]]
            for command in commands:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        trajectory = output / "trajectory.json"
        data = _timeline(trajectory)
        details = config
    data["simulation"] = details
    target = output / "trajectory.json"
    write_json(target, data)
    physics.validate_timeline(target, count)
    world_end = output / "world_end.json"
    write_json(world_end, data["frames"][-1])
    report = output / "simulation.json"
    write_json(report, {**details, "fps": FPS, "frames": count, "trajectory_sha256": fingerprint(target)})
    checks = output / "checks.json"
    write_json(checks, {"passed": True, "frames": count, "fps": FPS,
                        "check": "finite solver timeline and declared contract-1 checkpoint"})
    result = {"project": str(project), "trajectory": str(target), "world_end": str(world_end),
              "simulation": str(report), "checks": str(checks)}
    write_json(output / "simulate.json", {"schema_version": 1, "contract": CONTRACT,
        "artifacts": result, "inputs": sources, "hashes": _inventory(result),
        "elapsed_seconds": time.monotonic() - started})
    return result


def restore(project, state, directory, *, modes, action=None):
    """Restore an independent one-frame view of an own simulation checkpoint."""
    output = Path(directory).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".restore-{uuid.uuid4().hex}.json"
    snapshot = read_json(state)
    snapshot = {**snapshot, "frame": 0, "time": 0}
    write_json(temporary, {"fps": FPS, "frames": [snapshot]})
    try:
        return render({"project": str(project), "trajectory": str(temporary), "recapture": True,
                       "frames": 1, "modes": proxy.validate_modes(modes),
                       **({"action": str(action)} if action is not None else {})}, output)
    finally:
        temporary.unlink(missing_ok=True)
