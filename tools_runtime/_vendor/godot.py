from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .common import ROOT, FPS, WIDTH, HEIGHT, read_json, write_json, fingerprint
from .media import encode_frames, restore_palette
from . import physics, proxy

VIRTUALGL = Path("/data/tools/virtualgl/root")


def display_command():
    """VirtualGL redirects Xvfb's OpenGL rendering to the NVIDIA EGL device."""
    executable = Path("/data/bin/vglrun")
    if not executable.is_file():
        raise FileNotFoundError(f"Install the shared VirtualGL launcher: {executable}")
    return ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24", str(executable),
            "-d", "egl"]


def export_depth(output, result, count, source_frames=None):
    """Decode the native depth buffer into a portable, per-frame depth bundle."""
    import numpy as np
    from PIL import Image
    from .depth import export_prediction, GUIDANCE

    directory = output / "godot_depth"
    directory.mkdir()
    items = []
    for index in range(count):
        source = output / "semantic" / f"{index:06d}.png"
        packed = output / "depth" / f"{index:06d}.png"
        metadata = read_json(packed.with_suffix(".json"))
        metadata["instances"] = read_json(output / "instance" / f"{index:06d}.json")
        with Image.open(packed) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.int32)
        if np.any((rgb < 64) | (rgb > 191)):
            raise ValueError("Native depth transport was altered by rendering/postprocessing.")
        rgb -= 64
        code = (rgb[..., 0] << 14) | (rgb[..., 1] << 7) | rgb[..., 2]
        values = code.astype(np.float32) * (metadata["far"] / 2097151.0)
        confidence = (code > 0).astype(np.float32)
        copied = directory / f"{index:03d}_input.png"
        shutil.copy2(source, copied)
        item = export_prediction(directory, index,
            {"path": str(copied), "role": f"Godot native depth, frame {index}",
             "frame": index, "time": (source_frames[index] if source_frames is not None else index) / FPS,
             **({"source_frame": source_frames[index]} if source_frames is not None else {}),
             "method": "godot", **metadata},
            values, confidence, metadata["intrinsic"])
        with Image.open(output / "instance" / f"{index:06d}.png") as image:
            digits = np.asarray(image.convert("RGB"), dtype=np.int32) - 64
        instance = (digits[..., 0] << 14) | (digits[..., 1] << 7) | digits[..., 2]
        with Image.open(output / "normal" / f"{index:06d}.png") as image:
            normal = (np.asarray(image.convert("RGB"), dtype=np.float32) - 64) / 127 * 2 - 1
        instance[code == 0] = 0
        normal[code == 0] = 0
        array_path = directory / item["data"]
        with np.load(array_path) as arrays:
            values = dict(arrays)
        np.savez_compressed(array_path, **values, instance=instance, normal=normal)
        item["path"] = copied.name
        items.append(item)
    write_json(directory / "manifest.json", {
        "method": "godot", "units": "scene_units", "depth_axis": "camera_z",
        "confidence_kind": "validity_mask", "encoding_bits": 21,
        "width": WIDTH, "height": HEIGHT, "fps": FPS,
        "frames": count, "rgb_sha256": fingerprint(result["semantic_end"] if source_frames is not None
                                                    else result.get("proxy", result["semantic_first"])),
        **({"source_frames": source_frames} if source_frames is not None else {}),
        "trajectory_sha256": fingerprint(result["trajectory"]),
        "world_state_sha256": fingerprint(result["world_state"]), "guidance": GUIDANCE, "items": items})
    # Packed PNGs are transport files; the float arrays and camera metadata are retained.
    for mode in ["depth", "instance", "normal"]:
        shutil.rmtree(output / mode)
    return str(directory)


def scaffold(destination, *, templates=None, runtime_assets=None):
    """Copy runtime plumbing and any learned files; the initial scene is unauthored."""
    from .labels import validate_palette

    destination = Path(destination).resolve()
    source = Path(templates) if templates is not None else ROOT / "templates"
    if not source.is_dir():
        raise ValueError("Templates must be an existing directory.")
    infrastructure = Path(runtime_assets) if runtime_assets is not None else Path(__file__).resolve().parents[1] / "assets/godot"
    shutil.copytree(infrastructure, destination)
    if (source / "godot").is_dir():
        shutil.copytree(source / "godot", destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".godot", ".gitkeep"))
    validate_palette(read_json(destination / "palette.json"))
    proxy.configuration(destination)
    return {"project": str(destination)}


def render(request, directory):
    requested_modes = proxy.validate_modes(request.get("modes"))
    source_project = Path(request["project"]).resolve(strict=True)
    output = Path(directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Render the authored project with its own templates and helpers unchanged.
    project = output / "project"
    shutil.copytree(source_project, project, ignore=shutil.ignore_patterns(".godot"))
    from .labels import validate_palette
    validate_palette(read_json(project / "palette.json"))
    proxy.configuration(project)
    count = int(request.get("frames", 1))
    if not 1 <= count <= 336:
        raise ValueError("Render frames must be in 1..336")
    display = display_command()
    env = {**os.environ, "GODOT_SILENCE_ROOT_WARNING": "1"}
    env.pop("LIBGL_ALWAYS_SOFTWARE", None)
    base = ["godot", "--path", str(project), "--audio-driver", "Dummy", "--fixed-fps", str(FPS)]
    def execute(args):
        with (output / "godot.log").open("a", encoding="utf-8") as log:
            subprocess.run(args, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    # Import generated assets before the script is loaded.
    execute(["godot", "--headless", "--editor", "--path", str(project), "--import"])
    for script in ["runner.gd", "scene.gd"]:
        execute(["godot", "--headless", "--path", str(project), "--check-only", "--script", f"res://{script}"])
    source_frames = None
    trajectory = Path(request.get("trajectory", output / "trajectory.json")).resolve()
    common = [f"--output={output}", f"--frames={count}",
              f"--state={request.get('state', '')}", f"--action={request.get('action', '')}"]
    simulation = None
    if not request.get("trajectory"):
        action = read_json(request["action"]) if request.get("action") else {}
        config = physics.configuration(action, project=project)
        if config["backend"] == "godot":
            ticks = FPS * config["substeps"]
            bake_base = [*base[:-1], str(ticks)]
            execute(bake_base + ["--headless", "--", "--mode=bake",
                                f"--substeps={config['substeps']}", *common])
            simulation = config
        else:
            initial = read_json(request["state"]) if request.get("state") else {}
            trajectory, simulation = physics.simulate(config, project, output, count, action, initial)
            request = {**request, "trajectory": str(trajectory), "recapture": True}
            # Import solver-generated meshes before restoring their snapshots.
            execute(["godot", "--headless", "--editor", "--path", str(project), "--import"])
    if request.get("recapture"):
        # Restore the authored timeline in the edited scene and capture actual nodes
        # and custom state, including additions and removals, without advancing time.
        execute(base + ["--headless", "--", "--mode=recapture", f"--trajectory={trajectory}", *common])
        trajectory = output / "trajectory.json"
    data = read_json(trajectory)
    if simulation is not None:
        data["simulation"] = simulation
        write_json(trajectory, data)
    if request.get("frame_indices") is not None:
        source_frames = request["frame_indices"]
        if (not source_frames or source_frames != sorted(set(source_frames)) or
                source_frames[0] < 0 or source_frames[-1] != len(data["frames"]) - 1):
            raise ValueError("Sampled frames must be sorted, unique, in range and include the ending.")
        data = {**data, "frames": [data["frames"][index] for index in source_frames]}
        trajectory = output / "trajectory.json"
        write_json(trajectory, data)
    if len(data["frames"]) != count:
        raise ValueError("Trajectory frame count mismatch")
    write_json(output / "world_end.json", data["frames"][-1])
    # Semantic anchors the native depth and instance data for every render.
    video_modes = [mode for mode in proxy.MODES if mode in {*requested_modes, "semantic"}]
    modes = (["rgb"] if "rgb" in requested_modes else []) + ["semantic", "instance", "normal", "depth"]
    video_directory = output / "proxy_videos"
    result = {"trajectory": str(trajectory), "world_end": str(output / "world_end.json"),
              "world_state": str(output / "world_end.json"), "project": str(project)}
    if data.get("simulation") is not None:
        provenance = output / "simulation.json"
        write_json(provenance, {**data["simulation"], "fps": FPS, "frames": count,
                               "trajectory_sha256": fingerprint(trajectory)})
        result["simulation"] = str(provenance)
    for mode in modes:
        execute([*display, *base, "--display-driver", "x11",
                 "--rendering-method", "gl_compatibility", "--rendering-driver", "opengl3",
                 "--resolution", f"{WIDTH}x{HEIGHT}", "--disable-vsync",
                 "--", "--require-gpu=true", f"--mode={mode}", f"--trajectory={trajectory}", *common])
        frames = sorted((output / mode).glob("*.png"))
        if len(frames) != count:
            raise ValueError(f"{mode}: expected {count} frames, got {len(frames)}; inspect Godot log")
        if mode == "depth":
            result["godot_depth"] = export_depth(output, result, count, source_frames)
            continue
        if mode in {"instance", "normal"}:
            continue
        if mode == "semantic":
            restore_palette(output / mode, read_json(project / "palette.json"))
        if count > 1 and source_frames is None:
            video = video_directory / f"{mode}.mp4"
            encode_frames(output / mode, video, semantic=mode == "semantic")
            if mode == "semantic":
                result["proxy"] = str(video)
        if mode == "semantic":
            result.update(semantic_first=str(frames[0]), semantic_end=str(frames[-1]))
        elif mode == "rgb":
            result.update(rgb_first=str(frames[0]), rgb_end=str(frames[-1]))
    colors = proxy.reserve_instance_colors(result)
    if count > 1 and source_frames is None:
        result["proxy_videos"] = proxy.export(video_directory, result, video_modes, colors)
    write_json(output / "render.json", {"artifacts": result, "trajectory_sha256": fingerprint(trajectory),
                                        "renderer": read_json(output / "renderer.json")})
    return result
