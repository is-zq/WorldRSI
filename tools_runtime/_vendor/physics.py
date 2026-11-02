"""Run a scene-authored physical solver once, before Godot replays its states."""
from pathlib import Path
import math
import subprocess
import sys
import time
import uuid

from .common import FPS, fingerprint, read_json, write_json
from . import simulators

BACKENDS = tuple(simulators.BACKENDS)


def configuration(action, project=None):
    config = action.get("simulation")
    if not isinstance(config, dict) or config.get("backend") not in BACKENDS:
        raise ValueError(f"simulation.backend must be one of {BACKENDS}.")
    config = dict(config)
    if config["backend"] == "godot":
        substeps = config.setdefault("substeps", 1)
        if type(substeps) is not int or not 1 <= substeps <= 64:
            raise ValueError("Godot simulation.substeps must be an integer in 1..64.")
    elif not isinstance(config.get("reason"), str) or not config["reason"].strip():
        raise ValueError("External simulation requires a recorded selection reason.")
    if "contract" in config and (type(config["contract"]) is not int or config["contract"] != 1):
        raise ValueError("simulation.contract must be 1 when supplied.")
    simulators.validate_selection(config, project=project)
    return config


def validate_timeline(path, count):
    timeline = read_json(path)
    if timeline.get("fps") != FPS or len(timeline.get("frames", [])) != count:
        raise ValueError("Simulation must export the requested frame count at 24 fps.")
    for index, frame in enumerate(timeline["frames"]):
        time = frame.get("time")
        if (frame.get("frame") != index or not isinstance(time, (int, float)) or
                not math.isfinite(time) or abs(time - index / FPS) > 1e-6 or
                not isinstance(frame.get("custom"), dict) or
                not isinstance(frame.get("nodes", {}), dict)):
            raise ValueError("Simulation frames require frame/time, custom state and optional node poses.")
        validate_finite(frame)
    return timeline


def validate_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Simulation state contains a non-finite number.")
    if isinstance(value, dict):
        for child in value.values():
            validate_finite(child)
    elif isinstance(value, list):
        for child in value:
            validate_finite(child)


def checkpoint(project, state, config):
    """Resolve a contract-1 ending checkpoint inside the copied world."""
    metadata = state.get("custom", {}).get("physics", {})
    if metadata.get("backend") != config["backend"] or metadata.get("profile") != config.get("profile"):
        raise ValueError("Continuation checkpoint backend/profile does not match the requested simulation.")
    resource = metadata.get("checkpoint", "")
    if not isinstance(resource, str) or not resource.startswith("res://"):
        raise ValueError("Continuation requires a res:// checkpoint.")
    path = (project / resource[6:]).resolve(strict=True)
    if not path.is_relative_to(project.resolve()) or not path.is_file():
        raise ValueError("Continuation checkpoint must be a file inside the project.")
    if fingerprint(path) != metadata.get("checkpoint_sha256"):
        raise ValueError("Continuation checkpoint hash mismatch.")
    return metadata


def simulate(config, project, output, count, action, state):
    """External drivers export restorable Godot state; their caches live in the project."""
    driver = Path(config.get("driver", ""))
    if driver.is_absolute() or not str(config.get("driver", "")).strip():
        raise ValueError("simulation.driver must be a project-relative file.")
    driver = (project / driver).resolve(strict=True)
    if not driver.is_relative_to(project) or not driver.is_file():
        raise ValueError("simulation.driver must be inside the rendered project.")
    command = config.get("command", [sys.executable, str(driver)])
    if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command):
        raise ValueError("simulation.command must be a nonempty argv list (no shell).")
    if config.get("contract") == 1 and state:
        checkpoint(project, state, config)
    cache = project / "simulation_cache" / uuid.uuid4().hex[:12]
    cache.mkdir(parents=True)
    request = output / "simulation_request.json"
    write_json(request, {"backend": config["backend"], "project": str(project),
        "fps": FPS, "frames": count, "action": action, "state": state,
        "output": str(cache), "cache_resource": "res://" + cache.relative_to(project).as_posix()})
    # The enclosing render worker supplies the durable job log.
    started = time.monotonic()
    with (output / "solver.log").open("a", encoding="utf-8") as log:
        subprocess.run([*command, "--request", str(request), "--output", str(cache)],
                       cwd=project, check=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    trajectory = cache / "trajectory.json"
    timeline = validate_timeline(trajectory, count)
    details = {}
    if config.get("contract") == 1:
        ending = checkpoint(project, timeline["frames"][-1], config)
        report = read_json(cache / "solver_report.json")
        validate_finite(report)
        if report.get("backend") != config["backend"] or report.get("profile") != config.get("profile"):
            raise ValueError("Solver report does not match the selected backend/profile.")
        if not all(report.get(key) for key in ("version", "device", "model_sha256")):
            raise ValueError("Solver report requires version, device and model_sha256.")
        if ending.get("model_sha256") != report["model_sha256"]:
            raise ValueError("Ending checkpoint and solver report refer to different models.")
        if state and ending["model_sha256"] != state["custom"]["physics"].get("model_sha256"):
            raise ValueError("Continuation changed its bound physical model.")
        details = {"solver": report, "checkpoint": ending}
    return trajectory, {**config, "driver_sha256": fingerprint(driver),
        "request_sha256": fingerprint(request), "cache": cache.relative_to(project).as_posix(),
        "source_trajectory_sha256": fingerprint(trajectory), "elapsed_seconds": elapsed,
        "cache_bytes": sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()), **details}
