"""Installed solver facts and template-owned physical model profiles."""
from pathlib import Path
import copy
import json
import shutil

from .common import ROOT, read_json, write_json

SIMULATORS = Path("/data/simulators")

# Installation identifiers and execution locations; no model-selection policy.
BACKENDS = {
    "godot": ("godot", "CPU"),
    "libuipc": ("python310", "CUDA"),
    "splishsplash": ("splishsplash-python", "CPU/CUDA"),
    "mls-mpm": ("mls-cpic-python", "CPU/CUDA"),
    "ck-mpm": (None, "CUDA"),
    "mpm-lite": ("mpm-lite-python", "CUDA"),
    "hot": ("hot", "CPU"),
    "anisompm": ("anisompm", "CPU"),
    "cd-mpm": ("cd-mpm", "CPU"),
    "silly-rubber": ("silly-rubber", "CPU"),
    "iq-mpm": ("iq-mpm", "CPU/CUDA"),
    "ppf": ("ppf-python", "CUDA"),
    "genesis": ("python310", "CUDA"),
    "fipy": ("fipy-python", "CPU"),
}

def _source_file(directory, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Profile files must be relative to the template or project directory.")
    path = (Path(directory) / name).resolve(strict=True)
    if not path.is_relative_to(Path(directory).resolve()) or not path.is_file():
        raise ValueError("Profile file must stay inside its template or project.")
    return path


def load_profiles(templates=None, *, project=None):
    """Read the selected tree's catalog, or a project's copied catalog."""
    directory = Path(project) if project is not None else Path(templates or ROOT / "templates") / "physics"
    path = directory / "profiles.json"
    if not path.is_file():
        return {}
    catalog = read_json(path)
    if (not isinstance(catalog, dict) or type(catalog.get("schema_version")) is not int or
            catalog["schema_version"] != 1 or set(catalog) != {"schema_version", "profiles"}):
        raise ValueError("Profile catalog requires schema_version=1 and profiles.")
    profiles = catalog["profiles"]
    if not isinstance(profiles, dict):
        raise ValueError("Profiles must be a name-to-definition mapping.")
    for name, spec in profiles.items():
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
            raise ValueError("Profile names use lowercase letters, digits, hyphens and underscores.")
        if not isinstance(spec, dict) or spec.get("backend") not in BACKENDS:
            raise ValueError(f"Unsupported backend in profile {name}.")
        if not isinstance(spec.get("scope"), str) or not spec["scope"].strip() or not isinstance(spec.get("model"), dict):
            raise ValueError(f"Profile {name} needs scope and model defaults.")
        if not isinstance(spec.get("driver_profile", name), str) or not spec.get("driver_profile", name):
            raise ValueError("driver_profile must identify the driver's implementation.")
        for key in ("action", "parameters", "labels"):
            if not isinstance(spec.get(key, {}), dict):
                raise ValueError(f"Profile {key} must be a mapping.")
        if not isinstance(spec.get("files", []), list):
            raise ValueError("Profile files must be a list.")
        for file in [spec.get("driver"), *spec.get("files", [])]:
            _source_file(directory, file)
        json.dumps(spec, allow_nan=False)
    return profiles


def _merge(defaults, overrides):
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_selection(config, project=None):
    """Validate a selected learned profile against its project-owned catalog."""
    profile = config.get("profile")
    if profile is not None:
        if project is None:
            raise ValueError("A profile requires its project catalog.")
        profiles = load_profiles(project=project)
        if not isinstance(profile, str) or profile not in profiles or profiles[profile]["backend"] != config["backend"]:
            raise ValueError("simulation.profile must belong to the selected backend.")


def describe(backend=None, *, templates=None):
    profiles = load_profiles(templates)
    names = [backend] if backend else list(BACKENDS)
    result = {}
    for name in names:
        if name not in BACKENDS:
            raise ValueError(f"Unknown simulator: {name}")
        entry, device = BACKENDS[name]
        path = (shutil.which(entry) if name == "godot" else str(SIMULATORS / "bin" / entry)) if entry else None
        result[name] = {
            "entry": path, "entry_present": bool(path and Path(path).is_file()),
            "device": device,
            "profiles": {key: copy.deepcopy(value) for key, value in profiles.items() if value["backend"] == name},
            "validation": "Entry presence is not a physics or continuation acceptance test.",
        }
        if name == "ck-mpm":
            result[name]["source"] = str(SIMULATORS / "sources/ck-mpm")
    return result


def scaffold(destination, profile, *, templates=None, runtime_assets=None):
    """Copy a self-contained editable driver, physical model and Godot scene."""
    from .godot import scaffold as new_project
    from .labels import validate_palette
    profiles = load_profiles(templates)
    if profile not in profiles:
        raise ValueError(f"Unknown physics profile: {profile}")
    spec = profiles[profile]
    template_root = Path(templates) if templates is not None else ROOT / "templates"
    project = Path(new_project(destination, templates=template_root, runtime_assets=runtime_assets)["project"])
    source = template_root / "physics"
    # Copy added helper modules/assets too, so evolved drivers remain self-contained.
    shutil.copytree(source, project, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".godot"))
    shutil.copy2(source / spec["driver"], project / "simulate.py")
    if not (project / "profiles.json").exists():
        write_json(project / "profiles.json", {"schema_version": 1, "profiles": profiles})
    palette = read_json(project / "palette.json")
    labels = spec.get("labels", {})
    for label, color in labels.items():
        if not any(v["label"] == label for v in palette.values()):
            palette[str(max(map(int, palette), default=0) + 1)] = {"label": label, "color": color}
    write_json(project / "palette.json", validate_palette(palette))
    model_path = project / "model.json"
    model = read_json(model_path) if model_path.exists() else {}
    if not isinstance(model, dict):
        raise ValueError("Template model.json must contain an object.")
    write_json(model_path, _merge(spec["model"], model))
    entry = describe(spec["backend"], templates=template_root)[spec["backend"]]["entry"]
    config = {"backend": spec["backend"], "profile": profile, "contract": 1,
              "driver": "simulate.py", "command": [entry, "simulate.py"],
              "reason": spec["scope"]}
    if "solvers" in spec:
        config["solvers"] = spec["solvers"]
    action = project / "action.json"
    action_values = _merge({"simulation": config}, spec.get("action", {}))
    if action.exists():
        overrides = read_json(action)
        if not isinstance(overrides, dict):
            raise ValueError("Template action.json must contain an object.")
        action_values = _merge(action_values, overrides)
    simulation = action_values.get("simulation")
    if (not isinstance(simulation, dict) or simulation.get("backend") != spec["backend"]
            or simulation.get("profile") != profile):
        raise ValueError("Template action must retain the selected backend and profile.")
    if entry is None and not simulation.get("command"):
        raise ValueError("A source-only backend requires an authored simulation.command.")
    if entry is None and any(value is None for value in simulation["command"]):
        raise ValueError("A source-only backend requires an authored simulation.command.")
    from .physics import configuration
    configuration(action_values, project=project)
    _source_file(project, simulation.get("driver"))
    write_json(action, action_values)
    return {"project": str(project), "action": str(action), "profile": profile, "scope": spec["scope"]}
