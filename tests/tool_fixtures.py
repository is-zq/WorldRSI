"""Synthetic interface fixtures; never included in an Agent snapshot."""
from pathlib import Path
import sys

from tools_runtime._vendor.common import write_json

SCENE = '''extends Node3D
const Helper = preload("res://helper.gd")
var elapsed := 0.0
var speed := 1.0
var body: MeshInstance3D
func build(action: Dictionary) -> void:
\tspeed = float(action["speed"])
\tbody = MeshInstance3D.new()
\tbody.name = "Object"
\tbody.mesh = BoxMesh.new()
\tbody.set_meta("instance_id", "fixture")
\tbody.set_meta("semantic_id", 1)
\tadd_child(body)
\tvar material := StandardMaterial3D.new()
\tmaterial.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
\tmaterial.albedo_color = Color.RED
\tbody.material_override = material
\tvar camera := Camera3D.new()
\tcamera.name = "Camera"
\tadd_child(camera)
\tcamera.position = Vector3(0, 0, 5)
\tcamera.current = true
func step(delta: float) -> void:
\telapsed += delta
\tbody.position.x = elapsed * speed
func export_state() -> Dictionary:
\treturn {"elapsed": elapsed, "marker": Helper.VALUE}
func restore_state(state: Dictionary) -> void:
\telapsed = float(state.get("elapsed", 0.0))
\tbody.position.x = elapsed * speed
'''

DRIVER = '''import argparse, hashlib, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--request"); p.add_argument("--output")
a = p.parse_args()
r = json.loads(Path(a.request).read_text())
out = Path(a.output)
project = Path(r["project"])
model = project / "model.json"
identity = hashlib.sha256(model.read_bytes()).hexdigest()
config = r["action"]["simulation"]
initial = r["state"].get("custom", {}).get("value", json.loads(model.read_text())["value"])
frames = []
for i in range(r["frames"]):
    checkpoint = out / f"checkpoint-{i}.json"
    value = initial + i
    checkpoint.write_text(json.dumps({"value": value}))
    metadata = {"backend": config["backend"], "profile": config["profile"],
        "model_sha256": identity, "checkpoint": "res://" + checkpoint.relative_to(project).as_posix(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    frames.append({"frame": i, "time": i / 24, "custom": {"value": value, "physics": metadata}})
(out / "trajectory.json").write_text(json.dumps({"fps": 24, "frames": frames}))
(out / "solver_report.json").write_text(json.dumps({"backend": config["backend"],
    "profile": config["profile"], "model_sha256": identity, "version": "fixture", "device": "CPU"}))
'''


def author_scene(project, *, speed=1.0):
    project = Path(project)
    (project / "scene.gd").write_text(SCENE)
    (project / "helper.gd").write_text("extends RefCounted\nconst VALUE = 7\n")
    write_json(project / "palette.json", {"0": {"label": "background", "color": [0, 0, 0]},
                                         "1": {"label": "fixture", "color": [19, 91, 173]}})
    write_json(project / "action.json", {"simulation": {"backend": "godot"}, "speed": speed})
    return {"project": str(project), "action": str(project / "action.json")}


def add_profile(templates, *, name="fixture-model", value=3):
    templates = Path(templates)
    (templates / "godot").mkdir(parents=True, exist_ok=True)
    (templates / "physics").mkdir(parents=True, exist_ok=True)
    author_scene(templates / "godot")
    # Profile action values supply the deliberately synthetic external interface.
    (templates / "godot/action.json").unlink()
    (templates / "physics/fixture_driver.py").write_text(DRIVER)
    spec = {"backend": "fipy", "driver": "fixture_driver.py", "scope": "Synthetic interface fixture.",
            "model": {"value": value}, "action": {"speed": 1.0,
                "simulation": {"command": [sys.executable, "simulate.py"]}}}
    write_json(templates / "physics/profiles.json", {"schema_version": 1, "profiles": {name: spec}})
    return spec
