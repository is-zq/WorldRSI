"""Public runtime contracts, using task-authored synthetic scenes."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import tools_runtime as tools
from tools_runtime.cli import main
from tools_runtime._vendor.common import ROOT, fingerprint, read_json, write_json
from tool_fixtures import author_scene, add_profile


class ToolsRuntimeTests(unittest.TestCase):
    def setUp(self):
        work = ROOT / "runs/tests"
        work.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="tools-", dir=work)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def project(self):
        return author_scene(tools.scaffold(self.work / "project")["project"])

    def test_first_frame_normalization_does_not_implicitly_call_a_model(self):
        source = self.work / "public.png"
        Image.new("RGB", (144, 96), "orange").save(source)
        with patch.object(tools.depth, "estimate_images") as estimate:
            result = tools.prepare_first_frame(source, self.work / "prepared")
        estimate.assert_not_called()
        self.assertEqual(set(result), {"first_frame"})
        with Image.open(result["first_frame"]) as image:
            self.assertEqual(image.size, (832, 480))
        self.assertEqual(read_json(self.work / "prepared/first_frame.json")["source"]["sha256"], fingerprint(source))

    def test_initial_discovery_and_project_have_no_modeling_implementations(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            main(["simulators"])
        entries = json.loads(stream.getvalue())
        self.assertEqual(set(entries), set(tools.simulators.BACKENDS))
        for entry in entries.values():
            self.assertEqual(set(entry) - {"source"}, {"entry", "entry_present", "device", "profiles", "validation"})
            self.assertEqual(entry["profiles"], {})
        self.assertIsNone(entries["ck-mpm"]["entry"])
        self.assertTrue(Path(entries["ck-mpm"]["source"]).is_dir())
        project = Path(tools.scaffold(self.work / "empty")["project"])
        self.assertEqual({p.name for p in project.iterdir()},
                         {"project.godot", "main.tscn", "runner.gd", "depth.gdshader", "surface.gdshader", "palette.json"})
        self.assertEqual(read_json(project / "palette.json"), {"0": {"label": "background", "color": [0, 0, 0]}})
        with self.assertRaisesRegex(ValueError, "scene.gd"):
            tools.simulate({"project": str(project)}, self.work / "unimplemented")
        self.assertFalse((self.work / "unimplemented").exists())

    def test_backend_and_condition_selection_are_explicit(self):
        project = self.project()
        with self.assertRaisesRegex(ValueError, "simulation.backend"):
            tools.physics.configuration({})
        with patch.object(tools.godot, "render", side_effect=AssertionError("must reject before graphics")):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                tools.render(project, self.work / "missing-modes")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            tools.proxy.validate_modes([])
        self.assertFalse((self.work / "missing-modes").exists())

    def test_learned_profile_copies_all_authored_dependencies(self):
        templates = self.work / "templates"
        add_profile(templates)
        (templates / "physics/helper.py").write_text("VALUE = 42\n")
        with redirect_stdout(io.StringIO()):
            main(["physics-new", str(self.work / "learned"), "--profile", "fixture-model", "--templates", str(templates)])
        project = self.work / "learned"
        self.assertEqual((project / "helper.py").read_text(), "VALUE = 42\n")
        self.assertEqual(fingerprint(project / "simulate.py"), fingerprint(templates / "physics/fixture_driver.py"))
        self.assertEqual(fingerprint(project / "scene.gd"), fingerprint(templates / "godot/scene.gd"))

    def test_unknown_fields_and_existing_outputs_are_rejected(self):
        project = self.project()
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            tools.render({**project, "modes": ["rgb"], "unknown": []}, self.work / "unknown")
        output = self.work / "existing"
        output.mkdir()
        (output / "keep.txt").write_text("accepted artifact")
        with self.assertRaises(FileExistsError):
            tools.render({**project, "modes": ["rgb"]}, output)
        self.assertEqual((output / "keep.txt").read_text(), "accepted artifact")
        with self.assertRaisesRegex(ValueError, "separate directory"):
            tools.render({**project, "modes": ["rgb"]}, Path(project["project"]) / "recursive")

    @unittest.skipUnless(shutil.which("godot"), "Godot required")
    def test_authored_simulation_and_checkpoint_preserve_timing_and_files(self):
        project = self.project()
        before = fingerprint(project["project"])
        with patch.object(tools.godot, "display_command", side_effect=AssertionError("No display in simulation")):
            result = tools.simulate({**project, "frames": 8}, self.work / "simulation")
            continued = tools.simulate({"project": result["project"], "action": project["action"],
                "state": result["world_end"], "frames": 2}, self.work / "continued")
        timeline = read_json(result["trajectory"])
        self.assertEqual(timeline["fps"], 24)
        self.assertEqual(read_json(result["world_end"]), timeline["frames"][-1])
        self.assertAlmostEqual(timeline["frames"][-1]["custom"]["elapsed"], 7 / 24, places=5)
        self.assertEqual(timeline["frames"][-1]["custom"]["marker"], 7)
        self.assertAlmostEqual(read_json(continued["world_end"])["custom"]["elapsed"], 8 / 24, places=5)
        self.assertEqual(fingerprint(project["project"]), before)
        self.assertEqual(fingerprint(Path(result["project"]) / "helper.gd"), fingerprint(Path(project["project"]) / "helper.gd"))
        self.assertTrue(read_json(result["checks"])["passed"])

    @unittest.skipUnless(shutil.which("godot"), "Godot required")
    def test_native_body_simulation_and_substeps_resume(self):
        project = self.project()
        directory = Path(project["project"])
        (directory / "scene.gd").rename(directory / "base_scene.gd")
        (directory / "scene.gd").write_text('extends "res://base_scene.gd"\nfunc build(action: Dictionary) -> void:\n\tsuper.build(action)\n\tvar falling := RigidBody3D.new()\n\tfalling.name = "Falling"\n\tfalling.position = Vector3(0, 3, 0)\n\tfalling.angular_velocity = Vector3(0, 1, 0)\n\tadd_child(falling)\n\tvar shape := CollisionShape3D.new()\n\tshape.shape = BoxShape3D.new()\n\tfalling.add_child(shape)\nfunc _process(_delta: float) -> void:\n\tpush_error("Autonomous script must be disabled")\n')
        action = read_json(project["action"])
        action["simulation"]["substeps"] = 4
        write_json(project["action"], action)
        first = tools.simulate({**project, "frames": 24}, self.work / "falling")
        states = read_json(first["trajectory"])["frames"]
        self.assertLess(states[-1]["nodes"]["Falling"]["origin"][1], states[0]["nodes"]["Falling"]["origin"][1] - .1)
        self.assertAlmostEqual(states[-1]["custom"]["elapsed"], 23 / 24, places=5)
        continued = tools.simulate({"project": first["project"], "action": project["action"],
            "state": first["world_end"], "frames": 2}, self.work / "falling-continued")
        tail = read_json(continued["trajectory"])["frames"]
        self.assertEqual(tail[0]["nodes"]["Falling"], states[-1]["nodes"]["Falling"])
        self.assertLess(tail[1]["nodes"]["Falling"]["origin"][1], tail[0]["nodes"]["Falling"]["origin"][1])


@unittest.skipUnless(os.environ.get("WORLDRSI_GPU_TESTS") == "1", "Set WORLDRSI_GPU_TESTS=1 for hardware rendering")
class ToolsHardwareIntegrationTests(unittest.TestCase):
    setUp = ToolsRuntimeTests.setUp
    project = ToolsRuntimeTests.project

    def test_render_replay_five_modes_and_restore(self):
        project = self.project()
        config = {"schema_version": 1, "depth": {"range": "fixed", "near": 0, "far": 20, "gamma": .8},
                  "instance": {"min_channel": 32, "max_channel": 180, "salt": "fixture"}}
        write_json(Path(project["project"]) / "proxy.json", config)
        simulation = tools.simulate({**project, "frames": 3}, self.work / "simulation")
        frozen = fingerprint(simulation["trajectory"])
        script = Path(simulation["project"]) / "scene.gd"
        script.write_text(script.read_text().replace("func step(delta: float) -> void:",
            'func step(delta: float) -> void:\n\tpush_error("Frozen replay must not integrate")'))
        rendered = tools.render({"project": simulation["project"], "action": project["action"],
            "frames": 3, "trajectory": simulation["trajectory"], "modes": list(tools.proxy.MODES)}, self.work / "render")
        check = tools.validate_render(rendered)
        self.assertEqual(set(check["modes"]), set(tools.proxy.MODES))
        self.assertEqual(fingerprint(simulation["trajectory"]), frozen)
        self.assertEqual(read_json(rendered["trajectory"]), read_json(simulation["trajectory"]))
        self.assertEqual(read_json(Path(rendered["proxy_videos"]) / "manifest.json")["configuration"], config)
        self.assertIn("NVIDIA", read_json(rendered["render_manifest"])["renderer"]["vendor"].upper())
        with np.load(Path(rendered["godot_depth"]) / "000.npz") as native:
            self.assertAlmostEqual(float(native["depth"][240, 416]), 4.5, delta=.003)
            np.testing.assert_allclose(native["normal"][240, 416], [0, 0, 1], atol=.02)
            self.assertGreater(int(native["instance"][240, 416]), 0)
        # A project-owned action is used when restoring its state.
        restored = tools.restore(rendered["project"], rendered["world_end"], self.work / "restored", modes=["depth", "normal"], action=project["action"])
        self.assertEqual(tools.validate_render(restored)["frames"], 1)
        self.assertEqual(read_json(restored["world_end"])["custom"], read_json(rendered["world_end"])["custom"])
        with np.load(Path(rendered["godot_depth"]) / "002.npz") as a, np.load(Path(restored["godot_depth"]) / "000.npz") as b:
            np.testing.assert_array_equal(a["depth"], b["depth"])
            np.testing.assert_array_equal(a["normal"], b["normal"])
        changed = read_json(rendered["world_end"])
        changed["custom"]["elapsed"] += 1
        write_json(rendered["world_end"], changed)
        with self.assertRaises(ValueError):
            tools.validate_render(rendered)
