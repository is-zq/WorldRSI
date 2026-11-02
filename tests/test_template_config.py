"""Template-owned profiles and condition mappings, using synthetic local fixtures."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import tools_runtime as tools
from tools_runtime.cli import main
from tools_runtime._vendor.common import ROOT, fingerprint, read_json, write_json
from tools_runtime._vendor.labels import validate_palette
from tools_runtime._vendor.media import video_frames
from tool_fixtures import add_profile


class TemplateConfigurationTests(unittest.TestCase):
    def setUp(self):
        work = ROOT / "runs/tests"
        work.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="template-config-", dir=work)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.templates = self.work / "templates"
        shutil.copytree(ROOT / "templates", self.templates, ignore=shutil.ignore_patterns("__pycache__"))

    def custom_profile(self):
        return add_profile(self.templates)

    def test_learned_catalog_discovery_and_author_defaults(self):
        spec = self.custom_profile()
        write_json(self.templates / "physics/model.json", {"value": 9})
        write_json(self.templates / "physics/action.json", {"simulation": {"reason": "Authored fixture reason."}})
        stream = io.StringIO()
        with redirect_stdout(stream):
            main(["simulators", "fipy", "--templates", str(self.templates)])
        self.assertEqual(json.loads(stream.getvalue())["fipy"]["profiles"]["fixture-model"], spec)
        project = self.work / "project"
        created = tools.scaffold(project, "fixture-model", templates=self.templates)
        self.assertEqual(read_json(project / "model.json"), {"value": 9})
        action = read_json(created["action"])
        self.assertEqual(action["simulation"]["reason"], "Authored fixture reason.")
        self.assertEqual(tools.physics.configuration(action, project=project), action["simulation"])
        self.assertEqual(fingerprint(project / "simulate.py"), fingerprint(self.templates / "physics/fixture_driver.py"))
        self.assertEqual(tools.simulators.describe("fipy")["fipy"]["profiles"], {})

    def test_authored_profile_executes_and_continues_bound_checkpoint(self):
        self.custom_profile()
        created = tools.scaffold(self.work / "project", "fixture-model", templates=self.templates)
        result = tools.simulate({"project": created["project"], "action": created["action"], "frames": 3}, self.work / "simulation")
        report = read_json(result["simulation"])
        ending = read_json(result["world_end"])
        self.assertEqual(report["solver"]["profile"], "fixture-model")
        self.assertEqual(ending["custom"]["value"], 5)
        continued = tools.simulate({"project": result["project"], "action": created["action"],
                                   "state": result["world_end"], "frames": 2}, self.work / "continued")
        self.assertEqual(read_json(continued["world_end"])["custom"]["value"], 6)
        metadata = ending["custom"]["physics"]
        checkpoint = Path(result["project"]) / metadata["checkpoint"][6:]
        checkpoint.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            tools.simulate({"project": result["project"], "action": created["action"],
                           "state": result["world_end"], "frames": 2}, self.work / "corrupt")

    def test_catalog_paths_and_palette_codes_are_validated(self):
        self.custom_profile()
        catalog_path = self.templates / "physics/profiles.json"
        catalog = read_json(catalog_path)
        catalog["profiles"]["fixture-model"]["driver"] = "../godot/scene.gd"
        write_json(catalog_path, catalog)
        with self.assertRaisesRegex(ValueError, "relative"):
            tools.simulators.describe(templates=self.templates)
        palette = {"0": {"label": "background", "color": [0, 0, 0]},
                   "8": {"label": "object", "color": [0, 0, 0]}}
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_palette(palette)
        palette["8"]["color"] = [5, 6, 7]
        palette["8"]["label"] = "background"
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_palette(palette)

    def test_empty_or_missing_catalog_does_not_load_other_profiles(self):
        self.assertEqual(tools.simulators.load_profiles(self.templates), {})
        self.custom_profile()
        project = self.work / "project-without-catalog"
        project.mkdir()
        self.assertEqual(tools.simulators.load_profiles(project=project), {})
        (self.templates / "physics/profiles.json").unlink()
        self.assertEqual(tools.simulators.load_profiles(self.templates), {})

    def test_source_backend_requires_authored_command(self):
        self.custom_profile()
        path = self.templates / "physics/profiles.json"
        catalog = read_json(path)
        spec = catalog["profiles"]["fixture-model"]
        spec["backend"] = "ck-mpm"
        command = spec["action"]["simulation"].pop("command")
        write_json(path, catalog)
        with self.assertRaisesRegex(ValueError, "authored simulation.command"):
            tools.scaffold(self.work / "missing-command", "fixture-model", templates=self.templates)
        spec["action"]["simulation"]["command"] = command
        write_json(path, catalog)
        created = tools.scaffold(self.work / "authored-command", "fixture-model", templates=self.templates)
        self.assertEqual(read_json(created["action"])["simulation"]["command"], command)

    def native_fixture(self):
        native = self.work / "native"
        native.mkdir()
        items = []
        for frame in range(2):
            depth = np.zeros((480, 832), dtype=np.float32)
            depth[100:200, 100:200] = 4
            depth[200:300, 100:200] = 2 if frame == 0 else 14
            codes = (depth > 0).astype(np.int32)
            np.savez_compressed(native / f"{frame}.npz", depth=depth, valid=depth > 0, instance=codes)
            items.append({"data": f"{frame}.npz", "instances": {"1": {"instance_id": "actor"}}})
        write_json(native / "manifest.json", {"items": items, "frames": 2,
                   "trajectory_sha256": "trajectory", "world_state_sha256": "world"})
        return {"godot_depth": str(native)}

    def test_fixed_depth_mapping_and_gamma_are_applied_to_encoded_frames(self):
        rendered = self.native_fixture()
        rendered["project"] = str(self.templates / "godot")
        config = {"schema_version": 1, "depth": {"range": "fixed", "near": 2, "far": 10, "gamma": 2},
                  "instance": {"min_channel": 20, "max_channel": 80, "salt": "candidate"}}
        write_json(self.templates / "godot/proxy.json", config)
        output = self.work / "videos"
        output.mkdir()
        tools.proxy.export(output, rendered, ["depth", "instance"])
        manifest = read_json(output / "manifest.json")
        self.assertEqual(manifest["configuration"], config)
        self.assertEqual(manifest["items"][0]["encoding"]["gamma"], 2)
        depth = video_frames(output / "depth.mp4")
        for frame in depth:
            self.assertEqual(frame.getpixel((150, 150)), (144, 144, 144))
            self.assertEqual(frame.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(depth[0].getpixel((150, 250)), (255, 255, 255))
        self.assertEqual(depth[1].getpixel((150, 250)), (1, 1, 1))
        colors = manifest["items"][1]["colors"]
        self.assertTrue(all(20 <= value <= 80 for value in colors["actor"]))
        for frame in video_frames(output / "instance.mp4"):
            self.assertEqual(frame.getpixel((150, 150)), tuple(colors["actor"]))

    def test_instance_mapping_changes_deliberately_and_stays_stable_in_copies(self):
        rendered = self.native_fixture()
        project = self.work / "project"
        project.mkdir()
        rendered["project"] = str(project)
        original = tools.proxy.allocate_instance_colors(["actor", "retired"])
        write_json(project / "instance_colors.json", {"version": 1, "settings": tools.proxy.configuration()["instance"], "colors": original})
        self.assertEqual(tools.proxy.reserve_instance_colors(rendered), original)
        config = {"schema_version": 1, "instance": {"min_channel": 1, "max_channel": 63, "salt": "new"}}
        write_json(project / "proxy.json", config)
        changed = tools.proxy.reserve_instance_colors(rendered)
        self.assertEqual(set(changed), {"actor", "retired"})
        self.assertNotEqual(changed, original)
        self.assertEqual(len(set(map(tuple, changed.values()))), len(changed))
        copy_project = self.work / "project-copy"
        shutil.copytree(project, copy_project)
        self.assertEqual(tools.proxy.reserve_instance_colors({**rendered, "project": str(copy_project)}), changed)
        self.assertEqual(read_json(copy_project / "instance_colors.json")["settings"], config["instance"])

    def test_invalid_proxy_settings_are_rejected(self):
        for values in ({"depth": {"range": "fixed", "near": 5, "far": 4}},
                       {"depth": {"gamma": float("nan")}}, {"depth": {"gamma": 0}},
                       {"depth": {"range": "per_frame"}}, {"depth": {"near": 1}},
                       {"instance": {"min_channel": 0}}, {"instance": {"max_channel": True}},
                       {"normal": {"space": "camera"}}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                tools.proxy.validate_configuration({"schema_version": 1, **values})


if __name__ == "__main__":
    unittest.main()
