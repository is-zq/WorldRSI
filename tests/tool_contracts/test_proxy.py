import tempfile
import shutil
import unittest
from pathlib import Path

import numpy as np

from tools_runtime._vendor import proxy
from tools_runtime._vendor.common import ROOT, read_json, write_json
from tools_runtime._vendor.media import video_frames


(ROOT / "runs" / "tests").mkdir(parents=True, exist_ok=True)

class ProxyVisualizationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=ROOT / "runs" / "tests")
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)

    def test_collision_allocation_is_unique_and_independent_of_iteration_order(self):
        identities = ["chair_2985", "chair_4722"]
        colors = proxy.allocate_instance_colors(identities)
        self.assertEqual(colors["chair_2985"], [122, 232, 76])
        self.assertNotEqual(colors["chair_2985"], colors["chair_4722"])
        self.assertGreater(np.linalg.norm(np.array(colors["chair_2985"]) - colors["chair_4722"]), 40)
        self.assertEqual(colors, proxy.allocate_instance_colors(reversed(identities)))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            proxy.allocate_instance_colors([], {identity: [122, 232, 76] for identity in identities})

    def test_colliding_instance_videos_preserve_prior_colors_across_project_copies(self):
        previous = None
        palettes = []
        for chunk, identities in enumerate([["chair_4722"], ["chair_2985", "chair_4722"], ["chair_2985"]]):
            project = self.work / f"project_{chunk}"
            if previous:
                shutil.copytree(previous, project)
            else:
                project.mkdir()
            native = self.work / f"native_{chunk}"
            native.mkdir()
            items = []
            for frame in range(2):
                z = np.zeros((480, 832), dtype=np.float32)
                codes = np.zeros(z.shape, dtype=np.int32)
                surfaces = {}
                for index, identity in enumerate(reversed(identities) if frame else identities):
                    # Raster codes and traversal change while object positions stay fixed.
                    x = 100 if identity == "chair_2985" else 300
                    code = index + 1 + frame * 5
                    z[100:200, x:x + 100] = 4
                    codes[100:200, x:x + 100] = code
                    surfaces[str(code)] = {"instance_id": identity}
                np.savez_compressed(native / f"{frame}.npz", depth=z, valid=z > 0, instance=codes)
                items.append({"data": f"{frame}.npz", "instances": surfaces})
            write_json(native / "manifest.json", {"items": items, "frames": 2,
                        "trajectory_sha256": "trajectory", "world_state_sha256": "world"})
            output = self.work / f"videos_{chunk}"
            output.mkdir()
            proxy.export(output, {"project": str(project), "godot_depth": str(native)}, ["instance"])
            colors = read_json(project / "instance_colors.json")["colors"]
            self.assertEqual(read_json(output / "manifest.json")["items"][0]["colors"], colors)
            if palettes:
                self.assertEqual({key: colors[key] for key in palettes[-1]}, palettes[-1])
                self.assertEqual(read_json(previous / "instance_colors.json")["colors"], palettes[-1])
            self.assertEqual(len(set(map(tuple, colors.values()))), len(colors))
            for image in video_frames(output / "instance.mp4"):
                for identity in identities:
                    x = 150 if identity == "chair_2985" else 350
                    self.assertEqual(image.getpixel((x, 150)), tuple(colors[identity]))
            palettes.append(colors)
            previous = project
        self.assertEqual(palettes[0]["chair_4722"], [122, 232, 76])
        self.assertIn("chair_4722", palettes[-1])  # Removed objects keep their reservation.

    def test_depth_scale_and_instance_identity_stay_fixed_across_frames(self):
        native = self.work / "native"
        native.mkdir()
        items = []
        for index in range(2):
            z = np.zeros((480, 832), dtype=np.float32)
            z[100:200, 100:200] = 4  # Equal distance stays equal despite changing extrema.
            z[200:300, 100:200] = 2 if index == 0 else 10
            codes = np.zeros(z.shape, dtype=np.int32)
            codes[100:200, 100:200] = 1 if index == 0 else 9
            codes[200:300, 100:200] = 2
            normals = np.broadcast_to([0, 0, 1], (*z.shape, 3))
            np.savez_compressed(native / f"{index}.npz", depth=z, valid=z > 0,
                                instance=codes, normal=normals)
            items.append({"data": f"{index}.npz", "instances": {
                str(1 if index == 0 else 9): {"instance_id": "chair"},
                "2": {"instance_id": "chair" if index == 0 else "table"}}})
        write_json(native / "manifest.json", {"items": items, "frames": 2,
                    "trajectory_sha256": "trajectory", "world_state_sha256": "world"})
        output = self.work / "videos"
        output.mkdir()
        proxy.export(output, {"godot_depth": str(native)}, ["depth", "normal", "instance"])
        manifest = read_json(output / "manifest.json")
        encoding = manifest["items"][0]["encoding"]
        self.assertEqual((encoding["near"], encoding["far"]), (2, 10))
        depth = video_frames(output / "depth.mp4")
        self.assertEqual(depth[0].getpixel((150, 150)), depth[1].getpixel((150, 150)))
        self.assertEqual(depth[0].getpixel((150, 250)), (255, 255, 255))
        self.assertEqual(depth[1].getpixel((150, 250)), (1, 1, 1))
        self.assertEqual(depth[0].getpixel((0, 0)), (0, 0, 0))
        normal = video_frames(output / "normal.mp4")
        self.assertEqual(normal[0].getpixel((150, 150)), (128, 128, 255))
        self.assertEqual(normal[0].getpixel((0, 0)), (0, 0, 0))
        instance = video_frames(output / "instance.mp4")
        self.assertEqual(instance[0].getpixel((150, 150)), instance[1].getpixel((150, 150)))
        self.assertEqual(instance[0].getpixel((150, 150)), instance[0].getpixel((150, 250)))
        self.assertNotEqual(instance[1].getpixel((150, 150)), instance[1].getpixel((150, 250)))

    def test_empty_and_constant_depth_have_finite_encoding(self):
        for value in [0, 5]:
            with self.subTest(value=value):
                native = self.work / f"native_{value}"
                native.mkdir()
                z = np.full((480, 832), value, dtype=np.float32)
                np.savez_compressed(native / "0.npz", depth=z, valid=z > 0)
                write_json(native / "manifest.json", {"items": [{"data": "0.npz"}], "frames": 1,
                            "trajectory_sha256": "trajectory", "world_state_sha256": "world"})
                output = self.work / f"videos_{value}"
                output.mkdir()
                proxy.export(output, {"godot_depth": str(native)}, ["depth"])
                self.assertEqual(video_frames(output / "depth.mp4")[0].getpixel((0, 0)),
                                 (0, 0, 0) if value == 0 else (255, 255, 255))
