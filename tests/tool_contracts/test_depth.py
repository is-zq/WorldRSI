import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools_runtime._vendor import depth
from tools_runtime._vendor.common import ROOT, read_json, write_json
from tools_runtime._vendor.media import encode_frames
from .depth_fixtures import fake_generate, fake_estimate, fake_native


(ROOT / "runs" / "tests").mkdir(parents=True, exist_ok=True)

class DepthTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs" / "tests")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.source = self.work / "source.png"
        self.candidate = self.work / "candidate.png"
        Image.new("RGB", (832, 480), "tan").save(self.source)
        Image.new("RGB", (832, 480), "brown").save(self.candidate)


    def test_relative_depth_is_scale_invariant_and_masks_unreliable_pixels(self):
        values = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
        confidence = values.copy()
        a, valid = depth.relative_depth(values, confidence)
        b, _ = depth.relative_depth(values * 17, confidence)
        np.testing.assert_allclose(a, b, atol=1e-6)
        self.assertFalse(valid[0, 0])
        self.assertTrue(valid[-1, -1])
        values[-1, -1] = np.nan
        _, valid = depth.relative_depth(values, confidence)
        self.assertFalse(valid[-1, -1])
        flat, mask = depth.relative_depth(np.ones((2, 2)), np.ones((2, 2)))
        np.testing.assert_array_equal(flat, 0.5)
        self.assertTrue(mask.all())

    def test_depth_arrays_camera_and_previews_align_to_480p(self):
        result = fake_generate({"images": [{"path": str(self.source), "role": "Anchor"}]}, self.work / "depth")
        manifest = depth.validate_bundle(result["depth"])
        item = manifest["items"][0]
        with np.load(Path(result["depth"]) / item["data"]) as arrays:
            self.assertEqual(arrays["depth"].shape, (480, 832))
            self.assertEqual(arrays["confidence"].shape, (480, 832))
            self.assertAlmostEqual(arrays["intrinsic"][0, 2], 416)
            self.assertAlmostEqual(arrays["intrinsic"][1, 2], 240)
        with Image.open(Path(result["depth"]) / item["preview"]) as preview:
            self.assertGreater(preview.getpixel((0, 240))[0], preview.getpixel((831, 240))[0])
        self.assertAlmostEqual(item["vertical_fov_degrees"], 43.602819, places=4)
        self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            depth.validate_bundle(result["depth"])


    def test_native_only_preparation_skips_estimator_and_rejects_mismatched_rgb(self):
        bundle = fake_native(self.work / "native", self.source)
        item = {"path": str(self.source), "role": "Godot", "native_depth": {"path": bundle, "frame": 0}}
        with patch.object(depth, "estimate_images") as estimate:
            result = depth.generate({"images": [item]}, self.work / "copied")
            estimate.assert_not_called()
        self.assertEqual(depth.validate_bundle(result["depth"])["items"][0]["method"], "godot")
        with self.assertRaisesRegex(ValueError, "does not match the RGB frame"):
            depth.generate({"images": [{**item, "path": str(self.candidate)}]}, self.work / "wrong")
        with self.assertRaisesRegex(ValueError, "does not match the RGB render"):
            depth.native_frame(bundle, 0, str(self.candidate))


if __name__ == "__main__":
    unittest.main()
