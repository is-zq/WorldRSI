import os
from pathlib import Path
import tempfile
import unittest

from worldrsi import learning
from worldrsi.common import ROOT, sha256


class LearningEvidenceTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=ROOT / "runs", prefix="evidence-test-")
        self.addCleanup(temp.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        self.root = Path(temp.name)
        self.source = self.root / "task"
        self.source.mkdir()
        self.artifact = self.source / "checked.txt"
        self.artifact.write_text("checked implementation")
        other = self.root / "unrelated"
        other.mkdir()
        os.chdir(other)

    def test_task_relative_evidence_is_independent_of_cwd(self):
        self.assertEqual(learning.evidence(self.source, ["checked.txt"]),
                         [{"path": str(self.artifact), "sha256": sha256(self.artifact)}])

    def test_absolute_evidence_remains_supported(self):
        self.assertEqual(learning.evidence(self.source, [str(self.artifact)])[0]["path"],
                         str(self.artifact))

    def test_legacy_project_relative_evidence_remains_supported(self):
        os.chdir(ROOT)
        relative = self.artifact.relative_to(ROOT)
        self.assertEqual(learning.evidence(self.source, [str(relative)])[0]["path"],
                         str(self.artifact))

    def test_relative_traversal_cannot_escape_source(self):
        (self.root / "outside.txt").write_text("outside")
        with self.assertRaisesRegex(ValueError, "belong"):
            learning.evidence(self.source, ["../outside.txt"])

    def test_symlink_cannot_escape_source(self):
        outside = self.root / "outside.txt"
        outside.write_text("outside")
        (self.source / "link.txt").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "belong"):
            learning.evidence(self.source, ["link.txt"])

    def test_outside_absolute_evidence_is_rejected(self):
        outside = self.root / "outside.txt"
        outside.write_text("outside")
        with self.assertRaisesRegex(ValueError, "belong"):
            learning.evidence(self.source, [str(outside)])
