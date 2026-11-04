from pathlib import Path
import tempfile
import unittest

from worldrsi import snapshots
from worldrsi.common import ROOT, digest, tree, write_json


class SnapshotDerivationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs", prefix="snapshot-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "base"
        for relative, content in {"source/worldrsi/frozen.py": "original implementation\n",
                                  "source/templates/layout.json": "{}\n",
                                  "source/config/runtime.json": '{"frozen": true}\n',
                                  "policy/policy.md": "original policy\n",
                                  "manuals/operations.md": "original manual\n",
                                  "h3-prompt-writing/SKILL.md": "original prompt skill\n"}.items():
            target = self.base / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        binding = {"schema_version": 1, "code_commit": "frozen-commit",
                   "source": tree(self.base / "source"), "policy": tree(self.base / "policy"),
                   "manuals": tree(self.base / "manuals"), "runtime": {"frozen": True},
                   "h3_prompt_skill": tree(self.base / "h3-prompt-writing")}
        self.record = {"snapshot_id": digest(binding), "binding": binding, "path": str(self.base)}
        write_json(self.base / "snapshot.json", self.record)

    def test_candidate_retains_frozen_implementation_and_updates_all_components(self):
        replacements = {}
        for component in ("policy", "templates", "manuals"):
            directory = self.root / component
            directory.mkdir()
            (directory / "learned.md").write_text("learned " + component)
            replacements[component] = directory
        candidate = snapshots.derive(self.base, root=self.root, **replacements)
        binding = candidate["binding"]
        original = self.record["binding"]
        self.assertEqual(binding["runtime"], original["runtime"])
        self.assertEqual(binding["code_commit"], original["code_commit"])
        self.assertEqual(binding["h3_prompt_skill"], original["h3_prompt_skill"])
        fixed = lambda b: {k: v for k, v in b["source"].items() if not k.startswith("templates/")}
        self.assertEqual(fixed(binding), fixed(original))
        self.assertEqual(binding["policy"], tree(replacements["policy"]))
        self.assertEqual(binding["manuals"], tree(replacements["manuals"]))
        self.assertEqual(tree(Path(candidate["path"]) / "source/templates"), tree(replacements["templates"]))
        self.assertEqual(snapshots.verify(candidate["path"]), candidate)
        self.assertEqual(snapshots.verify(self.base), self.record)
        self.assertEqual(snapshots.derive(self.base, root=self.root, **replacements), candidate)

    def test_changed_base_is_rejected_before_publication(self):
        (self.base / "source/worldrsi/frozen.py").write_text("modified implementation\n")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            snapshots.derive(self.base, root=self.root)
        self.assertFalse((self.root / "snapshots").exists())
