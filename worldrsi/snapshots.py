from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import tempfile

from .common import ROOT, digest, immutable_json, lock, read_json, sha256, tree, write_json


def freeze(policy=None, root=ROOT, *, templates=None, manuals=None):
    root = Path(root)
    policy = Path(policy or root / "policy")
    template_source = Path(templates) if templates is not None else root / "templates"
    manual_source = Path(manuals) if manuals is not None else root / "manuals"
    if templates is not None and not template_source.is_dir():
        raise ValueError("Templates must be an existing directory")
    if manuals is not None and not manual_source.is_dir():
        raise ValueError("Manuals must be an existing directory")
    with lock(root / ".shared-state.lock"):
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        runtime = read_json(root / "config/runtime.json")
        # Bind actual source bytes as well as commit, so dirty source cannot impersonate that commit.
        source = {}
        directories = {name: root / name for name in
                       ("worldrsi", "tools_runtime", "templates", "skills", "prompts", "config", "protocol", "docs")}
        directories["templates"] = template_source
        for directory, origin in directories.items():
            source.update({f"{directory}/{p}": value for p, value in tree(origin).items()})
        binding = {"schema_version": 1, "code_commit": commit, "source": source,
                   "policy": tree(policy), "manuals": tree(manual_source), "runtime": runtime,
                   "h3_prompt_skill": tree(runtime["h3"]["skill"])}
        snapshot_id = digest(binding)
        target = root / "snapshots" / snapshot_id
        if target.exists():
            verify(target)
            return read_json(target / "snapshot.json")
        record = {"snapshot_id": snapshot_id, "binding": binding, "path": str(target.resolve())}
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}-", dir=target.parent))
        try:
            shutil.copytree(policy, stage / "policy")
            shutil.copytree(manual_source, stage / "manuals")
            # Publish the full snapshot atomically; an interrupted copy is never a usable version.
            shutil.copytree(runtime["h3"]["skill"], stage / "h3-prompt-writing")
            for directory, origin in directories.items():
                if origin.exists():
                    shutil.copytree(origin, stage / "source" / directory,
                                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.lock"))
            (stage / "source/templates").mkdir(parents=True, exist_ok=True)
            write_json(stage / "snapshot.json", record)
            verify(stage)
            os.rename(stage, target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return record


def verify(path):
    path = Path(path)
    record = read_json(path / "snapshot.json")
    binding = record["binding"]
    actual_source = tree(path / "source")
    if (digest(binding) != record["snapshot_id"] or tree(path / "manuals") != binding["manuals"]
            or tree(path / "policy") != binding["policy"] or tree(path / "h3-prompt-writing") != binding["h3_prompt_skill"]
            or actual_source != binding["source"]):
        raise ValueError("Agent snapshot changed after freezing")
    return record


def evaluation_key(snapshot, data_revision, scorer):
    return digest({"snapshot_id": snapshot["snapshot_id"], "data_revision": data_revision, "scorer": scorer})


def template_binding(snapshot):
    return {name: value for name, value in snapshot["binding"]["source"].items()
            if name.startswith("templates/")}
