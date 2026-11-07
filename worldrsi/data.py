"""Frozen scene partitions and public/private Physics-IQ task indexes."""
from pathlib import Path
import random

from .common import ROOT, digest, immutable_json, jsonl, read_json, sha256, write_jsonl
from .benchmark import VIEWS, build_index, validate_index


def assignments(split):
    groups = split["groups"]
    learning = [s for names in groups["learning"].values() for s in names]
    validation = [s for names in groups["validation"].values() for s in names]
    if len(learning) != 33 or len(validation) != 33 or len(set(learning + validation)) != 66:
        raise ValueError("The partition requires 33 distinct scenes in each group")
    order = sorted(learning)
    random.Random(split["view_seed"]).shuffle(order)
    return {scene: VIEWS[i % 3] for i, scene in enumerate(order)}


def prepare(destination, root=ROOT):
    root, destination = Path(root), Path(destination).resolve()
    runtime = read_json(root / "config/runtime.json")
    split = read_json(root / "protocol/split.json")
    views = assignments(split)
    base = destination / "benchmark"
    expected_frames = Path(runtime["physics_iq"]) / "physics-IQ-benchmark-verified/first-frames"
    if Path(runtime["first_frames"]).resolve() != expected_frames.resolve():
        raise ValueError("First-frame configuration must bind the benchmark first-frames directory")
    source = build_index(base, source_root=runtime["physics_iq"])
    categories = {s: c for group in split["groups"].values() for c, names in group.items() for s in names}
    private = {p["task_id"]: p for p in source["binding"]["private"]}
    actual = {p["scenario"].removeprefix("trimmed-").removesuffix(".mp4") for p in private.values()}
    if actual != set(categories):
        raise ValueError("Partition scenes must exactly match the benchmark index")
    public, secrets = {}, {}
    # Preserve the legacy partition in the index binding and its data revision.
    # Promotion and held-out test selection are derived without changing this binding.
    for phase in ("learning", "transfer", "validation", "test"):
        public[phase], secrets[phase] = [], []
        for p in source["binding"]["public"]:
            q = private[p["task_id"]]
            scene = q["scenario"].removeprefix("trimmed-").removesuffix(".mp4")
            if p["category"] != categories[scene]:
                raise ValueError("Scene category must match the frozen partition")
            include = (phase == "test" or phase == "validation" and scene not in views
                       or phase == "learning" and views.get(scene) == q["view"]
                       or phase == "transfer" and scene in views and views[scene] != q["view"])
            if include:
                public[phase].append({**p, "scene_id": scene, "view": q["view"], "split": phase})
                secrets[phase].append(q)
    binding = {"protocol": runtime["protocol"], "split": split, "learning_views": views,
               "benchmark_revision": source["revision"], "public": public, "private": secrets,
               "first_frames": runtime["first_frames"]}
    revision = digest(binding)
    for phase in public:
        for side, records in (("public", [{**p, "data_revision": revision} for p in public[phase]]),
                              ("private", secrets[phase])):
            path = destination / side / (phase + ".jsonl")
            if path.exists() and jsonl(path) != records:
                raise ValueError("Existing index must match its frozen content")
            if not path.exists():
                write_jsonl(path, records)
    return immutable_json(destination / "manifest.json", {"revision": revision, "binding": binding,
                          "benchmark_index": str(base), "directory": str(destination)})


def verify(directory, check_inputs=False):
    directory = Path(directory)
    m = read_json(directory / "manifest.json")
    if digest(m["binding"]) != m["revision"]:
        raise ValueError("Index binding must match its revision")
    for phase, rows in m["binding"]["public"].items():
        expected = [{**r, "data_revision": m["revision"]} for r in rows]
        if jsonl(directory / "public" / (phase + ".jsonl")) != expected or jsonl(directory / "private" / (phase + ".jsonl")) != m["binding"]["private"][phase]:
            raise ValueError("Indexed tasks must match their frozen binding")
    if check_inputs:
        validate_index(m["benchmark_index"], decode=False)
    return m


def tasks(directory, phase):
    m = verify(directory)
    return [{**r, "data_revision": m["revision"], "split": phase}
            for r in m["binding"]["public"]["learning" if phase == "promotion" else phase]]


def task(directory, phase, task_id):
    m = verify(directory)
    indexed_phase = "learning" if phase == "promotion" else phase
    public = next(r for r in m["binding"]["public"][indexed_phase] if r["task_id"] == task_id)
    private = next(r for r in m["binding"]["private"][indexed_phase] if r["task_id"] == task_id)
    return {**public, "data_revision": m["revision"], "split": phase}, private


def reflection_inputs(directory, task_id):
    """Expose only the learning view's take-1 reference and scoring mask."""
    public, private = task(directory, "learning", task_id)
    if private["view"] != public["view"]:
        raise ValueError("Reflection references must match the learning view")
    references = {}
    for field in ("gt_v1", "mask_v1"):
        path = Path(private[field])
        expected = private["sha256"][field]
        if sha256(path) != expected:
            raise ValueError("Reflection reference must match its frozen hash")
        references[field] = {"path": str(path), "sha256": expected,
                             "media": private["media"][field]}
    return public, {"task_id": task_id, "view": public["view"], "take": 1, **references}
