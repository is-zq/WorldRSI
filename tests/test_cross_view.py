import copy
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from worldrsi import agents, cli, data, evaluation, learning, records, reports, snapshots, workflow
from worldrsi.common import ROOT, digest, read_json, sha256, write_json, write_jsonl


def fixture_index(root):
    split = read_json(ROOT / "protocol/split.json")
    views = data.assignments(split)
    first, prompt = root / "frame.png", root / "prompt.txt"
    first.write_bytes(b"fixture-first-frame")
    prompt.write_text("A public physics task.")
    public, private = {p: [] for p in ("learning", "transfer", "validation", "test")}, {}
    number = 0
    for group, cats in split["groups"].items():
        for category, scenes in cats.items():
            for scene in scenes:
                for view in ("perspective-left", "perspective-center", "perspective-right"):
                    number += 1
                    phases = ["test", "validation" if group == "validation" else "learning" if views[scene] == view else "transfer"]
                    for phase in phases:
                        public[phase].append({"task_id": f"view_{number:03d}", "scene_id": scene, "category": category,
                            "view": view, "split": phase, "first_frame": str(first), "prompt": str(prompt),
                            "input_hashes": {"first_frame": sha256(first), "prompt": sha256(prompt)},
                            "valid_frames": 120, "generation_frames": 120, "valid_duration": 5, "generation_duration": 5,
                            "fps": 24, "width": 832, "height": 480, "start_seconds": 0})
    gt, mask = root / "take1.mp4", root / "mask1.mp4"
    gt.write_bytes(b"take-one-reference")
    mask.write_bytes(b"take-one-mask")
    media = {"fps": 24, "frames": 120, "width": 832, "height": 480}
    private = {p: [{"task_id": r["task_id"], "view": r["view"],
                   "gt_v1": str(gt), "mask_v1": str(mask),
                   "gt_v2": str(root / "excluded-take2.mp4"), "mask_v2": str(root / "excluded-mask2.mp4"),
                   "sha256": {"gt_v1": sha256(gt), "mask_v1": sha256(mask)},
                   "media": {"gt_v1": media, "mask_v1": media}} for r in rows] for p, rows in public.items()}
    binding = {"public": public, "private": private, "learning_views": views, "split": split}
    revision = digest(binding)
    index = root / "index"
    for phase in public:
        write_jsonl(index / "public" / (phase + ".jsonl"), [{**r, "data_revision": revision} for r in public[phase]])
        write_jsonl(index / "private" / (phase + ".jsonl"), private[phase])
    write_json(index / "manifest.json", {"revision": revision, "binding": binding, "benchmark_index": str(index / "benchmark")})
    return index


def fixture_report(directory, value=0.4, tokens=200):
    plan = evaluation.load(directory)
    values = {str(s): {t: value for t in plan["task_ids"]} for s in plan["seeds"]}
    report = {**plan, "count": len(plan["task_ids"]), "scores": values, "mean": value, "sample_std": 0,
              "categories": {"fixture": value}, "evidence": {}, "costs": {"unknown_fields": [],
              "totals": {"h3_generations": len(plan["task_ids"])*4, "local_gpu_seconds": 0,
                         "model_tokens": tokens, "local_cpu_seconds": 10}}}
    report["report_sha256"] = digest(report)
    write_json(Path(directory) / "report.json", report)
    return report


class CrossViewTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=ROOT / "runs", prefix="cross-view-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.index = fixture_index(self.root)
        self.snap = snapshots.freeze()
        self.source_check = patch("worldrsi.data.validate_index", return_value={"status": "passed"})
        self.source_check.start()
        self.addCleanup(self.source_check.stop)

    def run_start(self):
        state = workflow.init_run("fixture", self.index, self.snap["path"], root=self.root)
        run = Path(state["directory"])
        return run, workflow.begin_round(run)

    def test_learning_starts_without_any_test_or_reference(self):
        run, batch = self.run_start()
        self.assertEqual(len(batch["sources"]), 11)
        self.assertEqual(workflow.begin_round(run), batch)
        self.assertFalse((run / "initial-test").exists())
        self.assertNotIn("reference", read_json(run / "run.json"))
        self.assertEqual(read_json(Path(batch["directory"]) / "history.json"), {"bundles": [], "promotion": []})


    def scored_source(self, source, initial, best):
        source = Path(source)
        def read_score(path):
            return read_json(Path(path) / "scoring/score.json")
        fixture_report(source / "initial", value=initial)
        with patch("worldrsi.evaluation.score", side_effect=read_score):
            branch = learning.branch(source, {"id": "improve", "hypothesis": "Different representation helps",
                "prediction": "More accurate contact", "executor_instruction": "Use continuous collision geometry"})
        d = Path(branch["directory"])
        fixture_report(d, value=best)
        (d / "checked.txt").write_text("Checked implemented geometry")
        record = {"status": "sufficient_gain", "best_attempt": "improve", "reason": "Gain is sufficient for this initial score and remaining headroom",
                  "evidence": [str(d / "checked.txt")], "observations": ["Remaining appearance uncertainty"],
                  "bundle": {"trigger": "Visible collision", "instruction": "Use continuous geometry", "scope": "Collision guidance",
                             "implementation_checks": ["Inspect generated collider geometry"]}}
        return source, record, read_score

    def test_split_and_epoch_schedule(self):
        m = data.verify(self.index)
        self.assertEqual([len(data.tasks(self.index, p)) for p in ("learning", "transfer", "validation", "test")], [33,66,99,198])
        self.assertEqual(sorted(list(m["binding"]["learning_views"].values()).count(v) for v in data.VIEWS), [11,11,11])
        scenes = list(m["binding"]["learning_views"])
        for epoch in range(1,5):
            batches = workflow.epoch_order(scenes, epoch)
            self.assertEqual(list(map(len,batches)), [11,11,11])
            self.assertEqual(set(sum(batches,[])),set(scenes))
            self.assertEqual(batches,workflow.epoch_order(list(reversed(scenes)),epoch))
        self.assertNotEqual(workflow.epoch_order(scenes,1),workflow.epoch_order(scenes,2))

    def test_contextual_gain_closure_and_immutable_experience(self):
        run, batch = self.run_start()
        for item, initial, best in zip(batch["sources"], [0.3, 0.93], [0.43, 0.94]):
            source, record, read_score = self.scored_source(item["directory"], initial, best)
            with patch("worldrsi.evaluation.score", side_effect=read_score):
                result = learning.close(source, record)
                self.assertAlmostEqual(result["gain"], best-initial)
                self.assertEqual(learning.close(source, record),result)
                with self.assertRaises(ValueError):
                    learning.close(source,{**record,"reason":"changed"})
            with self.assertRaises(ValueError):
                learning.branch(source, {"id":"next"})
            with self.assertRaises(ValueError):
                agents.packet("reflector", self.root / "closed-packet", source_dir=source)

    def test_sufficient_gain_requires_improvement(self):
        _, batch = self.run_start()
        for item, best in zip(batch["sources"], [0.4,0.39]):
            source, record, read_score = self.scored_source(item["directory"],0.4,best)
            if best < 0.4: record["best_attempt"] = "initial"
            with patch("worldrsi.evaluation.score",side_effect=read_score), self.assertRaises(ValueError):
                learning.close(source,record)

    def test_reflector_summary_and_frozen_learning_evidence(self):
        _,batch=self.run_start()
        source,record,read_score=self.scored_source(batch["sources"][0]["directory"],0.2,0.4)
        packet=self.root/"reflector-packet"
        with patch("worldrsi.evaluation.score",side_effect=read_score):
            agents.packet("reflector",packet,source_dir=source)
            learning.close(source,record)
        payload=read_json(packet/"input.json")
        self.assertEqual(payload["progress"]["initial_score"],0.2)
        self.assertEqual(payload["attempts"]["improve"]["score"]["score"],0.4)
        self.assertNotIn("attempts",payload["progress"])
        self.assertEqual(payload["public"],read_json(source/"initial/task.json")["public"])
        self.assertEqual(Path(payload["public"]["first_frame"]).read_bytes(), b"fixture-first-frame")
        self.assertEqual(Path(payload["public"]["prompt"]).read_text(), "A public physics task.")
        self.assertEqual(payload["history"], {"bundles": [], "promotion": []})
        self.assertEqual(set(payload["attempts"]), {"initial", "improve"})
        self.assertIn("decisions", payload["attempts"]["initial"])
        self.assertIn("outputs", payload["attempts"]["initial"]["score"])
        truth = payload["ground_truth"]
        self.assertEqual(truth["take"], 1)
        self.assertEqual(truth["view"], payload["public"]["view"])
        self.assertEqual(set(truth), {"task_id", "view", "take", "gt_v1", "mask_v1"})
        for field in ("gt_v1", "mask_v1"):
            self.assertEqual(sha256(truth[field]["path"]), truth[field]["sha256"])
        self.assertNotIn("excluded-", json.dumps(payload))
        write_json(source/"branches/improve/report.json",{"mean":0.5})
        with self.assertRaises(ValueError):learning.reviewed_experience(source)

    def test_closed_source_is_ready_without_transfer_and_retains_evidence_checks(self):
        _, batch = self.run_start()
        source, record, _ = self.scored_source(batch["sources"][0]["directory"], 0.2, 0.4)
        closed = learning.close(source, record)
        self.assertFalse((source / "transfer").exists())
        reviewed = learning.reviewed_experience(source)
        self.assertEqual(reviewed["observations"], record["observations"])
        self.assertEqual(reviewed["bundle"], record["bundle"])
        self.assertNotIn("transfer", reviewed)
        bundle = learning.accepted_bundle(source)
        self.assertEqual(bundle["experience_sha256"], digest(closed))
        self.assertNotIn("transfer", bundle)
        plan = evaluation.load(source / "initial")
        with self.assertRaises(ValueError):
            agents.packet("executor", self.root / "replica-executor",
                          task_dir=next(iter(plan["outputs"]["43"].values())))
        checked = Path(record["evidence"][0])
        checked.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Frozen implementation evidence"):
            learning.accepted_bundle(source)

    def test_closed_source_rejects_changed_learning_scores(self):
        _, batch = self.run_start()
        source, record, _ = self.scored_source(batch["sources"][0]["directory"], 0.2, 0.4)
        learning.close(source, record)
        fixture_report(source / "branches/improve", value=0.5)
        with self.assertRaisesRegex(ValueError, "frozen score binding"):
            learning.reviewed_experience(source)

    def test_transfer_cannot_be_created_or_dispatched_but_archives_remain_readable(self):
        run, _ = self.run_start()
        archived = run / "archived-transfer"
        # Persist the former evaluation format directly, without its retired creator.
        rows = data.tasks(self.index, "transfer")[:2]
        binding = {"snapshot": self.snap["snapshot_id"], "revision": rows[0]["data_revision"],
                   "phase": "transfer", "ids": [r["task_id"] for r in rows], "intervention": None,
                   "seeds": [42, 43, 44, 45], "views": sorted(r["view"] for r in rows), "mode": "full",
                   "extended_from": None, "scorer": read_json(ROOT / "protocol/scorer.json"), "boundary": None}
        outputs = {str(seed): {r["task_id"]: str(archived / "seeds" / str(seed) / r["task_id"])
                              for r in rows} for seed in binding["seeds"]}
        plan = {k: binding[k] for k in ("phase", "intervention", "seeds", "views", "mode",
                                      "extended_from", "scorer", "boundary")}
        plan.update(key=digest(binding), directory=str(archived), index_dir=str(self.index),
                    snapshot_id=self.snap["snapshot_id"], snapshot_path=self.snap["path"],
                    data_revision=rows[0]["data_revision"], task_ids=binding["ids"], outputs=outputs,
                    expected_generations=8, additional_generations=8)
        write_json(archived / "evaluation.json", plan)
        for seed, tasks in outputs.items():
            for row in rows:
                write_json(Path(tasks[row["task_id"]]) / "task.json",
                           {"public": row, "phase": "transfer", "generation_seed": int(seed)})
        frozen = (archived / "evaluation.json").read_bytes()
        historical = fixture_report(archived)
        self.assertEqual(evaluation.report(archived), historical)
        self.assertEqual(evaluation.pending(archived)["expected_generations"], 8)
        tid = plan["task_ids"][0]
        target = outputs["42"][tid]
        packet = self.root / "retired-packet"
        with self.assertRaisesRegex(ValueError, "Active phases"):
            evaluation.prepare(self.root / "new-transfer", self.index, self.snap["path"],
                               "transfer", task_ids=plan["task_ids"])
        self.assertFalse((self.root / "new-transfer").exists())
        with self.assertRaisesRegex(ValueError, "Active phases"):
            evaluation.create_task(self.root / "new-task", rows[0], self.snap)
        for role, args in (("executor", {"task_dir": target}), ("scorer", {"evaluation_dir": archived})):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "Active phases"):
                agents.packet(role, packet, **args)
        self.assertFalse(packet.exists())
        with self.assertRaisesRegex(ValueError, "Active phases"):
            evaluation.submit(archived, tid)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            cli.parser().parse_args(["transfer", "prepare", str(archived)])
        self.assertEqual(error.exception.code, 2)
        records.cost(target, "archived-generation", "h3", {"h3_generations": 1})
        summary = reports.summarize(run)
        self.assertEqual(summary["costs"]["transfer"]["totals"]["h3_generations"], 1)
        self.assertTrue(all("transfer" not in src for r in summary["rounds"] for src in r["sources"]))
        self.assertEqual((archived / "evaluation.json").read_bytes(), frozen)

    def test_stopping_does_not_require_tests(self):
        for count, streak, expected in [(1, 1, False), (3, 3, False), (5, 5, True), (12, 0, True)]:
            state = {"rounds": [{"decision": {}} for _ in range(count)], "rejection_streak": streak, "tests": []}
            self.assertEqual(workflow.stopping(state), expected)


    def test_cost_tie_and_incompatible_evaluations(self):
        run,batch=self.run_start()
        plan=evaluation.prepare(run/"comparison",self.index,self.snap["path"],"promotion",
                                task_ids=[q["task_id"] for q in batch["sources"]],boundary={"round":1})
        base=fixture_report(plan["directory"])
        candidate=copy.deepcopy(base)
        candidate["mean"]+=0.01
        self.assertTrue(evaluation.compare(candidate,base)["accepted"])
        candidate["mean"]=base["mean"]
        candidate["costs"]["totals"]["model_tokens"]-=1
        self.assertTrue(evaluation.compare(candidate,base)["accepted"])
        candidate["costs"]["unknown_fields"]=["local_gpu_seconds"]
        self.assertFalse(evaluation.compare(candidate,base)["accepted"])
        candidate["task_ids"]=candidate["task_ids"][:-1]
        with self.assertRaises(ValueError):evaluation.compare(candidate,base)

    def test_eleven_source_barrier_and_history_without_transfer_or_promotion(self):
        run, batch = self.run_start()
        history_path = Path(batch["directory"]) / "history.json"
        frozen_history = history_path.read_bytes()
        frozen_sources = {}
        with self.assertRaises(FileNotFoundError):
            workflow.manual_inputs(run)
        for i, item in enumerate(batch["sources"]):
            d = Path(item["directory"])
            if i < 3:
                d, record, _ = self.scored_source(d, 0.2, 0.4)
                closed = learning.close(d, record)
                if i:
                    # Historical pass/fail receipts are no longer a prerequisite.
                    receipt = d / "transfer/decision.json"
                    write_json(receipt, {"status": "evaluated", "passed": i == 2,
                                         "experience_sha256": digest(closed)})
                    frozen_sources[receipt] = receipt.read_bytes()
            else:
                status = ("stalled", "blocked", "saturated")[i % 3]
                fixture_report(d / "initial", value=1 if status == "saturated" else 0.4)
                learning.close(d, {"status": status, "best_attempt": "initial", "reason": "Fixture closure",
                                   "recovery_condition": "Tool restored" if status == "blocked" else None,
                                   "observations": ["Keep the source's useful observations"]})
            frozen_sources[d / "experience.json"] = (d / "experience.json").read_bytes()
            if i != len(batch["sources"]) - 1:
                with self.assertRaises(FileNotFoundError):
                    workflow.manual_inputs(run)
            if i not in (1, 2):
                self.assertFalse((d / "transfer").exists())
        inputs = workflow.manual_inputs(run)
        shutil.copytree(inputs["base_manuals"], inputs["output_manuals"])
        Path(inputs["merge_notes"]).write_text("Retained checked operations.")
        workflow.publish_manuals(run)
        packet = self.root / "editor"
        agents.packet("editor", packet, run_dir=run)
        experiences = read_json(packet / "input.json")["experiences"]
        self.assertEqual(len(experiences), 11)
        statuses = {"sufficient_gain", "stalled", "blocked", "saturated"}
        self.assertEqual({e["status"] for e in experiences}, statuses)
        self.assertTrue(all(e["observations"] and "transfer" not in e for e in experiences))
        c = workflow.candidate(run)
        self.assertEqual(c["snapshot_id"], self.snap["snapshot_id"])
        fixture_report(c["baseline_evaluation"])
        fixture_report(c["evaluation"])
        self.assertFalse(workflow.decide(run)["accepted"])
        state = read_json(run / "run.json")
        self.assertEqual(state["current_snapshot"], self.snap["path"])
        self.assertEqual(state["rejection_streak"], 1)
        history = workflow.history(run)
        self.assertEqual({b["scene_id"] for b in history["bundles"]},
                         {s["scene_id"] for s in batch["sources"][:3]})
        self.assertTrue(all("transfer" not in b for b in history["bundles"]))
        next_batch = workflow.begin_round(run)
        self.assertEqual(read_json(Path(next_batch["directory"]) / "history.json"), history)
        self.assertEqual(history_path.read_bytes(), frozen_history)
        for path, original in frozen_sources.items():
            self.assertEqual(path.read_bytes(), original)
        for item in next_batch["sources"]:
            d = Path(item["directory"])
            fixture_report(d / "initial", value=1)
            learning.close(d, {"status": "saturated", "best_attempt": "initial", "reason": "Already at ceiling"})
        combined = workflow.editor_experiences(run)
        self.assertEqual(len(combined), 22)
        self.assertEqual({e["round"] for e in combined}, {1, 2})
        self.assertEqual({e["status"] for e in combined}, statuses)
        self.assertNotIn("transfer", reports.summarize(run)["costs"])

    def test_accepted_candidate_activates_complete_snapshot_and_next_batch_history(self):
        run,batch=self.run_start()
        old_history=read_json(Path(batch["directory"])/"history.json")
        for item in batch["sources"]:
            d=Path(item["directory"])
            fixture_report(d/"initial")
            learning.close(d,{"status":"blocked","best_attempt":"initial", "reason":"External tool unavailable",
                              "recovery_condition":"Tool restored", "observations":["Keep uncertain scope explicit"]})
        inputs=workflow.manual_inputs(run)
        shutil.copytree(inputs["base_manuals"],inputs["output_manuals"])
        (Path(inputs["output_manuals"])/"operations.md").write_text("# Checked operational guidance\nA fixture operation.\n")
        Path(inputs["merge_notes"]).write_text("Merged checked operational evidence.")
        workflow.publish_manuals(run)
        policy = self.root / "synthesized-policy"
        shutil.copytree(Path(self.snap["path"])/"policy",policy)
        common = policy / snapshots.DECISION_EXPERIENCE
        content = common.read_text() + "\nPreserve contact geometry.\nFor marked spheres on turntables, retain attached landmarks.\nA failed contact test leaves timing uncertain.\n"
        common.write_text(content)
        raw = policy / "references/raw.json"
        write_json(raw,{"bundle":"raw experience"})
        with self.assertRaisesRegex(ValueError,"consolidated decision experience"):
            workflow.candidate(run,policy=policy)
        raw.unlink()
        c=workflow.candidate(run,policy=policy)
        self.assertNotEqual(c["snapshot_id"],self.snap["snapshot_id"])
        fixture_report(c["baseline_evaluation"])
        fixture_report(c["evaluation"],value=0.5)
        self.assertTrue(workflow.decide(run)["accepted"])
        state=read_json(run/"run.json")
        self.assertEqual(state["current_snapshot"],c["snapshot_path"])
        self.assertEqual(state["rounds"][-1]["decision"]["baseline_mean"],0.4)
        next_batch=workflow.begin_round(run)
        self.assertEqual(next_batch["snapshot_path"],c["snapshot_path"])
        self.assertEqual(read_json(Path(batch["directory"])/"history.json"),old_history)
        self.assertEqual(len(read_json(Path(next_batch["directory"])/"history.json")["promotion"]),1)
        packet = self.root / "fresh-executor"
        agents.packet("executor",packet,task_dir=Path(next_batch["sources"][0]["directory"])/"initial")
        payload = read_json(packet/"input.json")
        self.assertEqual(Path(payload["decision_experience"]).read_text(),content)
        self.assertIsNone(payload["intervention"])
        self.assertNotIn("ground_truth",payload)
        self.assertNotIn("experiences",payload)
        self.assertNotIn("take1.mp4",json.dumps(payload))

    def test_reflection_references_reject_missing_changed_or_wrong_view(self):
        _, batch = self.run_start()
        source = Path(batch["sources"][0]["directory"])
        fixture_report(source/"initial")
        public = read_json(source/"initial/task.json")["public"]
        _, private = data.task(self.index,"learning",public["task_id"])
        with patch("worldrsi.data.task",return_value=(public,{**private,"view":"wrong-view"})):
            with self.assertRaisesRegex(ValueError,"learning view"):
                agents.packet("reflector",self.root/"wrong-view",source_dir=source)
        for field in ("gt_v1","mask_v1"):
            file = Path(private[field]); original = file.read_bytes()
            file.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"frozen hash"):
                agents.packet("reflector",self.root/(field+"-changed"),source_dir=source)
            file.unlink()
            with self.assertRaises(FileNotFoundError):
                agents.packet("reflector",self.root/(field+"-missing"),source_dir=source)
            file.write_bytes(original)

    def test_quick_test_is_optional_and_only_heldout(self):
        state=workflow.init_run("quick",self.index,self.snap["path"],root=self.root,mode="quick")
        run=Path(state["directory"])
        test=workflow.prepare_test(run,round_number=0)
        self.assertEqual(test["expected_generations"],132)
        task = Path(next(iter(test["outputs"]["42"].values()))) / "task.json"
        original = task.read_bytes()
        task.unlink()  # A crash during preparation can leave only the plan.
        self.assertEqual(workflow.prepare_test(run,round_number=0),test)
        self.assertEqual(task.read_bytes(),original)
        heldout={row["task_id"] for row in data.tasks(self.index,"validation") if row["view"]=="perspective-left"}
        self.assertEqual(set(test["task_ids"]),heldout)
        batch=workflow.begin_round(run)
        self.assertEqual(len(batch["sources"]),11)
        fixture_report(test["directory"])
        self.assertEqual(workflow.register_test(run,test["directory"])["test"]["round"],0)


    def test_full_test_completion_registers_after_agent_advances(self):
        state=workflow.init_run("quick",self.index,self.snap["path"],root=self.root,mode="quick")
        run=Path(state["directory"])
        write_json(run/"run.json",state)
        quick=workflow.prepare_test(run,round_number=0)
        fixture_report(quick["directory"])
        workflow.register_test(run,quick["directory"])
        full=evaluation.extend(quick["directory"],self.root/"full-test")
        fixture_report(full["directory"])
        changed=self.root/"other-manuals";shutil.copytree(ROOT/"manuals",changed)
        (changed/"operations.md").write_text("# Updated fixture operations\n")
        advanced=snapshots.freeze(manuals=changed)
        state=read_json(run/"run.json");state["current_snapshot"]=advanced["path"];write_json(run/"run.json",state)
        result=workflow.register_test(run,full["directory"])
        self.assertEqual(result["test"]["mode"],"full")
        self.assertEqual([r["mode"] for r in read_json(run/"run.json")["tests"]],["quick","full"])
        self.assertNotIn("pending_test",reports.summarize(run))

    def test_round_selection_keeps_rejected_candidate_and_active_harness_distinct(self):
        run, batch = self.run_start()
        manuals = self.root / "alternate-manuals"
        shutil.copytree(ROOT / "manuals", manuals)
        (manuals / "operations.md").write_text("# Alternate operations\n")
        candidate = snapshots.derive(self.snap["path"], manuals=manuals)
        state = read_json(run / "run.json")
        state["rounds"][0]["candidate"] = {"snapshot_path": candidate["path"], "snapshot_id": candidate["snapshot_id"]}
        state["rounds"][0]["decision"] = {"accepted": False}
        write_json(run / "run.json", state)
        rejected = workflow.prepare_test(run, "quick", round_number=1)
        active = workflow.prepare_test(run, "quick", round_number=1, target="active")
        self.assertEqual(rejected["snapshot_id"], candidate["snapshot_id"])
        self.assertEqual(active["snapshot_id"], self.snap["snapshot_id"])
        self.assertNotEqual(rejected["key"], active["key"])
        # No pending test blocks the next learning round.
        with patch("worldrsi.learning.accepted_bundle", return_value=None):
            self.assertEqual(workflow.begin_round(run)["round"], 2)
        fixture_report(rejected["directory"])
        self.assertEqual(workflow.register_test(run, rejected["directory"])["test"]["snapshot_id"], candidate["snapshot_id"])
        with self.assertRaises(ValueError):
            workflow.prepare_test(run, round_number=99)
        with self.assertRaises(ValueError):
            workflow.prepare_test(run, round_number=2, target="active")

    def test_promotion_rejects_cross_round_baseline_and_wrong_snapshot(self):
        run, batch = self.run_start()
        ids = [q["task_id"] for q in batch["sources"]]
        attempts = [Path(q["directory"]) / "initial" for q in batch["sources"]]
        for attempt in attempts:
            fixture_report(attempt)
        plan = evaluation.prepare(run / "comparison", self.index, self.snap["path"], "promotion",
                                  task_ids=ids, boundary={"round": 1}, learning_attempts=attempts)
        base = fixture_report(plan["directory"])
        other = copy.deepcopy(base)
        other["boundary"] = {"round": 2}
        with self.assertRaises(ValueError):
            evaluation.compare(other, base)
        with self.assertRaises(ValueError):
            evaluation.prepare(run / "missing", self.index, self.snap["path"], "promotion",
                               task_ids=ids[:-1], boundary={"round": 1})
        manuals = self.root / "different-manuals"
        shutil.copytree(ROOT / "manuals", manuals)
        (manuals / "operations.md").write_text("# Different\n")
        different = snapshots.derive(self.snap["path"], manuals=manuals)
        with self.assertRaisesRegex(ValueError, "same Agent"):
            evaluation.prepare(run / "wrong", self.index, different["path"], "promotion",
                               task_ids=ids, boundary={"round": 1}, learning_attempts=attempts)

    def test_learning_rejects_a_different_run_protocol_without_creating_rounds(self):
        state = workflow.init_run("old", self.index, self.snap["path"], root=self.root)
        run = Path(state["directory"])
        state["protocol"] = "physics-iq-cross-view-v1"
        write_json(run / "run.json", state)
        before = (run / "run.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "run protocol"):
            workflow.begin_round(run)
        with self.assertRaises(ValueError):
            workflow.history(run)
        self.assertFalse((run / "rounds").exists())
        self.assertEqual((run / "run.json").read_bytes(), before)


if __name__ == "__main__": unittest.main()
