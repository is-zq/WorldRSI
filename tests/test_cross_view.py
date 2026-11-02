import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from worldrsi import agents, data, evaluation, learning, snapshots, workflow
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
    private = {p: [{"task_id": r["task_id"]} for r in rows] for p, rows in public.items()}
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
        plan = workflow.initial_reference(run)
        fixture_report(plan["directory"])
        workflow.set_reference(run)
        return run, workflow.begin_round(run)

    def scored_source(self, source, initial, best):
        source = Path(source)
        def read_score(path):
            return read_json(Path(path) / "scoring/score.json")
        write_json(source / "initial/scoring/score.json", {"score": initial})
        with patch("worldrsi.evaluation.score", side_effect=read_score):
            branch = learning.branch(source, {"id": "improve", "hypothesis": "Different representation helps",
                "prediction": "More accurate contact", "executor_instruction": "Use continuous collision geometry"})
        d = Path(branch["directory"])
        write_json(d / "scoring/score.json", {"score": best})
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
        write_json(source/"branches/improve/scoring/score.json",{"score":0.5})
        with self.assertRaises(ValueError):learning.prepare_transfer(source)

    def test_transfer_pairs_and_privacy_projection(self):
        run, batch = self.run_start()
        source, record, read_score = self.scored_source(batch["sources"][0]["directory"],0.2,0.4)
        with patch("worldrsi.evaluation.score", side_effect=read_score):
            closed = learning.close(source,record)
        plans = learning.prepare_transfer(source)
        self.assertEqual([p["expected_generations"] for p in plans.values()],[8,8])
        a,b = [fixture_report(p["directory"],v) for p,v in zip(plans.values(),[0.3,0.4])]
        with self.assertRaises(FileNotFoundError): learning.finalize_transfer(source)
        for tid,d in b["outputs"]["42"].items():
            p=Path(d); (p/"checked.txt").write_text("Target-view implementation checked")
            write_json(p/"implementation.json",{"experience_sha256":digest(closed),"applied":True,"assessment":"Verified collider change",
                       "evidence":[str(p/"checked.txt")]})
        result = learning.finalize_transfer(source)
        self.assertTrue(result["passed"])
        public = learning.accepted_bundle(source)
        self.assertEqual(public["transfer"]["passed"],True)
        self.assertNotIn("gains",public["transfer"])
        self.assertNotIn("reports",public["transfer"])
        self.assertEqual(learning.finalize_transfer(source),result)
        with self.assertRaises(ValueError):
            agents.packet("executor",self.root/"replica-executor",task_dir=next(iter(b["outputs"]["43"].values())))
        proof_path=Path(next(iter(b["outputs"]["42"].values())))/"implementation.json"
        changed=read_json(proof_path);changed["applied"]=False;write_json(proof_path,changed)
        with self.assertRaises(ValueError): learning.accepted_bundle(source)

    def test_both_reserved_view_means_must_improve(self):
        base={"task_ids":["left","right"],"seeds":evaluation.SEEDS,"snapshot_id":"s","data_revision":"d","scorer":{},
              "scores":{str(s):{"left":0.4,"right":0.4} for s in evaluation.SEEDS}}
        for left,right,expected in [(0.6,0.41,True),(0.9,0.39,False),(0.9,0.4,False),(0.4,0.4,False)]:
            candidate=copy.deepcopy(base)
            candidate["scores"]={str(s):{"left":left,"right":right} for s in evaluation.SEEDS}
            self.assertEqual(learning.transfer_gains(base,candidate)[1],expected)
        candidate=copy.deepcopy(base)
        for s in evaluation.SEEDS: candidate["scores"][str(s)]={"left":0.6,"right":0.6}
        candidate["scores"]["42"]={"left":0.3,"right":0.3}
        self.assertTrue(learning.transfer_gains(base,candidate)[1])

    def test_pending_boundaries_and_patience(self):
        for count,streak,expected in [(1,1,None),(3,3,"epoch"),(5,5,"partial_epoch"),(6,5,"epoch"),(12,0,"epoch")]:
            state={"rounds":[{"decision":{}} for _ in range(count)],"rejection_streak":streak,"tests":[]}
            due=workflow.pending_test(state)
            self.assertEqual(due and due["label"],expected)
            if due:
                state["tests"].append(due)
                self.assertIsNone(workflow.pending_test(state))

    def test_cost_tie_and_incompatible_evaluations(self):
        run,_=self.run_start()
        base=evaluation.report(run/"initial-validation")
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

    def test_eleven_source_barrier_manual_only_candidate_and_atomic_rejection(self):
        run,batch=self.run_start()
        with self.assertRaises(FileNotFoundError): workflow.manual_inputs(run)
        for item in batch["sources"]:
            d=Path(item["directory"])
            e={"task_id":item["task_id"],"status":"stalled","manual_updates":[]}
            write_json(d/"experience.json",e)
            write_json(d/"transfer/decision.json",{"passed":False,"experience_sha256":digest(e)})
        inputs=workflow.manual_inputs(run)
        shutil.copytree(inputs["base_manuals"],inputs["output_manuals"])
        Path(inputs["merge_notes"]).write_text("Retained checked operations.")
        workflow.publish_manuals(run)
        packet=self.root/"editor"
        agents.packet("editor",packet,run_dir=run)
        self.assertEqual(read_json(packet/"input.json")["experiences"],[])
        c=workflow.candidate(run)
        self.assertEqual(c["snapshot_id"],self.snap["snapshot_id"])
        decision=workflow.decide(run)
        self.assertFalse(decision["accepted"])
        state=read_json(run/"run.json")
        self.assertEqual(state["current_snapshot"],self.snap["path"])
        self.assertEqual(state["rejection_streak"],1)
        self.assertEqual(workflow.history(run)["bundles"],[])

    def test_accepted_candidate_activates_complete_snapshot_and_next_batch_history(self):
        run,batch=self.run_start()
        old_history=read_json(Path(batch["directory"])/"history.json")
        for item in batch["sources"]:
            d=Path(item["directory"])
            e={"task_id":item["task_id"],"status":"stalled","manual_updates":[]}
            write_json(d/"experience.json",e)
            write_json(d/"transfer/decision.json",{"passed":False,"experience_sha256":digest(e)})
        inputs=workflow.manual_inputs(run)
        shutil.copytree(inputs["base_manuals"],inputs["output_manuals"])
        (Path(inputs["output_manuals"])/"operations.md").write_text("# Checked operational guidance\nA fixture operation.\n")
        Path(inputs["merge_notes"]).write_text("Merged checked operational evidence.")
        workflow.publish_manuals(run)
        c=workflow.candidate(run)
        self.assertNotEqual(c["snapshot_id"],self.snap["snapshot_id"])
        fixture_report(c["evaluation"],value=0.5)
        self.assertTrue(workflow.decide(run)["accepted"])
        state=read_json(run/"run.json")
        self.assertEqual(state["current_snapshot"],c["snapshot_path"])
        self.assertEqual(read_json(state["reference"]["path"])["mean"],0.5)
        next_batch=workflow.begin_round(run)
        self.assertEqual(next_batch["snapshot_path"],c["snapshot_path"])
        self.assertEqual(read_json(Path(batch["directory"])/"history.json"),old_history)
        self.assertEqual(len(read_json(Path(next_batch["directory"])/"history.json")["validation"]),1)

    def test_quick_run_starts_learning_while_initial_test_is_pending(self):
        state=workflow.init_run("quick",self.index,self.snap["path"],root=self.root,mode="quick")
        run=Path(state["directory"])
        test=workflow.prepare_initial_test(run)
        self.assertEqual(test["expected_generations"],264)
        val=workflow.initial_reference(run)
        self.assertEqual(val["expected_generations"],132)
        fixture_report(val["directory"])
        workflow.set_reference(run)
        batch=workflow.begin_round(run)
        self.assertEqual(len(batch["sources"]),11)
        fixture_report(test["directory"])
        self.assertEqual(workflow.register_initial_test(run)["seeds"],[42,43,44,45])

    def test_full_test_completion_registers_after_agent_advances(self):
        state=workflow.init_run("quick",self.index,self.snap["path"],root=self.root,mode="quick")
        run=Path(state["directory"])
        state["rounds"]=[{"decision":{"accepted":False}} for _ in range(3)]
        write_json(run/"run.json",state)
        quick=workflow.prepare_test(run)
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
        self.assertIsNone(workflow.pending_test(read_json(run/"run.json")))


if __name__ == "__main__": unittest.main()
