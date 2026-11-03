from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from worldrsi import agents, data, evaluation, scoring, sessions, snapshots, workflow, learning
from worldrsi.common import ROOT, read_json, write_json
from worldrsi.media import lock_failure
from test_cross_view import fixture_index, fixture_report


class ScorerProtocolTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="scorer-test-", dir=ROOT / "runs")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.index = fixture_index(self.root)
        self.snap = snapshots.freeze()
        check = patch("worldrsi.data.validate_index", return_value={"status": "passed"})
        check.start()
        self.addCleanup(check.stop)
        state = workflow.init_run("run", self.index, self.snap["path"], root=self.root, mode="quick")
        self.run = Path(state["directory"])

    def packet(self, plan, name):
        packet = self.root / name
        spec = agents.packet("scorer", packet, run_dir=self.run, evaluation_dir=plan["directory"])
        sessions.register(packet, spec["arguments"]["target"] if spec["method"] == "followup_task" else "scorer-" + name,
                          spec["model"], spec["reasoning_effort"])
        return packet, spec

    def lock_outputs(self, plan):
        for rows in plan["outputs"].values():
            for d in rows.values():
                lock_failure(d, read_json(Path(d) / "task.json")["public"], "Fixture failed final output")

    def test_scorer_scope_and_high_configuration(self):
        test = workflow.prepare_initial_test(self.run)
        first, spec = self.packet(test, "first")
        self.assertEqual(spec["method"], "spawn_agent")
        self.assertEqual((spec["model"], spec["reasoning_effort"]), ("gpt-6.1-sol", "high"))
        _, followup = self.packet(test, "retry")
        self.assertEqual(followup["method"], "followup_task")
        self.assertEqual(followup["arguments"]["target"], "scorer-first")
        val = workflow.initial_reference(self.run)
        second, spec = self.packet(val, "second")
        self.assertEqual(spec["method"], "spawn_agent")
        self.assertNotEqual(read_json(first / "session.json")["session_id"], read_json(second / "session.json")["session_id"])
        fixture_report(val["directory"])
        workflow.set_reference(self.run)
        sources = workflow.begin_round(self.run)["sources"]
        for i, source in enumerate(sources[:2]):
            attempt = Path(source["directory"]) / "initial"
            plan = evaluation.load(attempt)
            packet, spec = self.packet(plan, "source-" + str(i))
            self.assertEqual(spec["method"], "spawn_agent")
            self.assertEqual(spec["session_scope"]["directory"], source["directory"])
            fixture_report(attempt)
            branch = learning.branch(source["directory"], {"id": "retry", "hypothesis": "Improve",
                                     "prediction": "Higher mean", "executor_instruction": "Change geometry"})
            _, followup = self.packet(evaluation.load(branch["directory"]), "branch-" + str(i))
            self.assertEqual(followup["method"], "followup_task")
            self.assertEqual(followup["arguments"]["target"], "scorer-source-" + str(i))

    def test_initial_validation_reuses_exact_test_scores_and_costs(self):
        val = workflow.initial_reference(self.run)
        test = workflow.prepare_initial_test(self.run)
        self.assertEqual(val["additional_generations"], 0)
        self.assertEqual(workflow.initial_reference(self.run), val)
        self.assertFalse((Path(val["directory"]) / "seeds").exists())
        for seed, rows in test["outputs"].items():
            for tid, d in rows.items():
                write_json(Path(d) / "scoring/score.json", {"fixture": True})
                write_json(Path(d) / "costs/fixture.json", {"event_id": "fixture", "kind": "scorer",
                           "values": {"local_cpu_seconds": 2}})
        with patch("worldrsi.evaluation.score", return_value={"valid_output": False, "score": 0.25}):
            report = evaluation.finalize(val["directory"])
        self.assertEqual(report["mean"], 0.25)
        self.assertEqual(report["reused_generations"], 132)
        self.assertEqual(report["fresh_generations"], 0)
        self.assertEqual(report["reused_costs"]["totals"]["local_cpu_seconds"], 264)
        self.assertEqual(len(report["evidence"]), 132)
        evaluation.report(val["directory"])
        d = next(iter(val["outputs"]["42"].values()))
        write_json(Path(d) / "scoring/score.json", {"changed": True})
        with self.assertRaises(ValueError):
            evaluation.report(val["directory"])
        with self.assertRaisesRegex(ValueError, "same Agent"):
            evaluation.prepare(self.run / "mismatch", self.index, self.snap["path"], "validation",
                               mode="quick", reuse_test=test["directory"], intervention={"other": True})

    def test_batch_preflight_resume_and_no_validation_remeasurement(self):
        test = workflow.prepare_initial_test(self.run)
        packet, _ = self.packet(test, "batch")
        with patch("worldrsi.benchmark.measure_view") as measure:
            with self.assertRaises(FileNotFoundError):
                scoring.batch(test["directory"], packet / "input.json")
            measure.assert_not_called()
        self.lock_outputs(test)
        count = 0

        def finish(task, private, judgment):
            nonlocal count
            count += 1
            if count == 3:
                raise RuntimeError("Transient scoring infrastructure failure")
            self.assertEqual(judgment["session_id"], "scorer-batch")
            self.assertEqual(set(judgment), {"session_id", "components"})
            write_json(Path(task) / "scoring/score.json", {"valid_output": False, "score": 0})

        with patch("worldrsi.benchmark.measure_view") as measure, \
             patch("worldrsi.benchmark.finalize_scorer", side_effect=finish), \
             patch("worldrsi.evaluation.score", side_effect=lambda d: read_json(Path(d) / "scoring/score.json")):
            with self.assertRaises(RuntimeError):
                scoring.batch(test["directory"], packet / "input.json")
            receipt = scoring.batch(test["directory"], packet / "input.json")
            self.assertEqual(len(receipt["scores"]), 264)
            self.assertEqual(measure.call_count, 265)
            self.assertEqual(scoring.batch(test["directory"], packet / "input.json"), receipt)
            self.assertEqual(measure.call_count, 265)
            val = workflow.initial_reference(self.run)
            val_packet, _ = self.packet(val, "val-batch")
            scoring.batch(val["directory"], val_packet / "input.json")
            self.assertEqual(measure.call_count, 265)

    def test_learning_mean_requires_all_seeds_and_reflector_gets_numbers(self):
        val = workflow.initial_reference(self.run)
        fixture_report(val["directory"])
        workflow.set_reference(self.run)
        source = Path(workflow.begin_round(self.run)["sources"][0]["directory"])
        attempt = source / "initial"
        plan = evaluation.load(attempt)
        self.assertEqual(plan["expected_generations"], 4)
        self.assertEqual(next(iter(plan["outputs"]["42"].values())), str(attempt))
        self.lock_outputs(plan)
        values = {"42": 0.9, "43": 0.5, "44": 0.2, "45": 0.0}
        for seed, rows in plan["outputs"].items():
            task = Path(next(iter(rows.values())))
            self.assertEqual(read_json(task / "task.json")["generation_seed"], int(seed))
            if seed != "45":
                write_json(task / "scoring/score.json", {"valid_output": False, "score": values[seed]})
        scored = lambda d: read_json(Path(d) / "scoring/score.json")
        with patch("worldrsi.evaluation.score", side_effect=scored):
            with self.assertRaises(FileNotFoundError):
                evaluation.finalize(attempt)
            self.assertEqual(learning.progress(source)["attempts"], {})
            last = Path(next(iter(plan["outputs"]["45"].values())))
            write_json(last / "scoring/score.json", {"valid_output": False, "score": 0})
            evaluation.finalize(attempt)
        progress = learning.progress(source)
        self.assertAlmostEqual(progress["initial_score"], 0.4)
        self.assertEqual(progress["attempts"]["initial"]["per_seed"], values)
        packet = self.root / "reflector"
        agents.packet("reflector", packet, source_dir=source)
        feedback = read_json(packet / "input.json")["attempts"]["initial"]["score"]
        self.assertEqual(feedback["per_seed"], values)
        self.assertNotIn("observations", feedback)
        self.assertNotIn("uncertainties", feedback)
        branch = learning.branch(source, {"id": "lower-best-higher-mean", "hypothesis": "Reduce variance",
                                 "prediction": "Improve mean", "executor_instruction": "Change geometry"})
        fixture_report(branch["directory"], value=0.6)
        self.assertEqual(learning.progress(source)["best_attempt"], "lower-best-higher-mean")
        write_json(last / "scoring/score.json", {"changed": True})
        with self.assertRaises(ValueError):
            learning.progress(source)

    def test_legacy_primary_is_preserved_when_adding_learning_seeds(self):
        public = data.tasks(self.index, "learning")[0]
        task = self.run / "legacy-primary"
        context = evaluation.create_task(task, public, self.snap)
        lock_failure(task, public, "Frozen legacy output")
        write_json(task / "scoring/score.json", {"legacy": True})
        before = (task / "output.lock.json").read_bytes()
        plan = evaluation.prepare_learning(task, self.index)
        self.assertEqual(plan["expected_generations"], 4)
        self.assertEqual(read_json(task / "task.json"), context)
        self.assertEqual(read_json(task / "scoring/score.json"), {"legacy": True})
        self.assertEqual((task / "output.lock.json").read_bytes(), before)
        self.assertEqual(evaluation.prepare_learning(task, self.index), plan)

    def test_full_extension_keeps_reused_quick_view(self):
        val = workflow.initial_reference(self.run)
        full = evaluation.extend(val["directory"], self.run / "full-validation")
        self.assertEqual(full["additional_generations"], 264)
        self.assertTrue(evaluation.is_completion(val, full))
        for seed in val["seeds"]:
            for tid, d in val["outputs"][str(seed)].items():
                self.assertEqual(full["outputs"][str(seed)][tid], d)


if __name__ == "__main__":
    unittest.main()
