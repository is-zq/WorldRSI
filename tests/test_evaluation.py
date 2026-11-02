import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from worldrsi import evaluation, h3, snapshots
from worldrsi.common import ROOT, read_json, sha256, write_json
from test_cross_view import fixture_index


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix="evaluation-test-",dir=ROOT/"runs")
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.index=fixture_index(self.root)
        self.snap=snapshots.freeze()

    def package(self,plan):
        tid=plan["task_ids"][0]
        primary=Path(plan["outputs"]["42"][tid])
        p=primary/"package";p.mkdir()
        (p/"first_frame.png").write_bytes(b"first-frame")
        (p/"prompt.txt").write_text("conditioning")
        write_json(p/"request.json",{"schema_version":2,"proxy_videos":[{"path":"proxy.mp4","modality":"semantic"}]})
        write_json(primary/"provenance.json",{})
        return tid,primary

    @staticmethod
    def validated(package,task,provenance):
        package=Path(package)
        return {"manifest":read_json(package/"request.json"),"files":{p.name:sha256(p) for p in package.iterdir() if p.is_file()},
                "binding":{"trajectory_sha256":"trajectory","camera_sha256":"camera"},"media":[],"references":[]}

    def test_seed_fanout_queue_routing_and_resume(self):
        for phase,count in (("validation",396),("test",792)):
            plan=evaluation.prepare(self.root/phase,self.index,self.snap["path"],phase)
            self.assertEqual(plan["expected_generations"],count)
            tid,primary=self.package(plan)
            with patch("worldrsi.h3.validate_package",side_effect=self.validated):
                first=evaluation.submit(plan["directory"],tid,queue=self.root/"queue")
                self.assertEqual(evaluation.submit(plan["directory"],tid,queue=self.root/"queue"),first)
            for seed in evaluation.SEEDS:
                task=Path(plan["outputs"][str(seed)][tid])
                intent=read_json(task/"h3.submission.json")
                self.assertEqual(read_json(task/"h3-inputs/request.json")["seed"],seed)
                self.assertEqual(Path(intent["input_path"]).parent.name,"worldrsi-test-inputs" if phase=="test" else "inputs")
                self.assertEqual(intent["expected"]["seed"],seed)
                write_json(task/"h3-output/complete.json",{"task_id":intent["queue_id"],"metadata":{"seed":seed+1}})
                with self.assertRaises(ValueError):evaluation.verify_seed(task,seed,completed=True)
            self.assertEqual(read_json(primary/"package/request.json").get("seed"),None)

    def test_finalize_fixed_denominator_and_per_seed_statistics(self):
        plan=evaluation.prepare(self.root/"validation",self.index,self.snap["path"],"validation")
        for seed,rows in plan["outputs"].items():
            for tid,d in rows.items():write_json(Path(d)/"scoring/score.json",{"value":float(seed)-42})
        def scored(d):
            c=read_json(Path(d)/"task.json")
            return {"valid_output":False,"score":0 if c["public"]["task_id"]==plan["task_ids"][0] else (c["generation_seed"]-41)/10}
        with patch("worldrsi.data.validate_index",return_value={}),patch("worldrsi.evaluation.score",side_effect=scored):
            result=evaluation.finalize(plan["directory"])
        self.assertEqual(result["count"],99)
        self.assertAlmostEqual(result["mean"],0.25*98/99)
        self.assertGreater(result["sample_std"],0)
        self.assertEqual(evaluation.report(plan["directory"]),result)
        path=Path(plan["outputs"]["42"][plan["task_ids"][0]])/"scoring/score.json"
        write_json(path,{"changed":True})
        with self.assertRaises(ValueError):evaluation.report(plan["directory"])

    def test_seed_and_phase_plan_identity(self):
        plan=evaluation.prepare(self.root/"test",self.index,self.snap["path"],"test",boundary={"after_round":3})
        with self.assertRaises(ValueError):
            evaluation.prepare(self.root/"test",self.index,self.snap["path"],"test",boundary={"after_round":6})
        self.assertEqual(plan["seeds"],[42,43,44,45])
        self.assertEqual(evaluation.generation_seed({"phase":"learning","generation_seed":42}),42)
        with self.assertRaises(ValueError):evaluation.generation_seed({"phase":"learning","generation_seed":43})


if __name__=="__main__":unittest.main()
