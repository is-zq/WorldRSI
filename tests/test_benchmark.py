from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from worldrsi import benchmark, evaluation, media, scoring
from worldrsi.common import ROOT, digest, read_json, sha256, write_json


def video(path, mask=False, shift=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),24,(64,48))
    if not writer.isOpened(): raise RuntimeError("Fixture codec is required")
    for i in range(120):
        frame=np.full((48,64,3),0 if mask else 40,dtype=np.uint8)
        if i:
            x=4+(i//3)%38+shift
            frame[18:30,x:x+12]=255 if mask else 230
        writer.write(frame)
    writer.release()


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/"runs").mkdir(exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(prefix="official-scoring-",dir=ROOT/"runs")
        cls.root=Path(cls.temp.name)
        source=cls.root/"source"
        (source/"descriptions").mkdir(parents=True)
        (source/"physiq").mkdir()
        for name in ("binary_mask_generator.py","calculate_and_write_metrics_to_csv.py","calculate_iq_score.py","calculate_iq_score_stable.py"):
            shutil.copy2(benchmark.BENCHMARK_ROOT/"physiq"/name,source/"physiq"/name)
        scene={"scene":"fixture","category":"Solid Mechanics","description":"A block moves.",
               "takes":[{"take":1,"left":1,"center":2,"right":3},{"take":2,"left":4,"center":5,"right":6}]}
        (source/"descriptions/data.yaml").write_text(yaml.safe_dump({"scenes":[scene]}))
        cls.data=source/"physics-IQ-benchmark-verified"
        (cls.data/"first-frames").mkdir(parents=True)
        for take in (1,2):
            for i,view in enumerate(benchmark.VIEWS):
                number=(take-1)*3+i+1
                if take==1:
                    cv2.imwrite(str(cls.data/"first-frames"/f"{number:04d}_first-frames_24FPS_{view}_take-1_trimmed-fixture.png"),np.full((48,64,3),70,dtype=np.uint8))
                for mask,folder,kind in ((False,"split-videos/testing","testing-videos"),(True,"video-masks/real","video-masks")):
                    path=cls.data/folder/"24FPS"/f"{number:04d}_{kind}_24FPS_{view}_take-{take}_trimmed-fixture.mp4"
                    video(path,mask,take-1)
        cls.manifest=benchmark.build_index(cls.root/"index",source_root=source,expected_scenes=1)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def output(self,phase,valid=True):
        d=self.root/(self._testMethodName+phase)
        d.mkdir()
        p={**self.manifest["binding"]["public"][0],"split":phase,"data_revision":self.manifest["revision"],
           "scene_id":"fixture","view":"perspective-left"}
        private=self.manifest["binding"]["private"][0]
        write_json(d/"task.json",{"public":p,"phase":phase,"generation_seed":42,"snapshot_id":"fixture-snapshot",
            "roles":read_json(ROOT/"config/runtime.json")["roles"],"intervention":None})
        if valid:
            media.normalize(private["gt_v1"],d/"generated.mp4",120)
            media.lock_output(d,d/"generated.mp4",p)
        else: media.lock_failure(d,p,"Fixture failed to produce video")
        return d,private

    def test_index_binds_first_frames_and_three_views(self):
        self.assertEqual(len(self.manifest["binding"]["public"]),3)
        self.assertEqual(self.manifest["status"],"validated")
        for row in self.manifest["binding"]["public"]:
            self.assertEqual(Path(row["first_frame"]).parent,self.data/"first-frames")
            self.assertIn("_take-1_",row["first_frame"])
        self.assertEqual({p["view"] for p in self.manifest["binding"]["private"]},set(benchmark.VIEWS))

    def test_reuse_bound_source_validation_and_detect_implementation_changes(self):
        binding={"implementation":self.manifest["binding"]["implementation"],"sources":{}}
        for row in self.manifest["binding"]["private"]:
            for key in ("gt_v1","gt_v2","mask_v1","mask_v2"):
                binding["sources"][row[key]]={"sha256":row["sha256"][key],"media":row["media"][key]}
        target=self.root/"reused-index"
        write_json(target/"source-validation.json",{"binding":binding,"revision":digest(binding)})
        with patch("worldrsi.benchmark._source_media",side_effect=AssertionError("Matching sources use validated metadata")):
            result=benchmark.build_index(target,source_root=self.root/"source",expected_scenes=1)
        self.assertEqual(result["status"],"validated")
        one_source=next(iter(binding["sources"]))
        binding["sources"][one_source]["sha256"]="0"*64
        refreshed=self.root/"refreshed-index"
        write_json(refreshed/"source-validation.json",{"binding":binding,"revision":digest(binding)})
        with patch("worldrsi.benchmark._source_media",wraps=benchmark._source_media) as decode:
            benchmark.build_index(refreshed,source_root=self.root/"source",expected_scenes=1)
        self.assertEqual(decode.call_count,1)
        self.assertEqual(str(decode.call_args.args[0]),one_source)
        module=self.root/"source/physiq/binary_mask_generator.py"
        original=module.read_bytes()
        try:
            module.write_bytes(original+b"\n# fixture change\n")
            with self.assertRaises(ValueError):benchmark.validate_index(target,decode=False)
        finally:module.write_bytes(original)

    def test_all_phases_use_official_components_and_bound_score(self):
        values=[]
        for phase in ("learning","transfer","validation","test"):
            d,private=self.output(phase)
            component=benchmark.measure_view(d,private)
            self.assertEqual(component["status"],"measured")
            path=d/"scoring/physics_iq/components.json"
            session_id="/root/scorer-fixture"
            result=benchmark.finalize_scorer(d,private,{"session_id":session_id, "components":str(path)})
            self.assertEqual(result["session_id"],session_id)
            self.assertEqual(result["observations"],[])
            self.assertEqual(result["uncertainties"],[])
            values.append(result["score"])
            self.assertEqual(evaluation.score(d),result)
            self.assertGreaterEqual(result["score"],0)
            self.assertLessEqual(result["score"],1)
            sampled=scoring.evidence(d,d/"generated.mp4",[0,119],"Check initial and final positions")
            self.assertEqual(len(sampled["images"]),2)
        self.assertEqual(len(set(values)),1)

    def test_failed_output_has_fixed_zero_credit(self):
        d,private=self.output("validation",False)
        benchmark.measure_view(d,private)
        result=benchmark.finalize_scorer(d,private,{"session_id":"fixture-scorer","components":str(d/"scoring/physics_iq/components.json")})
        self.assertEqual(result["score"],0)
        self.assertEqual(evaluation.score(d)["score"],0)

    def test_score_and_intervention_binding_are_checked(self):
        d,private=self.output("transfer")
        benchmark.measure_view(d,private)
        path=d/"scoring/physics_iq/components.json"
        result=benchmark.finalize_scorer(d,private,{"session_id":"fixture-scorer","components":str(path),
                "observations":[{"observation":"Measured", "evidence":[str(path)]}]})
        write_json(d/"scoring/score.json",{**result,"score":-0.1})
        with self.assertRaises(ValueError):evaluation.score(d)
        write_json(d/"scoring/score.json",result)
        context=read_json(d/"task.json");context["intervention"]={"instruction":"changed"};write_json(d/"task.json",context)
        with self.assertRaises(ValueError):evaluation.score(d)


if __name__=="__main__":unittest.main()
