from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from PIL import Image

from worldrsi import h3, media, records
from worldrsi.common import ROOT, digest, read_json, sha256, write_json


class H3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/"runs").mkdir(exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(prefix="h3-contract-",dir=ROOT/"runs")
        cls.root=Path(cls.temp.name)
        cls.video=cls.root/"proxy.mp4"
        subprocess.run([media.FFMPEG,"-v","error","-f","lavfi","-i","color=c=white:size=832x480:rate=24",
                        "-frames:v","120","-c:v","libx264","-pix_fmt","yuv420p",str(cls.video)],check=True)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.task=self.root/self._testMethodName
        self.package=self.task/"package";self.package.mkdir(parents=True)
        Image.new("RGB",(832,480),"white").save(self.package/"first_frame.png")
        (self.package/"prompt.txt").write_text("<Picture 1> <Video 1> subject_definitions summary retention_analysis detailed_description overall_soundscape non_diegetic_music")
        shutil.copy2(self.video,self.package/"proxy.mp4")
        write_json(self.package/"request.json",{"schema_version":2,"proxy_videos":[{"path":"proxy.mp4","modality":"semantic"}]})
        public={"task_id":"fixture","split":"learning","valid_frames":120,"generation_frames":120,"valid_duration":5,"fps":24,
                "width":832,"height":480,"input_hashes":{"first_frame":sha256(self.package/"first_frame.png")}}
        write_json(self.task/"task.json",{"public":public,"phase":"learning","generation_seed":42})
        camera={"camera_transform":[1,0,0,0],"current":True}
        trajectory=self.task/"trajectory.json";cameras=self.task/"camera.json"
        write_json(trajectory,{"fps":24,"frames":[{"frame":i,"time":i/24,"nodes":{"camera":camera}} for i in range(120)]})
        write_json(cameras,[camera]*120)
        binding={"trajectory_sha256":sha256(trajectory),"camera_sha256":sha256(cameras),"timeline_sha256":digest({"frames":120,"fps":24,"start_seconds":0})}
        self.provenance={"trajectory":str(trajectory),"camera":str(cameras),"videos":{"proxy.mp4":{**binding,"sha256":sha256(self.package/"proxy.mp4")}}}

    def completion(self,intent):
        out=Path(intent["output_path"]);out.mkdir(parents=True)
        shutil.copy2(self.video,out/"video.mp4")
        shutil.copytree(self.task/"h3-inputs",out/"inputs")
        frames=120+(5-120)%17
        done={"task_id":intent["queue_id"],"inputs":{"schema_version":2,"references":intent["spec"]["references"]},
              "metadata":{**intent["expected"],"reference_videos":[{**v,"model_frames":frames} for v in intent["spec"]["media"]]}}
        write_json(out/"complete.json",done)
        return out,done

    def test_real_package_collection_and_usage_resume(self):
        intent=h3.submit(self.task,self.package,self.provenance,self.root/"queue")
        self.assertEqual(h3.collect(self.task)["status"],"pending")
        self.assertEqual(h3.submit(self.task,self.package,self.provenance,self.root/"queue"),intent)
        self.completion(intent)
        collected=h3.collect(self.task)
        self.assertEqual(collected["output"]["media"]["frames"],120)
        self.assertEqual(h3.collect(self.task),collected)
        self.assertEqual(records.total_cost([self.task])["totals"]["h3_generations"],1)

    def test_completion_metadata_preserves_pending_state_until_correct(self):
        intent=h3.submit(self.task,self.package,self.provenance,self.root/"queue")
        out,done=self.completion(intent)
        done["metadata"]["seed"]=43;write_json(out/"complete.json",done)
        with self.assertRaises(ValueError):h3.collect(self.task)
        self.assertFalse((self.task/"output.lock.json").exists())
        done["metadata"]["seed"]=42;write_json(out/"complete.json",done)
        self.assertEqual(h3.collect(self.task)["status"],"locked")


if __name__=="__main__":unittest.main()
