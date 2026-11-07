import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldrsi import agents, evaluation, learning, snapshots, workflow
from worldrsi.common import ROOT, read_json, write_json
from test_cross_view import fixture_index, fixture_report


class PromotionRevisionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='revision-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.index = fixture_index(self.root)
        validation = patch('worldrsi.data.validate_index', return_value={})
        validation.start()
        self.addCleanup(validation.stop)
        self.snap = snapshots.freeze()
        state = workflow.init_run('revision', self.index, self.snap['path'], root=self.root)
        self.run = Path(state['directory'])
        self.round = workflow.begin_round(self.run)
        for source in self.round['sources']:
            d = Path(source['directory'])
            fixture_report(d / 'initial')
            learning.close(d, {'status': 'stalled', 'best_attempt': 'initial', 'reason': 'Fixture'})
        inputs = workflow.manual_inputs(self.run)
        shutil.copytree(inputs['base_manuals'], inputs['output_manuals'])
        Path(inputs['merge_notes']).write_text('Checked unchanged manuals.')
        workflow.publish_manuals(self.run)

    def candidate(self, number, score=None):
        folder = self.root / f'editor-{number}'
        policy = folder / 'policy'
        shutil.copytree(Path(self.snap['path']) / 'policy', policy)
        path = policy / snapshots.DECISION_EXPERIENCE
        path.write_text(path.read_text() + f'\nRevision {number}: preserve conditional lessons.\n')
        (folder / 'revision.md').write_text(f'Revision {number} based on measured regression.')
        c = workflow.candidate(self.run, policy=policy)
        if not (Path(c['baseline_evaluation']) / 'report.json').exists():
            fixture_report(c['baseline_evaluation'])
        if score is not None:
            fixture_report(c['evaluation'], value=score)
        return c, policy

    def test_three_revisions_then_one_rejection_and_idempotence(self):
        baseline = None
        for number in range(1, 5):
            c, policy = self.candidate(number, 0.3)
            baseline = baseline or c['baseline_evaluation']
            self.assertEqual(c['baseline_evaluation'], baseline)
            result = workflow.decide(self.run)
            self.assertEqual(workflow.decide(self.run), result)
            self.assertEqual(workflow.candidate(self.run, policy=policy)['attempt'], number)
            state = read_json(self.run / 'run.json')
            self.assertEqual(len(state['rounds'][0]['promotion_attempts']), number)
            self.assertEqual(state['current_snapshot'], self.snap['path'])
            if number < 4:
                self.assertEqual(result['status'], 'revision_required')
                self.assertEqual(result['remaining_revisions'], 4-number)
                self.assertIsNone(state['rounds'][0]['decision'])
                self.assertEqual(state['rejection_streak'], 0)
                self.assertEqual(workflow.begin_round(self.run)['round'], 1)
                self.assertFalse((Path(self.round['directory']) / 'decision.json').exists())
            else:
                self.assertEqual(result['status'], 'rejected')
                self.assertEqual(state['rejection_streak'], 1)
        with self.assertRaisesRegex(ValueError, 'retries remaining'):
            self.candidate(5)
        self.assertEqual(len(workflow.history(self.run)['promotion'][0]['promotion_attempts']), 4)

    def test_success_after_revision_activates_complete_candidate(self):
        self.candidate(1, 0.3)
        workflow.decide(self.run)
        c, _ = self.candidate(2, 0.5)
        result = workflow.decide(self.run)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['attempt'], 2)
        state = read_json(self.run / 'run.json')
        self.assertEqual(state['current_snapshot'], c['snapshot_path'])
        self.assertEqual(state['rejection_streak'], 0)
        self.assertEqual(workflow.begin_round(self.run)['snapshot_path'], c['snapshot_path'])

    def test_pending_does_not_consume_revision_and_feedback_is_bound(self):
        c, _ = self.candidate(1)
        before = (self.run / 'run.json').read_bytes()
        with self.assertRaises(FileNotFoundError):
            workflow.decide(self.run)
        self.assertEqual((self.run / 'run.json').read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'completed score decline'):
            agents.packet('editor', self.root / 'early', run_dir=self.run)
        fixture_report(c['evaluation'], value=0.3)
        decision = workflow.decide(self.run)
        agents.packet('editor', self.root / 'editor-packet', run_dir=self.run)
        payload = read_json(self.root / 'editor-packet/input.json')
        self.assertEqual(payload['attempt'], 2)
        self.assertEqual(payload['policy'], str(Path(c['snapshot_path']) / 'policy'))
        inputs = payload['promotion_feedback'][0]['comparison']['inputs']
        self.assertEqual(len(inputs), 11)
        self.assertEqual(set(inputs[0]['candidate']['per_seed']), {'42', '43', '44', '45'})
        self.assertNotIn('tests', payload)
        file = Path(decision['feedback']['path'])
        file.write_text(file.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'feedback changed'):
            agents.packet('editor', self.root / 'tampered', run_dir=self.run)

    def test_tie_and_cost_rejections_are_final(self):
        c, _ = self.candidate(1, 0.4)
        self.assertEqual(workflow.decide(self.run)['reason'], 'equivalent')
        state = read_json(self.run / 'run.json')
        self.assertEqual(state['rejection_streak'], 1)
        self.assertFalse(workflow.revision_required(state['rounds'][0]))

    def test_test_target_is_bound_to_attempt_after_new_candidate(self):
        first, _ = self.candidate(1, 0.3)
        test = workflow.prepare_test(self.run, 'quick', round_number=1)
        fixture_report(test['directory'])
        workflow.decide(self.run)
        second, _ = self.candidate(2, 0.5)
        latest = workflow.prepare_test(self.run, 'quick', round_number=1)
        self.assertEqual(latest['boundary']['attempt'], 2)
        self.assertEqual(latest['snapshot_id'], second['snapshot_id'])
        self.assertEqual(workflow.prepare_test(self.run, 'quick', round_number=1, attempt=1), test)
        receipt = workflow.register_test(self.run, test['directory'])['test']
        self.assertEqual(receipt['snapshot_id'], first['snapshot_id'])
        with self.assertRaises(ValueError):
            workflow.prepare_test(self.run, round_number=1, target='active', attempt=1)

    def test_cost_failure_and_epsilon_do_not_trigger_revisions(self):
        c, _ = self.candidate(1)
        fixture_report(c['evaluation'], value=0.4-evaluation.EPSILON/2, tokens=300)
        result = workflow.decide(self.run)
        self.assertEqual(result['reason'], 'cost')
        self.assertEqual(result['status'], 'rejected')

    def test_legacy_test_report_registers_after_revision(self):
        first, _ = self.candidate(1, 0.3)
        legacy = evaluation.prepare(self.root / 'legacy-test', self.index, first['snapshot_path'],
                                    'test', mode='quick', boundary={'run_id': 'revision', 'round': 1, 'target': 'candidate'})
        fixture_report(legacy['directory'])
        workflow.decide(self.run)
        self.candidate(2, 0.5)
        self.assertEqual(workflow.register_test(self.run, legacy['directory'])['test']['snapshot_id'], first['snapshot_id'])
