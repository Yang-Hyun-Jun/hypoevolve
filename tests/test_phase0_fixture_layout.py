import unittest
from pathlib import Path
import json


class TestPhase0FixtureLayout(unittest.TestCase):
    def test_parity_fixture_directories_exist(self):
        root = Path('tests/fixtures/parity')
        self.assertTrue(root.exists())
        expected = {
            'single_worker_provided_hypothesis',
            'single_worker_generated_seed',
            'multi_worker_completed',
            'duplicate_skip',
            'steering_failure_skip',
            'evaluator_repair',
        }
        actual = {path.name for path in root.iterdir() if path.is_dir()}
        self.assertTrue(expected.issubset(actual))

    def test_parity_fixture_readme_documents_required_files(self):
        text = Path('tests/fixtures/parity/README.md').read_text(encoding='utf-8')
        for required in (
            'manifest.json',
            'input_config.json',
            'stubbed_llm_responses.json',
            'normalized_expected/',
        ):
            self.assertIn(required, text)

    def test_each_parity_scenario_has_manifest_placeholder(self):
        root = Path('tests/fixtures/parity')
        for scenario_dir in root.iterdir():
            if not scenario_dir.is_dir():
                continue
            manifest = scenario_dir / 'manifest.json'
            self.assertTrue(manifest.exists(), msg=f"missing manifest for {scenario_dir.name}")

    def test_each_parity_scenario_has_placeholder_support_files(self):
        root = Path('tests/fixtures/parity')
        for scenario_dir in root.iterdir():
            if not scenario_dir.is_dir():
                continue
            self.assertTrue((scenario_dir / 'input_config.json').exists())
            self.assertTrue((scenario_dir / 'stubbed_llm_responses.json').exists())
            self.assertTrue((scenario_dir / 'normalized_expected').is_dir())

    def test_each_manifest_declares_required_placeholder_contract(self):
        root = Path('tests/fixtures/parity')
        for scenario_dir in root.iterdir():
            if not scenario_dir.is_dir():
                continue
            payload = json.loads((scenario_dir / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['scenario'], scenario_dir.name)
            self.assertIn(payload['status'], {'placeholder', 'implemented'})
            self.assertEqual(payload['mode'], 'stubbed')
            self.assertIn('required_artifacts', payload)
            self.assertIn('normalization', payload)
            self.assertEqual(
                payload['required_artifacts'],
                ['manifest.json', 'input_config.json', 'stubbed_llm_responses.json'],
            )
            for normalization_key in (
                'run_id',
                'timestamps',
                'temp_paths',
                'executor_work_dir',
                'log_noise',
            ):
                self.assertIn(normalization_key, payload['normalization'])

    def test_support_files_match_scenario_name(self):
        root = Path('tests/fixtures/parity')
        for scenario_dir in root.iterdir():
            if not scenario_dir.is_dir():
                continue
            for filename in ('input_config.json', 'stubbed_llm_responses.json'):
                payload = json.loads((scenario_dir / filename).read_text(encoding='utf-8'))
                self.assertEqual(payload['scenario'], scenario_dir.name)
                self.assertEqual(payload['status'], 'placeholder')

    def test_implemented_scenarios_declare_expected_normalized_files(self):
        root = Path('tests/fixtures/parity')
        expected_files = {
            'duplicate_skip': ['best.json', 'checkpoint.json', 'run_summary.json', 'score_history.json', 'trace.json'],
            'steering_failure_skip': ['best.json', 'checkpoint.json', 'run_summary.json', 'score_history.json', 'trace.json'],
            'multi_worker_completed': ['best.json', 'checkpoint.json', 'run_summary.json', 'score_history.json'],
            'single_worker_provided_hypothesis': ['best.json', 'checkpoint.json', 'run_summary.json', 'score_history.json', 'trace.json'],
            'single_worker_generated_seed': ['best.json', 'checkpoint.json', 'run_summary.json', 'score_history.json', 'trace.json'],
            'evaluator_repair': ['metrics.json', 'artifacts.json', 'second_user_prompt.txt'],
        }
        for scenario_name, expected_manifest_files in expected_files.items():
            manifest = json.loads((root / scenario_name / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'implemented')
            self.assertEqual(manifest['implemented_files'], expected_manifest_files)
            normalized_expected = root / scenario_name / 'normalized_expected'
            for filename in expected_manifest_files:
                self.assertTrue((normalized_expected / filename).exists(), msg=f'missing {filename} for {scenario_name}')


if __name__ == '__main__':
    unittest.main()
