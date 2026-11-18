"""HW2 change: CPU-only tests for the retry-on-invalid step (fake generator, no model)."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import ranker
import test_ranker
from test_ranker import IDS


class SequenceGenerator:
    """Returns the given answers in order; records every prompt it receives."""
    metadata = {'backend': 'fake_test_not_model', 'new_calls': 0}

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def generate(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        return {'text': self.answers.pop(0), 'input_tokens': 1, 'output_tokens': 1, 'seconds': 0}


def args(directory, retry):
    return SimpleNamespace(output_dir=Path(directory), history_events=15, shuffle_seed=42,
                           prepare_only=False, on_invalid='fallback', retry_invalid=retry)


class RetryTests(unittest.TestCase):
    bundle = test_ranker.ExecutionTests.bundle

    def run_ranker(self, answers, retry=1):
        fake = SequenceGenerator(answers)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(ranker.data_adapter, 'save_results') as save:
            metadata = ranker.run(args(directory, retry), self.bundle(), fake)
            final = [json.loads(line) for line in
                     (Path(directory) / 'final_responses.jsonl').read_text().splitlines()]
        return metadata, save.call_args.args[2], fake, final

    def test_retry_recovers_invalid_first_answer(self):
        good = list(reversed(IDS[:10]))
        # user u: invalid (9 IDs), then valid on retry; user v: valid first time.
        metadata, predictions, fake, final = self.run_ranker(
            [json.dumps(IDS[:9]), json.dumps(good), json.dumps(good)])
        self.assertEqual(len(fake.prompts), 3)
        self.assertIn('Validator error', fake.prompts[1])
        self.assertEqual(predictions['u']['llmrank'], IDS[:10])        # control keeps fallback
        self.assertEqual(predictions['u']['llmrank_retry'], good)      # ours uses the retry
        self.assertEqual(predictions['v']['llmrank'], good)
        self.assertEqual(predictions['v']['llmrank_retry'], good)      # valid users identical
        self.assertEqual(metadata['raw_invalid_count'], 1)
        self.assertEqual(metadata['retry_attempts'], 1)
        self.assertEqual(metadata['retry_recovered'], 1)
        self.assertEqual(metadata['final_invalid_after_retry'], 0)
        self.assertTrue(final[0]['retry_recovered'])

    def test_failed_retry_falls_back_to_candidate_order(self):
        metadata, predictions, fake, final = self.run_ranker(
            ['invalid', 'still invalid', 'invalid', 'still invalid'])
        self.assertEqual(len(fake.prompts), 4)
        for user in ('u', 'v'):
            self.assertEqual(predictions[user]['llmrank_retry'], IDS[:10])
        self.assertEqual(metadata['retry_recovered'], 0)
        self.assertEqual(metadata['final_invalid_after_retry'], 2)
        self.assertIsNotNone(final[0]['retry_error'])

    def test_no_retry_keeps_original_behaviour(self):
        metadata, predictions, fake, _ = self.run_ranker(['invalid', 'invalid'], retry=0)
        self.assertEqual(len(fake.prompts), 2)
        self.assertNotIn('llmrank_retry', predictions['u'])


if __name__ == '__main__':
    unittest.main()
