"""CPU-only tests for the controlled unified/split history ranker modes (no inference)."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import ranker
import test_ranker


CUTOFF = '2021-06-01'
IDS = [f'ISIN{i:02d}' for i in range(15)]


def fixture():
    assets = {isin: {'assetName': 'Asset ' + isin, 'assetCategory': 'Stock',
                     'assetSubCategory': 'Equity', 'sector': 'Tech', 'industry': 'Software',
                     'marketID': 'X', 'timestamp': '2021-01-01'} for isin in IDS}
    context = {'user_id': 'u', 'cutoff': CUTOFF,
               'candidate_assets': [{'id': isin} for isin in IDS[:12]],
               'eligible_asset_ids': IDS + ['DO_NOT_LEAK'],
               'profile': {'timestamp': '2021-01-01', 'customerType': 'Mass',
                           'riskLevel': 'Low'}}
    history = Counter({IDS[0]: 3, IDS[1]: 2, IDS[2]: 1, IDS[3]: 1})
    events = [
        ('2021-02-01', IDS[0], 'Buy'),        # long-term; same ISIN also bought short-term
        ('2021-03-02', IDS[1], 'Buy'),        # long-term (day before boundary)
        ('2021-03-03', IDS[2], 'Buy'),        # short-term (boundary is inclusive)
        ('2021-05-31', IDS[0], 'Buy'),        # short-term
        ('2021-06-01', IDS[4], 'Buy'),        # future: must be excluded
        ('2021-05-30', IDS[5], 'Sell'),       # sell: must be excluded from ranker history
        ('2021-05-29', 'DO_NOT_LEAK', 'Buy'),  # not an asset: must be excluded
    ]
    return assets, context, history, events


def payloads(history=None, events=None):
    assets, context, default_history, default_events = fixture()
    history = default_history if history is None else history
    events = default_events if events is None else events
    unified = ranker.prompt_payload(context, assets, history, events, 15, None,
                                    history_mode='unified', short_term_days=90)
    split = ranker.prompt_payload(context, assets, history, events, 15, None,
                                  history_mode='split', short_term_days=90)
    return assets, context, unified, split


def all_records(payload):
    if 'history_unified' in payload:
        return list(payload['history_unified'])
    return list(payload['long_term_history']) + list(payload['short_term_history'])


class InvalidGenerator:
    metadata = {'backend': 'fake_test_not_model', 'new_calls': 0}

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        return {'text': json.dumps(IDS[:9]), 'input_tokens': 1,
                'output_tokens': 1, 'seconds': 0}


def run_args(directory, mode):
    return SimpleNamespace(output_dir=Path(directory), history_events=15, shuffle_seed=None,
                           prepare_only=False, on_invalid='fallback', retry_invalid=0,
                           history_mode=mode, short_term_days=90)


def bundle():
    assets, context, history, events = fixture()
    contexts = {'u': context}
    baseline = {uid: {name: IDS[:10] for name in
                      ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell')}
                for uid in contexts}
    return (test_ranker.FakeStarter(), assets, history, {}, contexts, baseline,
            {'u': events}, {}, {'users': ['u']})


class CanonicalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.assets, self.context, self.unified, self.split = payloads()

    def test_identical_underlying_evidence_and_hash(self):
        evidence = ranker.canonical_evidence(self.context, self.assets,
                                             Counter({IDS[0]: 3, IDS[1]: 2, IDS[2]: 1, IDS[3]: 1}),
                                             fixture()[3], 15)
        self.assertEqual(all_records(self.unified), evidence['records'])
        key = lambda record: record['ISIN']
        self.assertEqual(sorted(all_records(self.split), key=key),
                         sorted(evidence['records'], key=key))
        self.assertEqual(self.unified['evidence_sha256'], evidence['sha256'])
        self.assertEqual(self.split['evidence_sha256'], self.unified['evidence_sha256'])
        self.assertEqual(self.unified['most_recent_buy'], evidence['most_recent_buy'])
        self.assertEqual(self.split['most_recent_buy'], evidence['most_recent_buy'])

    def test_identical_candidate_ids_and_order(self):
        expected = ranker.candidate_ids(self.context)
        self.assertEqual(expected, IDS[:12])
        for payload in (self.unified, self.split):
            self.assertEqual([row['ISIN'] for row in payload['candidate_assets']], expected)
        self.assertEqual(self.unified['candidate_assets'], self.split['candidate_assets'])

    def test_no_future_transactions_or_sell_rows(self):
        dates = [day for record in all_records(self.unified) for day in record['dates']]
        self.assertTrue(dates)
        self.assertTrue(all(day < CUTOFF for day in dates))
        self.assertNotIn('2021-06-01', dates)
        self.assertNotIn(IDS[4], {record['ISIN'] for record in all_records(self.unified)})
        self.assertNotIn(IDS[5], {record['ISIN'] for record in all_records(self.unified)})

    def test_correct_90_day_boundary(self):
        self.assertEqual(ranker.temporal_boundary(CUTOFF, 90), '2021-03-03')
        self.assertEqual(self.split['temporal_boundary'], '2021-03-03')
        self.assertEqual(self.split['short_term_days'], 90)
        self.assertEqual({r['ISIN'] for r in self.split['long_term_history']},
                         {IDS[1], IDS[3]})
        self.assertEqual({r['ISIN'] for r in self.split['short_term_history']},
                         {IDS[0], IDS[2]})

    def test_no_duplicated_or_lost_records(self):
        unified = all_records(self.unified)
        split = all_records(self.split)
        self.assertEqual(len(unified), len(split))
        self.assertEqual(len({r['ISIN'] for r in unified}), len(unified))
        self.assertEqual({r['ISIN'] for r in unified}, {r['ISIN'] for r in split})
        self.assertFalse({r['ISIN'] for r in self.split['long_term_history']} &
                         {r['ISIN'] for r in self.split['short_term_history']})
        self.assertEqual(Counter({r['ISIN']: r['buy_count'] for r in split}),
                         Counter({IDS[0]: 3, IDS[1]: 2, IDS[2]: 1, IDS[3]: 1}))
        self.assertEqual(sorted(day for r in unified for day in r['dates']),
                         sorted(day for r in split for day in r['dates']))

    def test_empty_long_term_section(self):
        _, _, _, split = payloads(history=Counter({IDS[2]: 1}),
                                  events=[('2021-05-31', IDS[2], 'Buy')])
        self.assertEqual(split['long_term_history'], [])
        self.assertEqual([r['ISIN'] for r in split['short_term_history']], [IDS[2]])

    def test_empty_short_term_section(self):
        _, _, _, split = payloads(history=Counter({IDS[0]: 1}),
                                  events=[('2021-02-01', IDS[0], 'Buy')])
        self.assertEqual(split['short_term_history'], [])
        self.assertEqual([r['ISIN'] for r in split['long_term_history']], [IDS[0]])

    def test_invalid_short_term_days_rejected(self):
        assets, context, history, events = fixture()
        with self.assertRaises(ValueError):
            ranker.prompt_payload(context, assets, history, events, 15, None,
                                  history_mode='split', short_term_days=0)


class ModeExecutionTests(unittest.TestCase):
    def test_invalid_response_falls_back_in_both_modes(self):
        for mode in ('unified', 'split'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, \
                    patch.object(ranker.data_adapter, 'save_results') as save:
                fake = InvalidGenerator()
                metadata = ranker.run(run_args(directory, mode), bundle(), fake)
                predictions = save.call_args.args[2]
                self.assertEqual(predictions['u']['llmrank'], IDS[:10])
                self.assertEqual(predictions['u']['candidate_order_first10'], IDS[:10])
                self.assertEqual(metadata['history_mode'], mode)
                self.assertEqual(metadata['prompt_version'], ranker.PROMPT_VERSIONS[mode])
                self.assertEqual(metadata['temporal_boundary'], '2021-03-03')
                self.assertEqual(metadata['raw_invalid_count'], 1)
                self.assertEqual(len(fake.prompts), 1)

    def test_manifest_evidence_hashes_present(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(ranker.data_adapter, 'save_results') as save:
            ranker.run(run_args(directory, 'split'), bundle(), InvalidGenerator())
            metadata = save.call_args.args[5]
            self.assertIn('u', metadata['evidence_hashes'])
            self.assertEqual(len(metadata['evidence_hashes']['u']), 64)
            self.assertEqual(len(metadata['evidence_aggregate_sha256']), 64)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_legacy_payload_keeps_original_shape(self):
        assets, context, history, events = fixture()
        payload = ranker.prompt_payload(context, assets, history, events, 15, None)
        for key in ('buy_history_counts', 'recent_buy_events_oldest_to_newest',
                    'most_recent_buy', 'chronology_note', 'candidate_assets'):
            self.assertIn(key, payload)
        for key in ('history_unified', 'long_term_history', 'short_term_history',
                    'history_mode', 'evidence_sha256'):
            self.assertNotIn(key, payload)

    def test_default_build_prompt_has_no_experimental_instruction(self):
        assets, context, history, events = fixture()
        prompt = ranker.build_prompt(context, assets, history, events, 15, None)
        self.assertNotIn('long-term/short-term distinction', prompt)
        self.assertNotIn('split into long-term and short-term sections', prompt)
        self.assertEqual(ranker.PROMPT_VERSIONS['legacy'], 'ranker_prompt_v1_legacy')


if __name__ == '__main__':
    unittest.main()
