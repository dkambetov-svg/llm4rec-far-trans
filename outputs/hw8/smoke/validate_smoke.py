"""Validate the two bounded smoke arms without launching further inference."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

BASE = Path('/Users/davidkk/llm4rec-hw2/RecSys-LLMs/outputs/hw8/smoke')
ARMS = ('unified', 'split')
PINNED = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
MODEL_SETTING_KEYS = (
    'model', 'requested_revision', 'resolved_revision', 'device', 'dtype',
    'torch_version', 'transformers_version', 'tokenizer_class', 'do_sample',
    'thinking', 'context_limit', 'shuffle_seed', 'history_events',
    'short_term_days', 'temporal_boundary', 'retry_invalid',
)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def arm_data(arm):
    out = BASE / arm
    prompts = jsonl(BASE / f'prompts_{arm}/ranker_prompts.jsonl')
    prepared = {}
    for row in prompts:
        payload = json.loads(row['prompt'].rsplit('\n', 1)[1])
        prepared[row['user_id']] = {
            'candidate_ids': [asset['ISIN'] for asset in payload['candidate_assets']],
            'evidence_sha256': payload['evidence_sha256'],
            'history_mode': payload['history_mode'],
        }
    predictions = {}
    with (out / 'predictions.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['method'] == 'llmrank':
                predictions.setdefault(row['user_id'], []).append((int(row['rank']), row['ISIN']))
    predictions = {uid: [isin for _, isin in sorted(rows)] for uid, rows in predictions.items()}
    return {
        'dir': out,
        'prepared': prepared,
        'predictions': predictions,
        'manifest': json.loads((out / 'run_manifest.json').read_text()),
        'wrapper': json.loads((out / 'run_wrapper.json').read_text()),
        'generations': jsonl(out / 'generations.jsonl'),
        'responses': jsonl(out / 'final_responses.jsonl'),
    }


def sha256_ids(ids):
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()


def main():
    data = {arm: arm_data(arm) for arm in ARMS}
    users = {arm: sorted(data[arm]['prepared']) for arm in ARMS}
    evidence = {arm: {u: data[arm]['prepared'][u]['evidence_sha256'] for u in users[arm]}
                for arm in ARMS}
    candidate_hashes = {arm: {u: sha256_ids(data[arm]['prepared'][u]['candidate_ids'])
                              for u in users[arm]} for arm in ARMS}
    runtime = {arm: data[arm]['manifest']['runtime'] for arm in ARMS}

    validity = {}
    for arm in ARMS:
        validity[arm] = {}
        for uid in users[arm]:
            ids = data[arm]['predictions'][uid]
            eligible = set(data[arm]['prepared'][uid]['candidate_ids'])
            validity[arm][uid] = {
                'rank_length': len(ids),
                'unique': len(set(ids)) == len(ids),
                'eligible': set(ids) <= eligible,
            }

    settings = {k: {arm: runtime[arm].get(k) for arm in ARMS} for k in MODEL_SETTING_KEYS}
    settings_identical = all(values['unified'] == values['split'] for values in settings.values())

    generation_summary = {}
    for arm in ARMS:
        rows = data[arm]['generations']
        generation_summary[arm] = {
            'calls': len(rows),
            'new_calls': sum(not r['cache_hit'] for r in rows),
            'cache_hits': sum(r['cache_hit'] for r in rows),
            'input_tokens': sum(r['response']['input_tokens'] for r in rows),
            'output_tokens': sum(r['response']['output_tokens'] for r in rows),
            'generation_seconds': round(sum(r['response']['seconds'] for r in rows), 2),
            'wall_seconds': round(data[arm]['wrapper']['wall_seconds'], 2),
            'peak_child_rss_mb': data[arm]['wrapper']['peak_child_rss_mb'],
        }

    artifact_names = ('generations.jsonl', 'generation_cache', 'final_responses.jsonl',
                      'ranker_responses.jsonl', 'run_manifest.json', 'metrics.json',
                      'per_user.json', 'predictions.csv', 'run.log', 'run_wrapper.json')
    artifacts = {arm: {name: (data[arm]['dir'] / name).exists() for name in artifact_names}
                 for arm in ARMS}

    fallback_summary = {}
    for arm in ARMS:
        rows = data[arm]['responses']
        fallback_summary[arm] = {
            'manifest_raw_invalid_count': runtime[arm]['raw_invalid_count'],
            'manifest_parse_repair_count': runtime[arm]['parse_repair_count'],
            'manifest_final_invalid_after_retry': runtime[arm]['final_invalid_after_retry'],
            'response_rows': len(rows),
            'rows_with_fallback': sum(bool(r.get('fallback')) for r in rows),
            'rows_with_parse_repaired': sum(bool(r.get('parse_repaired')) for r in rows),
            'rows_with_raw_response': sum(bool(r.get('raw_response')) for r in rows),
        }

    checks = {
        'same_users': users['unified'] == users['split'],
        'same_evidence_hashes': evidence['unified'] == evidence['split'],
        'same_candidate_hashes_and_order': candidate_hashes['unified'] == candidate_hashes['split'],
        'exactly_ten_unique_eligible_outputs': all(
            v['rank_length'] == 10 and v['unique'] and v['eligible']
            for arm in ARMS for v in validity[arm].values()),
        'actual_model_inference': all(
            generation_summary[arm]['new_calls'] == 3
            and generation_summary[arm]['cache_hits'] == 0
            and fallback_summary[arm]['rows_with_raw_response'] == 3
            for arm in ARMS),
        'identical_model_settings': settings_identical,
        'pinned_revision_resolved': all(runtime[arm]['resolved_revision'] == PINNED for arm in ARMS),
        'explicit_mps_device': all(runtime[arm]['device'] == 'mps' for arm in ARMS),
        'artifacts_saved': all(all(values.values()) for values in artifacts.values()),
        'invalid_fallback_tokens_runtime_recorded': all(
            'raw_invalid_count' in runtime[arm]
            and 'final_invalid_after_retry' in runtime[arm]
            and generation_summary[arm]['input_tokens'] > 0
            and generation_summary[arm]['output_tokens'] > 0
            and generation_summary[arm]['wall_seconds'] > 0
            for arm in ARMS),
        'both_runs_within_budget': all(
            not data[arm]['wrapper']['timed_out']
            and data[arm]['wrapper']['exit_code'] == 0
            and data[arm]['wrapper']['wall_seconds'] <= 900 for arm in ARMS),
    }

    gen_seconds = [generation_summary[arm]['generation_seconds'] for arm in ARMS]
    wall_seconds = [generation_summary[arm]['wall_seconds'] for arm in ARMS]
    estimate = {
        'basis': 'measured generation seconds / 3 users, assuming linear scaling and no cache reuse',
        'generation_seconds_per_user_by_arm': [round(s / 3, 2) for s in gen_seconds],
        'mean_generation_seconds_per_user': round(sum(gen_seconds) / 6, 2),
        'estimated_seconds_per_100_users_per_arm': round(sum(gen_seconds) / 6 * 100, 1),
        'estimated_hours_per_100_users_per_arm': round(sum(gen_seconds) / 6 * 100 / 3600, 2),
        'estimated_hours_both_arms_100_users_each': round(sum(gen_seconds) / 6 * 100 * 2 / 3600, 2),
        'observed_wall_seconds_by_arm': wall_seconds,
        'caveat': 'Three-user nDCG is not scientific evidence; split user 3 hit the 512-token cap.',
    }

    result = {
        'model': runtime['unified']['model'],
        'pinned_revision': PINNED,
        'users': users,
        'checks': checks,
        'evidence_hashes': evidence,
        'candidate_hashes': candidate_hashes,
        'model_settings_compared': settings,
        'generation_summary': generation_summary,
        'fallback_summary': fallback_summary,
        'output_validity': validity,
        'artifacts': artifacts,
        'estimate': estimate,
    }
    (BASE / 'smoke_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'checks': checks,
        'generation_summary': generation_summary,
        'fallback_summary': fallback_summary,
        'estimate': estimate,
    }, indent=2))


if __name__ == '__main__':
    main()
