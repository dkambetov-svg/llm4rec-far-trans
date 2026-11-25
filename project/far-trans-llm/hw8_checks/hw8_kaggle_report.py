"""Read-only report over the extracted Kaggle HW8 outputs. Copies values; computes no nDCG.

Usage: python -I hw8_kaggle_report.py <package_dir> <kaggle_dir> <data_dir>
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

package, kaggle, data_dir = map(Path, sys.argv[1:4])
RUNS = {'mixed': kaggle / 'hw8_mixed', 'split': kaggle / 'hw8_split'}
METHOD = {'mixed': 'llmrank', 'split': 'llmrank_split'}


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def rel(path):
    return path.relative_to(kaggle.parent).as_posix()


metrics = {k: load_json(d / 'metrics.json') for k, d in RUNS.items()}
manifest = {k: load_json(d / 'run_manifest.json') for k, d in RUNS.items()}
per_user = {k: load_json(d / 'per_user.json') for k, d in RUNS.items()}
final = {k: load_jsonl(d / 'final_responses.jsonl') for k, d in RUNS.items()}
history_log = {k: load_jsonl(d / 'ranker_responses.jsonl') for k, d in RUNS.items()}
generations = {k: load_jsonl(d / 'generations.jsonl') for k, d in RUNS.items()}

print('=== 0. Run settings (run_manifest.json -> runtime / top level)')
for k, m in manifest.items():
    rt = m['runtime']
    keys = ('backend', 'model', 'requested_revision', 'resolved_revision', 'device', 'requested_dtype',
            'dtype', 'torch_version', 'transformers_version', 'do_sample', 'thinking', 'context_limit',
            'python_hash_seed', 'history_format', 'short_term_mode', 'history_events', 'adaptation',
            'raw_invalid_count', 'parse_repair_count', 'cache_hits', 'new_calls')
    print(f'[{rel(RUNS[k] / "run_manifest.json")}] method={m["method"]} users={len(m["users"])} '
          f'cutoff={m["cutoff"]} end_exclusive={m["end_exclusive"]} candidate_count={m["candidate_count"]}')
    print('   ', {key: rt.get(key) for key in keys})
    print('    failures listed:', len(m['failures']), '| data_sha256:', m['data_sha256'])
print('same users, same order (mixed vs split manifests):', manifest['mixed']['users'] == manifest['split']['users'])
print('final_responses records:', {k: len(v) for k, v in final.items()},
      '| ranker_responses (history) records:', {k: len(v) for k, v in history_log.items()},
      '| generations.jsonl records:', {k: len(v) for k, v in generations.items()})
print('generation_cache files:', {k: len(list((d / 'generation_cache').glob('*.json'))) for k, d in RUNS.items()})
for k, gens in generations.items():
    keys = Counter(g['cache_key'] for g in gens)
    print(f'  {k}: distinct cache keys in generations.jsonl={len(keys)}, cache_hit=True records='
          f'{sum(g["cache_hit"] for g in gens)}, keys used >1 time={sum(1 for v in keys.values() if v > 1)}')

print('\n=== 1. Results (metrics.json, values rounded to 4 decimals for display)')
base_names = ['popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell', 'candidate_order_first10']
for name in base_names:
    same = metrics['mixed'][name] == metrics['split'][name]
    print(f'  {name}: identical in mixed and split metrics.json: {same}')
rows = [(n, 'mixed') for n in base_names] + [('llmrank', 'mixed'), ('llmrank_split', 'split')]
labels = {'popularity': 'Popularity', 'cbf': 'CBF', 'latest_buy': 'Latest buy',
          'latest_buy_or_sell': 'Latest buy or sell', 'candidate_order_first10': 'candidate_order_first10',
          'llmrank': 'llmrank (mixed)', 'llmrank_split': 'llmrank_split'}
print('| Method | Users | Users with future purchases | nDCG@10 all | nDCG@10 nonempty | Failed/fallback cases | Source |')
print('|---|---:|---:|---:|---:|---:|---|')
for name, run in rows:
    v = metrics[run][name]
    if name in ('llmrank', 'llmrank_split'):
        rt = manifest[run]['runtime']
        fails = f'{rt["raw_invalid_count"]} (failures list: {len(manifest[run]["failures"])})'
        src = f'{rel(RUNS[run] / "metrics.json")}; fallbacks: {rel(RUNS[run] / "run_manifest.json")}'
    else:
        fails = 'n/a (not an LLM method)'
        src = rel(RUNS[run] / 'metrics.json')
    print(f'| {labels[name]} | {v["users"]} | {v["nonempty_truth_users"]} | {v["ndcg_at_10"]:.4f} | '
          f'{v["ndcg_at_10_nonempty"]:.4f} | {fails} | {src} |')

print('\n=== 2. Users with nonempty truth (per_user.json relevant_count > 0; fallback from final_responses.jsonl)')
score = {k: {(r['user_id'], r['method']): r for r in per_user[k]} for k in RUNS}
order = manifest['mixed']['users']
truth_users = [u for u in order if score['mixed'][(u, 'cbf')]['relevant_count'] > 0]
assert truth_users == [u for u in order if score['split'][(u, 'cbf')]['relevant_count'] > 0]
fallback = {k: {r['user_id']: r['fallback'] for r in final[k]} for k in RUNS}
print('| User | Truth count | Mixed | Split | nDCG mixed | nDCG split | nDCG CBF |')
print('|---|---:|---|---|---:|---:|---:|')
outcome = Counter()
changed = []
for u in truth_users:
    m = score['mixed'][(u, 'llmrank')]['ndcg_at_10']
    s = score['split'][(u, 'llmrank_split')]['ndcg_at_10']
    c = score['mixed'][(u, 'cbf')]['ndcg_at_10']
    assert c == score['split'][(u, 'cbf')]['ndcg_at_10']
    state = lambda k: 'fallback' if fallback[k][u] else 'valid'
    print(f'| {u[:6]} | {score["mixed"][(u, "cbf")]["relevant_count"]} | {state("mixed")} | {state("split")} | '
          f'{m:.4f} | {s:.4f} | {c:.4f} |')
    outcome['improved' if s > m else 'worse' if s < m else 'equal'] += 1
    if s != m:
        changed.append(u)
print('sources: per_user.json and final_responses.jsonl in outputs/kaggle/hw8_mixed and hw8_split')
print('\n=== 3. Split vs mixed among the', len(truth_users), 'users (per_user.json values compared, not recomputed)')
print(dict(outcome))
both_valid = [u for u in truth_users if not fallback['mixed'][u] and not fallback['split'][u]]
print('users valid in both runs:', len(both_valid), '| fallback in mixed only:',
      sum(fallback['mixed'][u] and not fallback['split'][u] for u in truth_users),
      '| fallback in split only:', sum(fallback['split'][u] and not fallback['mixed'][u] for u in truth_users),
      '| fallback in both:', sum(fallback['mixed'][u] and fallback['split'][u] for u in truth_users))

print('\n=== 4. Failure reasons (run_manifest.json failures; error = "Type: message")')
for k in RUNS:
    groups = defaultdict(list)
    for f in manifest[k]['failures']:
        groups[f['error']].append(f)
    total = sum(len(v) for v in groups.values())
    print(f'--- {k}: {total} failures (raw_invalid_count {manifest[k]["runtime"]["raw_invalid_count"]})')
    types = Counter(f['error'].split(':', 1)[0] for f in manifest[k]['failures'])
    print('    by exception type:', dict(types))
    for error, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ex = items[0]
        raw = ex['raw_response']
        shown = 'None (no text: exception before/while generating)' if raw is None else repr(raw[:400]) + (
            f' ... [truncated, {len(raw)} chars total]' if len(raw) > 400 else '')
        print(f'    [{len(items)}] {error}')
        print(f'        example user {ex["user_id"][:6]} raw_response: {shown}')
    non_format = [f for f in manifest[k]['failures'] if f['raw_response'] is None
                  or not f['error'].startswith('ValueError')]
    print(f'    non-ValueError or no-raw-text failures: {len(non_format)}',
          sorted({f["error"] for f in non_format}) if non_format else '')

print('\n=== 5. One changed user (first in manifest order among truth users with split != mixed)')
if changed:
    u = changed[0]
    sys.path.insert(0, str(package))
    import data_adapter  # noqa: E402
    args = SimpleNamespace(starter_dir=package.parent / 'far-trans', data_dir=data_dir, cutoff='2021-06-01',
                           horizon_days=30, max_users=100, candidate_count=50)
    bundle = data_adapter.load(args)
    starter, truth, contexts = bundle[0], bundle[3], bundle[4]
    relevant = truth.get(u, set()) & set(contexts[u]['eligible_asset_ids'])
    print('user:', u[:6], '| truth ISINs (local data via starter.load_data, eligible):', sorted(relevant))
    for k in RUNS:
        with (RUNS[k] / 'predictions.csv').open(encoding='utf-8') as stream:
            preds = defaultdict(list)
            for row in csv.DictReader(stream):
                if row['user_id'] == u:
                    preds[row['method']].append((int(row['rank']), row['ISIN']))
        for method in ((METHOD[k], 'candidate_order_first10') if k == 'mixed' else (METHOD[k],)):
            top = [isin for _rank, isin in sorted(preds[method])]
            hits = [rank for rank, isin in sorted(preds[method]) if isin in relevant]
            print(f'  {method} ({rel(RUNS[k] / "predictions.csv")}): {top}')
            print(f'    hit ranks: {hits} | per_user ndcg_at_10: {score[k][(u, method)]["ndcg_at_10"]}'
                  f' ({rel(RUNS[k] / "per_user.json")}) | fallback: {fallback[k][u]}')
