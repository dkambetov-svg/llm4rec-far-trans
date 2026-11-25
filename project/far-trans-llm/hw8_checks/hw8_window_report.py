"""Read-only report adding the Kaggle split-window run. Copies values; computes no nDCG.

Usage: python -I hw8_window_report.py <package_dir> <kaggle_dir> <data_dir>
"""
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

package, kaggle, data_dir = map(Path, sys.argv[1:4])
RUNS = {'mixed': kaggle / 'hw8_mixed', 'split': kaggle / 'hw8_split', 'window': kaggle / 'hw8_split_window'}
METHOD = {'mixed': 'llmrank', 'split': 'llmrank_split', 'window': 'llmrank_split_window'}


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
generations = {k: load_jsonl(d / 'generations.jsonl') for k, d in RUNS.items()}

print('=== 1. run_manifest.json side by side')
keys = sorted(set().union(*(m['runtime'] for m in manifest.values())))
for key in keys:
    vals = [manifest[k]['runtime'].get(key, '<absent>') for k in RUNS]
    flag = '' if len({json.dumps(v) for v in vals}) == 1 else '   <-- DIFFERS'
    print(f'  runtime.{key}: ' + ' | '.join(f'{k}={v!r}' for k, v in zip(RUNS, vals)) + flag)
for key in ('method', 'cutoff', 'end_exclusive', 'candidate_count', 'repeat_allowed', 'empty_truth_policy'):
    vals = [manifest[k].get(key) for k in RUNS]
    flag = '' if len({json.dumps(v) for v in vals}) == 1 else '   <-- DIFFERS'
    print(f'  {key}: ' + ' | '.join(f'{k}={v!r}' for k, v in zip(RUNS, vals)) + flag)
print('  data_sha256 equal (all three):', manifest['mixed']['data_sha256'] == manifest['split']['data_sha256']
      == manifest['window']['data_sha256'])
print('  users equal and same order (all three):',
      manifest['mixed']['users'] == manifest['split']['users'] == manifest['window']['users'])
print('  final_responses / generations records:', {k: (len(final[k]), len(generations[k])) for k in RUNS})
print('  generation_cache files:', {k: len(list((d / 'generation_cache').glob('*.json'))) for k, d in RUNS.items()})
for k in RUNS:
    print(f'  {k}: final_responses user order == manifest users:',
          [r['user_id'] for r in final[k]] == manifest[k]['users'])
    settings = {json.dumps(g['settings'], sort_keys=True) for g in generations[k]}
    print(f'  {k}: distinct settings blocks in generations.jsonl: {len(settings)}; '
          f'max_new_tokens: {sorted({g["max_new_tokens"] for g in generations[k]})}')

users = manifest['mixed']['users']


def payload(prompt):
    return json.loads(prompt.split('\n', 1)[1])


payloads = {}
for k in RUNS:
    assert len(generations[k]) == len(users), (k, len(generations[k]))
    payloads[k] = {u: payload(g['prompt']) for u, g in zip(users, generations[k])}

print('\n=== 2. Candidate lists (candidate_assets in the logged prompts, generations.jsonl)')
for a, b in (('window', 'split'), ('window', 'mixed'), ('split', 'mixed')):
    order_diff = sum([c['ISIN'] for c in payloads[a][u]['candidate_assets']]
                     != [c['ISIN'] for c in payloads[b][u]['candidate_assets']] for u in users)
    set_diff = sum({c['ISIN'] for c in payloads[a][u]['candidate_assets']}
                   != {c['ISIN'] for c in payloads[b][u]['candidate_assets']} for u in users)
    rec_diff = sum(payloads[a][u]['candidate_assets'] != payloads[b][u]['candidate_assets'] for u in users)
    print(f'  {a} vs {b}: users with different candidate order={order_diff}, different set={set_diff}, '
          f'different candidate records (full dicts)={rec_diff}')
print('  candidate counts per user (window):', sorted({len(payloads['window'][u]['candidate_assets']) for u in users}))
pred = {}
for k in RUNS:
    with (RUNS[k] / 'predictions.csv').open(encoding='utf-8') as stream:
        rows = defaultdict(list)
        for row in csv.DictReader(stream):
            rows[(row['user_id'], row['method'])].append((int(row['rank']), row['ISIN']))
    pred[k] = {key: [isin for _r, isin in sorted(v)] for key, v in rows.items()}
print('  candidate_order_first10 in predictions.csv equal window vs split for all users:',
      all(pred['window'][(u, 'candidate_order_first10')] == pred['split'][(u, 'candidate_order_first10')]
          for u in users))
print('  long_term_preferences.buy_counts equal window vs split for all users:',
      all(payloads['window'][u]['long_term_preferences']['buy_counts']
          == payloads['split'][u]['long_term_preferences']['buy_counts'] for u in users))

print('\n=== 3. Results row (metrics.json)')
for name in ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell', 'candidate_order_first10'):
    print(f'  {name} identical in window and split metrics.json:', metrics['window'][name] == metrics['split'][name])
print('  window metrics.json methods:', sorted(metrics['window']))
v = metrics['window']['llmrank_split_window']
rt = manifest['window']['runtime']
print(f'| llmrank_split_window | {v["users"]} | {v["nonempty_truth_users"]} | {v["ndcg_at_10"]:.4f} | '
      f'{v["ndcg_at_10_nonempty"]:.4f} | {rt["raw_invalid_count"]} (failures list: {len(manifest["window"]["failures"])}) '
      f'| {rel(RUNS["window"] / "metrics.json")}; fallbacks: {rel(RUNS["window"] / "run_manifest.json")} |')
print('  raw values:', v)
print('  parse_repair_count:', {k: manifest[k]['runtime']['parse_repair_count'] for k in RUNS})
repaired = [(r['user_id'][:6], r['fallback']) for r in final['window'] if r['parse_repaired']]
print('  window users with parse_repaired=True (user, fallback):', repaired)

print('\n=== 4. Users with nonempty truth')
score = {k: {(r['user_id'], r['method']): r for r in per_user[k]} for k in RUNS}
truth_users = [u for u in users if score['mixed'][(u, 'cbf')]['relevant_count'] > 0]
assert truth_users == [u for u in users if score['window'][(u, 'cbf')]['relevant_count'] > 0]
fallback = {k: {r['user_id']: r['fallback'] for r in final[k]} for k in RUNS}
state = lambda k, u: 'fallback' if fallback[k][u] else 'valid'
print('| User | Mixed | Split | Window | nDCG mixed | nDCG split | nDCG window | nDCG CBF |')
print('|---|---|---|---|---:|---:|---:|---:|')
cmp = {'window_vs_split': Counter(), 'window_vs_mixed': Counter()}
for u in truth_users:
    m, s, w = (score[k][(u, METHOD[k])]['ndcg_at_10'] for k in RUNS)
    c = score['window'][(u, 'cbf')]['ndcg_at_10']
    assert c == score['mixed'][(u, 'cbf')]['ndcg_at_10']
    print(f'| {u[:6]} | {state("mixed", u)} | {state("split", u)} | {state("window", u)} | '
          f'{m:.4f} | {s:.4f} | {w:.4f} | {c:.4f} |')
    for name, other in (('window_vs_split', s), ('window_vs_mixed', m)):
        cmp[name]['improved' if w > other else 'worse' if w < other else 'equal'] += 1
print('  comparisons (per_user.json values, not recomputed):', {k: dict(v) for k, v in cmp.items()})
print('  window fallbacks among the 11:', sum(fallback['window'][u] for u in truth_users))

print('\n=== 5. Failure reasons, window run (run_manifest.json failures)')
fails = manifest['window']['failures']
groups = defaultdict(list)
for f in fails:
    groups[f['error']].append(f)
print('  total', len(fails), '| by exception type:', dict(Counter(f['error'].split(':', 1)[0] for f in fails)))
for error, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    ex = items[0]
    raw = ex['raw_response']
    shown = 'None' if raw is None else repr(raw[:400]) + (f' ...[{len(raw)} chars]' if len(raw) > 400 else '')
    print(f'  [{len(items)}] {error}\n      example {ex["user_id"][:6]}: {shown}')
non_format = [f for f in fails if f['raw_response'] is None or not f['error'].startswith('ValueError')]
print('  non-ValueError or no-raw-text failures:', len(non_format), sorted({f['error'] for f in non_format}))
print('  final_responses fallback count:', sum(r['fallback'] for r in final['window']))

print('\n=== 6. short_term_interest.buy_events_oldest_to_newest length (logged prompts)')
for k in ('window', 'split'):
    n = [len(payloads[k][u]['short_term_interest']['buy_events_oldest_to_newest']) for u in users]
    print(f'  {k}: min={min(n)} median={statistics.median(n)} max={max(n)} empty={sum(x == 0 for x in n)} '
          f'| users={len(n)}')
w = payloads['window'][users[0]]['short_term_interest']
print('  window block keys/settings (first user):', {key: w[key] for key in w if key != 'buy_events_oldest_to_newest'})
dates = sorted(e['date'] for u in users for e in payloads['window'][u]['short_term_interest']['buy_events_oldest_to_newest'])
print('  window event dates min/max over all users:', (dates[0], dates[-1]) if dates else None)
both = Counter()
for u in users:
    nw = len(payloads['window'][u]['short_term_interest']['buy_events_oldest_to_newest'])
    nl = len(payloads['split'][u]['short_term_interest']['buy_events_oldest_to_newest'])
    both['window<last_k' if nw < nl else 'window>last_k' if nw > nl else 'equal'] += 1
print('  per-user window vs last_k event count:', dict(both))
tn = [len(payloads['window'][u]['short_term_interest']['buy_events_oldest_to_newest']) for u in truth_users]
print('  window events for the 11 truth users (manifest order):', dict(zip([u[:6] for u in truth_users], tn)))

print('\n=== 7. User 6B1591')
u = next(x for x in users if x.startswith('6B1591'))
sys.path.insert(0, str(package))
import data_adapter  # noqa: E402
args = SimpleNamespace(starter_dir=package.parent / 'far-trans', data_dir=data_dir, cutoff='2021-06-01',
                       horizon_days=30, max_users=100, candidate_count=50)
bundle = data_adapter.load(args)
truth, contexts = bundle[3], bundle[4]
relevant = truth.get(u, set()) & set(contexts[u]['eligible_asset_ids'])
print('  truth ISINs (local data via starter.load_data, eligible):', sorted(relevant))
for k in RUNS:
    top = pred[k][(u, METHOD[k])]
    hits = [i + 1 for i, isin in enumerate(top) if isin in relevant]
    rec = next(r for r in final[k] if r['user_id'] == u)
    print(f'  {METHOD[k]} ({rel(RUNS[k] / "predictions.csv")}): {top}')
    print(f'    hit ranks: {hits} | per_user ndcg_at_10: {score[k][(u, METHOD[k])]["ndcg_at_10"]} '
          f'({rel(RUNS[k] / "per_user.json")}) | fallback: {rec["fallback"]} | error: {rec["error"]} '
          f'| parse_repaired: {rec["parse_repaired"]}')
print('  window short_term_interest events for this user:',
      payloads['window'][u]['short_term_interest']['buy_events_oldest_to_newest'])
