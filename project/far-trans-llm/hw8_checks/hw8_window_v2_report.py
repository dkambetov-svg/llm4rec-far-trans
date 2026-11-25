"""Read-only report for the Kaggle split-window v2 run. Copies values; computes no nDCG.

Usage: python -I hw8_window_v2_report.py <kaggle_dir>
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

kaggle = Path(sys.argv[1])
RUNS = {'mixed': kaggle / 'hw8_mixed', 'split': kaggle / 'hw8_split',
        'v2': kaggle / 'hw8_split_window_v2', 'w520': kaggle / 'hw8_split_window'}
METHOD = {'mixed': 'llmrank', 'split': 'llmrank_split', 'v2': 'llmrank_split_window',
          'w520': 'llmrank_split_window'}


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
users = manifest['split']['users']

print('=== 1. runtime fields: v2 vs split (run_manifest.json)')
rt = {k: manifest[k]['runtime'] for k in RUNS}
for key in sorted(set(rt['v2']) | set(rt['split'])):
    a, b = rt['v2'].get(key, '<absent>'), rt['split'].get(key, '<absent>')
    mark = '   <-- DIFFERS' if a != b else ''
    print(f'  {key}: v2={a!r} | split={b!r}{mark}')
print('  differing runtime fields:', sorted(k for k in set(rt['v2']) | set(rt['split'])
                                         if rt['v2'].get(k, '<absent>') != rt['split'].get(k, '<absent>')))
for key in ('method', 'cutoff', 'end_exclusive', 'candidate_count', 'data_sha256', 'repeat_allowed',
            'empty_truth_policy', 'excluded_rows', 'teaching_settings_not_official_test'):
    print(f'  top-level {key} equal v2 vs split:', manifest['v2'].get(key) == manifest['split'].get(key),
          '' if manifest['v2'].get(key) == manifest['split'].get(key)
          else f'(v2={manifest["v2"].get(key)!r}, split={manifest["split"].get(key)!r})')
print('  users equal, same order v2 vs split:', manifest['v2']['users'] == users)
print('  records final/generations:', {k: (len(final[k]), len(generations[k])) for k in RUNS})
print('  generation_cache files:', {k: len(list((d / 'generation_cache').glob('*.json'))) for k, d in RUNS.items()})
for k in RUNS:
    print(f'  {k}: final_responses order == users: {[r["user_id"] for r in final[k]] == users}; '
          f'distinct settings in generations.jsonl: {len({json.dumps(g["settings"], sort_keys=True) for g in generations[k]})}')

payloads = {}
for k in RUNS:
    assert len(generations[k]) == len(users)
    payloads[k] = {u: json.loads(g['prompt'].split('\n', 1)[1]) for u, g in zip(users, generations[k])}

print('\n=== 2. candidate_assets in logged prompts (generations.jsonl)')
for a, b in (('v2', 'split'), ('v2', 'mixed'), ('v2', 'w520')):
    order = sum([c['ISIN'] for c in payloads[a][u]['candidate_assets']]
                != [c['ISIN'] for c in payloads[b][u]['candidate_assets']] for u in users)
    full = sum(payloads[a][u]['candidate_assets'] != payloads[b][u]['candidate_assets'] for u in users)
    print(f'  {a} vs {b}: users with different candidate ISIN order={order}, different candidate records={full}')
print('  v2 vs w520 prompts identical (text) for users:',
      sum(ga['prompt'] == gb['prompt'] for ga, gb in zip(generations['v2'], generations['w520'])), '/ 100')

pred = {}
for k in RUNS:
    with (RUNS[k] / 'predictions.csv').open(encoding='utf-8') as stream:
        rows = defaultdict(list)
        for row in csv.DictReader(stream):
            rows[(row['user_id'], row['method'])].append((int(row['rank']), row['ISIN']))
    pred[k] = {key: [isin for _r, isin in sorted(v)] for key, v in rows.items()}

print('\n=== 3. Final results table')
for name in ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell', 'candidate_order_first10'):
    print(f'  {name} identical across all four metrics.json:',
          all(metrics[k][name] == metrics['mixed'][name] for k in RUNS))
labels = [('Popularity', 'popularity', 'mixed'), ('CBF', 'cbf', 'mixed'), ('Latest buy', 'latest_buy', 'mixed'),
          ('Latest buy or sell', 'latest_buy_or_sell', 'mixed'),
          ('candidate_order_first10', 'candidate_order_first10', 'mixed'),
          ('llmrank (mixed)', 'llmrank', 'mixed'), ('llmrank_split (last_k)', 'llmrank_split', 'split'),
          ('llmrank_split_window (v2)', 'llmrank_split_window', 'v2'),
          ('llmrank_split_window, excluded: transformers 5.20.2', 'llmrank_split_window', 'w520')]
print('| Method | Users | Users with future purchases | nDCG@10 all | nDCG@10 nonempty | Failed/fallback | Source |')
print('|---|---:|---:|---:|---:|---:|---|')
for label, name, run in labels:
    v = metrics[run][name]
    if name.startswith('llmrank'):
        fb = f'{rt[run]["raw_invalid_count"]} (failures list {len(manifest[run]["failures"])})'
        src = f'{rel(RUNS[run] / "metrics.json")}; {rel(RUNS[run] / "run_manifest.json")}'
    else:
        fb, src = 'n/a', rel(RUNS[run] / 'metrics.json')
    print(f'| {label} | {v["users"]} | {v["nonempty_truth_users"]} | {v["ndcg_at_10"]:.4f} | '
          f'{v["ndcg_at_10_nonempty"]:.4f} | {fb} | {src} |')
print('  raw v2:', metrics['v2']['llmrank_split_window'])
print('  raw w520:', metrics['w520']['llmrank_split_window'])
print('  parse_repair_count:', {k: rt[k]['parse_repair_count'] for k in RUNS})
print('  v2 parse_repaired users (user, fallback):',
      [(r['user_id'][:6], r['fallback']) for r in final['v2'] if r['parse_repaired']])

print('\n=== 4. Users with nonempty truth')
score = {k: {(r['user_id'], r['method']): r for r in per_user[k]} for k in RUNS}
truth_users = [u for u in users if score['split'][(u, 'cbf')]['relevant_count'] > 0]
assert truth_users == [u for u in users if score['v2'][(u, 'cbf')]['relevant_count'] > 0]
fallback = {k: {r['user_id']: r['fallback'] for r in final[k]} for k in RUNS}
st = lambda k, u: 'fallback' if fallback[k][u] else 'valid'
print('| User | Mixed | Split | Window v2 | nDCG mixed | nDCG split | nDCG window v2 | nDCG CBF |')
print('|---|---|---|---|---:|---:|---:|---:|')
cmp = {'v2_vs_split': Counter(), 'v2_vs_mixed': Counter(), 'v2_vs_w520': Counter()}
for u in truth_users:
    m, s, w, x = (score[k][(u, METHOD[k])]['ndcg_at_10'] for k in ('mixed', 'split', 'v2', 'w520'))
    c = score['v2'][(u, 'cbf')]['ndcg_at_10']
    assert c == score['split'][(u, 'cbf')]['ndcg_at_10']
    print(f'| {u[:6]} | {st("mixed", u)} | {st("split", u)} | {st("v2", u)} | {m:.4f} | {s:.4f} | {w:.4f} | {c:.4f} |')
    for name, other in (('v2_vs_split', s), ('v2_vs_mixed', m), ('v2_vs_w520', x)):
        cmp[name]['improved' if w > other else 'worse' if w < other else 'equal'] += 1
print('  comparisons (per_user.json values, not recomputed):', {k: dict(v) for k, v in cmp.items()})
print('  v2 fallbacks among the 11:', sum(fallback['v2'][u] for u in truth_users))

print('\n=== 5. v2 vs w520 (transformers 5.20.2)')
same_top = sum(pred['v2'][(u, METHOD['v2'])] == pred['w520'][(u, METHOD['w520'])] for u in users)
same_status = sum(fallback['v2'][u] == fallback['w520'][u] for u in users)
raw = {k: {r['user_id']: r['raw_response'] for r in final[k]} for k in ('v2', 'w520')}
same_raw = sum(raw['v2'][u] == raw['w520'][u] for u in users)
print(f'  identical top-10 lists: {same_top}/100 | identical valid/fallback status: {same_status}/100 '
      f'| identical raw_response text: {same_raw}/100')
print('  status pairs (v2, w520):', dict(Counter((st('v2', u), st('w520', u)) for u in users)))
both_valid = [u for u in users if not fallback['v2'][u] and not fallback['w520'][u]]
print('  identical top-10 among users valid in both:',
      sum(pred['v2'][(u, METHOD['v2'])] == pred['w520'][(u, METHOD['w520'])] for u in both_valid), '/', len(both_valid))
types = {k: Counter(f['error'] for f in manifest[k]['failures']) for k in ('v2', 'w520')}
for err in sorted(set(types['v2']) | set(types['w520'])):
    print(f'  {err}: v2={types["v2"].get(err, 0)} | w520={types["w520"].get(err, 0)}')
print('  totals:', {k: sum(v.values()) for k, v in types.items()})
for k in ('v2', 'w520'):
    nf = [f for f in manifest[k]['failures'] if f['raw_response'] is None or not f['error'].startswith('ValueError')]
    print(f'  {k}: non-ValueError or no-raw-text failures: {len(nf)}')
fail_users = {k: {f['user_id'] for f in manifest[k]['failures']} for k in ('v2', 'w520')}
print('  failing users: both', len(fail_users['v2'] & fail_users['w520']), '| v2 only',
      len(fail_users['v2'] - fail_users['w520']), '| w520 only', len(fail_users['w520'] - fail_users['v2']))
u = next(x for x in users if x.startswith('6B1591'))
for k in ('split', 'v2', 'w520'):
    print(f'  6B1591 {k}: {pred[k][(u, METHOD[k])]} | ndcg {score[k][(u, METHOD[k])]["ndcg_at_10"]} | {st(k, u)}')
