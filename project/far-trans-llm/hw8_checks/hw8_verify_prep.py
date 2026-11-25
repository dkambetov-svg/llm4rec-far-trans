"""HW8 prepare-only verification on real FAR-Trans data (no inference, read-only).

Checks two --prepare-only prompt files (mixed vs split last_k) for the same cohort:
future dates, identical candidates, long-term counts vs starter history, byte-level
regrouping equivalence, prompt lengths (characters), and future-truth leakage for one
example user. Run with the same PYTHONHASHSEED as the prompt files were made with:

    PYTHONHASHSEED=0 python hw8_checks/hw8_verify_prep.py <data_dir> <mixed_dir> <split_dir> <out_json>

Writes <out_json> and example_split_prompt.txt next to it (keep both under outputs/).
"""
import json
import re
import statistics
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

PACKAGE = Path(__file__).resolve().parent.parent  # project/far-trans-llm
sys.path.insert(0, str(PACKAGE))
import data_adapter  # noqa: E402
import ranker  # noqa: E402

DATE = re.compile(r'\d{4}-\d{2}-\d{2}')
data_dir, mixed_dir, split_dir, out_json = map(Path, sys.argv[1:5])
CUTOFF = '2021-06-01'


def read(path):
    rows = [json.loads(line) for line in (path / 'ranker_prompts.jsonl').read_text(encoding='utf-8').splitlines()]
    return {r['user_id']: r for r in rows}, [r['user_id'] for r in rows]


def payload(prompt):
    head, body = prompt.split('\n', 1)
    return head, json.loads(body), list(json.loads(body).keys())


def dated_strings(value, path=''):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from dated_strings(child, f'{path}.{key}')
    elif isinstance(value, list):
        for child in value:
            yield from dated_strings(child, f'{path}[]')
    elif isinstance(value, str) and DATE.match(value):
        yield path, value


args = SimpleNamespace(starter_dir=PACKAGE.parent / 'far-trans', data_dir=data_dir, cutoff=CUTOFF,
                       horizon_days=30, max_users=100, candidate_count=50)
starter, assets, history, truth, contexts, _baseline, events, _pop, manifest = data_adapter.load(args)
mixed, mixed_order = read(mixed_dir)
split, split_order = read(split_dir)
report = {'cohort_users': len(manifest['users'])}

# user order and cohort identity
report['same_user_order'] = mixed_order == split_order == manifest['users']
report['split_record_settings'] = sorted({(r.get('history_format'), r.get('short_term_mode'))
                                          for r in split.values()}, key=str)

# key order (sort_keys)
report['top_level_key_order'] = {
    'mixed': payload(mixed[mixed_order[0]]['prompt'])[2],
    'split': payload(split[split_order[0]]['prompt'])[2]}
report['key_order_constant_within_format'] = {
    name: len({tuple(payload(r['prompt'])[2]) for r in rows.values()}) == 1
    for name, rows in (('mixed', mixed), ('split', split))}

# (a) dates
for name, rows in (('mixed', mixed), ('split', split)):
    at_or_after, max_by_field, raw_at_or_after = Counter(), {}, Counter()
    for r in rows.values():
        _, body, _ = payload(r['prompt'])
        for path, value in dated_strings(body):
            if path == '.cutoff_exclusive':
                continue
            max_by_field[path] = max(max_by_field.get(path, ''), value[:10])
            if value[:10] >= CUTOFF:
                at_or_after[path] += 1
        for d in DATE.findall(r['prompt']):
            if d >= CUTOFF:
                raw_at_or_after[d] += 1
    report[f'a_{name}'] = {'dated_fields_at_or_after_cutoff_excluding_cutoff_label': dict(at_or_after),
                           'max_date_per_field': max_by_field,
                           'raw_regex_dates_at_or_after_cutoff_anywhere_in_prompt': dict(raw_at_or_after)}

# (b) candidates identical per user
mismatch = [u for u in manifest['users']
            if payload(mixed[u]['prompt'])[1]['candidate_assets'] != payload(split[u]['prompt'])[1]['candidate_assets']]
sizes = Counter(len(payload(split[u]['prompt'])[1]['candidate_assets']) for u in manifest['users'])
starter_order = [u for u in manifest['users']
                 if [c['ISIN'] for c in payload(split[u]['prompt'])[1]['candidate_assets']]
                 != [c['id'] for c in contexts[u]['candidate_assets']]]
report['b_candidates'] = {'users_with_different_candidate_assets': len(mismatch),
                          'candidate_list_sizes': dict(sizes),
                          'users_where_split_order_differs_from_starter_CBF_order': len(starter_order)}

# (c) long-term counts == starter history; also window-mode source (pre_cutoff_buys over events)
c_hist, c_events, empty_short = [], [], 0
for u in manifest['users']:
    body = payload(split[u]['prompt'])[1]
    shown = {r['ISIN']: r['buy_count'] for r in body['long_term_preferences']['buy_counts']}
    starter_counts = {i: n for i, n in history.get(u, {}).items()}
    if shown != starter_counts:
        c_hist.append(u)
    derived = dict(Counter(i for _d, i in ranker.pre_cutoff_buys(events.get(u, ()), assets, CUTOFF)))
    if derived != starter_counts:
        c_events.append(u)
    empty_short += not body['short_term_interest']['buy_events_oldest_to_newest']
report['c_long_term'] = {
    'users_split_long_term_ne_starter_history': len(c_hist),
    'users_pre_cutoff_buys_counts_ne_starter_history': len(c_events),
    'users_with_history_assets_outside_universe': sum(any(i not in assets for i in history.get(u, {}))
                                                     for u in manifest['users']),
    'users_with_empty_short_term_list': empty_short}

# last_k equivalence on real data: regroup split -> mixed and compare
equal = 0
for u in manifest['users']:
    mh, mb, _ = payload(mixed[u]['prompt'])
    sh, sb, _ = payload(split[u]['prompt'])
    lt, st = sb.pop('long_term_preferences'), sb.pop('short_term_interest')
    sb.update(buy_history_counts=lt['buy_counts'],
              recent_buy_events_oldest_to_newest=st['buy_events_oldest_to_newest'],
              most_recent_buy=st['most_recent_buy'])
    rebuilt = sh.replace(ranker.SPLIT_LAST_K_INSTRUCTION, ranker.MIXED_HISTORY_INSTRUCTION) + '\n' + \
        json.dumps(sb, ensure_ascii=False, sort_keys=True)
    equal += rebuilt == mixed[u]['prompt']
report['last_k_regrouped_split_equals_mixed_bytes_users'] = equal

# (d) lengths (characters; no tokenizer available)
for name, rows in (('mixed', mixed), ('split', split)):
    lengths = sorted(len(r['prompt']) for r in rows.values())
    report[f'd_prompt_chars_{name}'] = {'min': lengths[0], 'median': statistics.median(lengths),
                                        'max': lengths[-1], 'unit': 'characters (no tokenizer)'}
diff = [len(split[u]['prompt']) - len(mixed[u]['prompt']) for u in manifest['users']]
report['d_split_minus_mixed_chars'] = {'min': min(diff), 'max': max(diff)}

# (e) truth
truth_users = [u for u in manifest['users'] if truth.get(u)]
relevant_users = [u for u in manifest['users']
                  if truth.get(u, set()) & set(contexts[u]['eligible_asset_ids'])]
report['e_truth'] = {'users_with_nonempty_truth': len(truth_users),
                     'users_with_nonempty_eligible_truth_used_in_metrics': len(relevant_users)}

# 4. pick truth user with the shortest split prompt
end = (date.fromisoformat(CUTOFF) + timedelta(days=30)).isoformat()
chosen = min(relevant_users, key=lambda u: (len(split[u]['prompt']), u))
truth_rows = sorted({(r['timestamp'], r['ISIN'], r['transactionID'])
                     for r in starter.rows(data_dir / 'transactions.csv')
                     if r['customerID'] == chosen and r['transactionType'] == 'Buy'
                     and CUTOFF <= r['timestamp'][:10] < end})
prompt = split[chosen]['prompt']
body = payload(prompt)[1]
history_blocks = json.dumps([body['long_term_preferences'], body['short_term_interest']])
candidates = {c['ISIN'] for c in body['candidate_assets']}
report['example'] = {
    'user_id': chosen,
    'truth_isins': sorted(truth[chosen]),
    'truth_buy_rows_timestamp_isin_txid': truth_rows,
    'truth_dates_present_in_prompt': sorted({t[:10] for t, _, _ in truth_rows if t[:10] in prompt}),
    'truth_timestamps_present_in_prompt': sorted({t for t, _, _ in truth_rows if t in prompt}),
    'truth_isins_in_history_blocks': sorted(i for i in truth[chosen] if i in history_blocks),
    'truth_isins_in_candidates': sorted(i for i in truth[chosen] if i in candidates),
    'truth_isins_pre_cutoff_buy_count': {i: history.get(chosen, {}).get(i, 0) for i in sorted(truth[chosen])},
    'prompt_chars': len(prompt)}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
(out_json.parent / 'example_split_prompt.txt').write_text(prompt, encoding='utf-8')
print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
