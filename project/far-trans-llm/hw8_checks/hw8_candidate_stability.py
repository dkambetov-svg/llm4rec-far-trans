"""Print a digest of the 100 users' CBF candidate lists and CBF nDCG@10 for this process.

The starter's CBF scores depend on Python string hashing, so run this in several processes
with different (or unset) PYTHONHASHSEED values and compare candidates_digest:

    PYTHONHASHSEED=0 python hw8_checks/hw8_candidate_stability.py <data_dir> <out_json>

Reference (local, Python 3.10.4, FAR-Trans, PYTHONHASHSEED=0): cohort_digest f7f33169daeb,
candidates_digest 68a3b833a890. Other Python versions may hash differently even with the
same seed, so compare digests only between runs of the same interpreter version.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PACKAGE = Path(__file__).resolve().parent.parent  # project/far-trans-llm
sys.path.insert(0, str(PACKAGE))
import data_adapter  # noqa: E402

data_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
args = SimpleNamespace(starter_dir=PACKAGE.parent / 'far-trans', data_dir=data_dir, cutoff='2021-06-01',
                       horizon_days=30, max_users=100, candidate_count=50)
starter, assets, history, truth, contexts, baseline, events, pop, manifest = data_adapter.load(args)
cands = {u: [c['id'] for c in contexts[u]['candidate_assets']] for u in manifest['users']}
scores = []
for u in manifest['users']:
    relevant = truth.get(u, set()) & set(contexts[u]['eligible_asset_ids'])
    scores.append(starter.ndcg10(baseline[u]['cbf'], relevant))
record = {'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED', '<unset>'),
          'python_version': sys.version.split()[0],
          'users': len(manifest['users']),
          'cohort_digest': hashlib.sha256(json.dumps(manifest['users']).encode()).hexdigest()[:12],
          'candidates_digest': hashlib.sha256(json.dumps(cands, sort_keys=True).encode()).hexdigest()[:12],
          'cbf_ndcg_all': sum(scores) / len(scores), 'candidates': cands}
out.write_text(json.dumps(record), encoding='utf-8')
print(record['PYTHONHASHSEED'], record['cohort_digest'], record['candidates_digest'],
      f"cbf={record['cbf_ndcg_all']:.6f}")
