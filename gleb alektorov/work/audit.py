"""Independently check prompts, cohorts, original ISIN rankings and binary nDCG."""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
read = lambda p:json.loads(p.read_text())
proto=read(root/'baseline/protocol.json')
reviews=[json.loads(l) for l in (root/'outputs/inference.jsonl').read_text().splitlines()]
assert len(reviews)==200
truth=defaultdict(set)
eligible=set(proto['eligible_assets'])
with (root/'FAR-Trans/transactions.csv').open(encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if (row['transactionType']=='Buy' and int(row['transactionID'])>=0 and row['ISIN'] in eligible
                and proto['cutoff']<=row['timestamp'][:10]<proto['end_exclusive']):
            truth[row['customerID']].add(row['ISIN'])
paired=defaultdict(dict)
for r in reviews: paired[r['user_id']][r['arm']]=r
scores={}
cases=[]
nonempty=0
for arm in ['flat','split']:
    assert read(root/arm/'protocol.json')==proto
    deltas=[]
    for idx,uid in enumerate(proto['users']):
        a,b=paired[uid]['flat'],paired[uid]['split']
        assert b['prompt'].replace('Long-term purchases:\n','').replace('Short-term purchases:\n','').replace('\n\n','\n')==a['prompt']
        assert a['label_map']==b['label_map']
        r=paired[uid][arm]
        assert len(set(r['ranked_ids']))==10 and set(r['ranked_ids'])<=set(r['label_map'])
        relevant=truth[uid]
        ranks=[i for i,v in enumerate(r['ranked_ids'],1) if v in relevant]
        dcg=sum(1/math.log2(1+i) for i in ranks)
        idcg=sum(1/math.log2(1+i) for i in range(1,min(10,len(relevant))+1))
        score=dcg/idcg if idcg else 0
        deltas.append(score)
        cases.append(dict(index=idx+1,arm=arm,truth_count=len(relevant),hit_ranks=ranks,dcg=dcg,idcg=idcg,score=score))
    nonempty=sum(bool(truth[uid]) for uid in proto['users'])
    actual=read(root/arm/'metrics.json')['methods']['custom']
    assert math.isclose(sum(deltas)/len(deltas),actual['ndcg_at_10'],abs_tol=1e-12)
    assert math.isclose(sum(deltas)/nonempty,actual['ndcg_at_10_nonempty'],abs_tol=1e-12)
    scores[arm]=deltas
diff=[b-a for a,b in zip(scores['flat'],scores['split'])]
print(json.dumps(dict(protocol=proto['protocol_hash'],users=len(diff),nonempty=nonempty,
    gain=sum(x>0 for x in diff),loss=sum(x<0 for x in diff),tie=diff.count(0),
    both_periods=sum(all(v['flat']['counts']) for v in paired.values()),
    changed=sum(v['flat']['ranked_ids']!=v['split']['ranked_ids'] for v in paired.values()),
    example=next((r for r in cases if r['hit_ranks']),cases[0]),
    invalid={arm:sum(r['error'] is not None for r in reviews if r['arm']==arm) for arm in scores},
    context_max=max(r['tokens'] for r in reviews)),indent=2))
