"""Paired FAR-Trans prompt comparison. See README for the starter invocation."""
import argparse
import hashlib
import json
import string
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = string.ascii_uppercase + string.ascii_lowercase[:24]


def prompt_pair(user, context, events, assets):
    cutoff = context['cutoff']
    boundary = (date.fromisoformat(cutoff) - timedelta(days=90)).isoformat()
    purchases = sorted((d, i) for d, i, kind in events if kind == 'Buy' and d < cutoff)[-30:]
    pool = [c['id'] for c in context['candidate_assets']]
    assert len(set(pool)) == len(LABELS) == len(pool)
    permutation = sorted(pool, key=lambda i: hashlib.sha256((user + ':' + i).encode()).digest())
    labels = dict(zip(permutation, LABELS))
    def describe(i):
        a = assets[i]
        return ' | '.join(a.get(k) or '?' for k in ('assetName','assetCategory','assetSubCategory'))
    candidates = '\n'.join(f'{labels[i]} = {describe(i)}' for i in pool)
    entries = [(d, f'{d} {labels.get(i, "-")} {describe(i)}') for d,i in purchases]
    older = [s for d,s in entries if d < boundary]
    recent = [s for d,s in entries if d >= boundary]
    flat = [s for _,s in entries]
    assert older + recent == flat
    histories = {'flat': '\n'.join(flat),
                 'split': 'Long-term purchases:\n' + '\n'.join(older) +
                          '\nShort-term purchases:\n' + '\n'.join(recent)}
    prompts = {arm: f'Predict the next asset purchase as of {cutoff}. Repeated purchases are allowed. '
               'Use the dated purchases and the candidate descriptions. Return the single letter of the best candidate.\n'
               + 'Purchases (oldest first):\n' + h + '\nCandidates:\n' + candidates for arm,h in histories.items()}
    return prompts, permutation, [len(older),len(recent)]


def main():
    parser = argparse.ArgumentParser()
    for field in ('starter','data','baseline','model','out'):
        parser.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(args.starter.resolve()))
    import starter
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    protocol = json.loads((args.baseline/'protocol.json').read_text())
    contexts = [json.loads(l) for l in (args.baseline/'contexts.jsonl').read_text().splitlines()]
    assert [c['user_id'] for c in contexts] == protocol['users']
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((args.data/name).read_bytes()).hexdigest()==digest
    aset = set(protocol['eligible_assets'])
    assets = starter.latest_before(args.data/'asset_information.csv','ISIN',protocol['cutoff'])
    events = starter.scan_recent_events(args.data,protocol['cutoff'],aset)
    start=time.perf_counter()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.float32).eval()
    codes = [tokenizer.encode(letter,add_special_tokens=False) for letter in LABELS]
    assert all(len(c)==1 for c in codes) and len(set(c[0] for c in codes))==50
    ids = [c[0] for c in codes]
    load_seconds=time.perf_counter()-start
    timings={'flat':0.,'split':0.}
    with (out/'inference.jsonl').open('w') as log, torch.inference_mode():
        for index,ctx in enumerate(contexts):
            uid=ctx['user_id']
            prompts,mapping,counts=prompt_pair(uid,ctx,events[uid],assets)
            for arm in (['flat','split'] if index%2==0 else ['split','flat']):
                tick=time.perf_counter()
                req=tokenizer.apply_chat_template([{'role':'user','content':prompts[arm]}],tokenize=True,add_generation_prompt=True,return_tensors='pt')
                scores=None
                error=None
                try:
                    if req.shape[-1]>model.config.max_position_embeddings: raise ValueError('context overflow')
                    scores=model(input_ids=req,use_cache=False,logits_to_keep=1).logits[0,-1,ids].float()
                    if not torch.isfinite(scores).all(): raise ValueError('nonfinite logits')
                    scores=scores.tolist()
                    ranked=[mapping[j] for j in sorted(range(50),key=lambda j:(-scores[j],mapping[j]))[:10]]
                    starter.validate_ranking(ranked,mapping)
                except (ValueError,RuntimeError) as exc:
                    error=str(exc)
                    ranked=[a['id'] for a in ctx['candidate_assets'][:10]]
                elapsed=time.perf_counter()-tick
                timings[arm]+=elapsed
                log.write(json.dumps(dict(user_id=uid,arm=arm,prompt=prompts[arm],label_map=mapping,logits=scores,
                    tokens=req.shape[-1],counts=counts,ranked_ids=ranked,error=error,seconds=elapsed))+'\n')
            log.flush()
            if index%10==9: print(index+1,flush=True)
    all_rows=[json.loads(l) for l in (out/'inference.jsonl').read_text().splitlines()]
    for arm in timings:
        with (out/(arm+'.jsonl')).open('w') as f:
            for r in all_rows:
                if r['arm']==arm: f.write(json.dumps(dict(stage='ranker',user_id=r['user_id'],ranked_ids=r['ranked_ids'],protocol_hash=protocol['protocol_hash']))+'\n')
    (out/'timing.json').write_text(json.dumps(dict(load=load_seconds,arms=timings,device='cpu',dtype='float32'),indent=2))


if __name__=='__main__':
    main()
