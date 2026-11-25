"""Fast local smoke/full runner with MLX batching and resumable artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import data_adapter
import ranker


PINNED_REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
DEFAULT_MLX_MODEL = '/Users/davidkk/llm4rec-hw2/RecSys-LLMs/outputs/hw8/mlx_qwen3_1p7b_4bit'


class MLXBatchGenerator:
    def __init__(self, model_path, max_new_tokens=96):
        from mlx_lm import batch_generate, load
        from mlx_lm.sample_utils import make_sampler

        self.model, self.tokenizer = load(model_path)
        self.batch_generate = batch_generate
        self.sampler = make_sampler(temp=0.0)
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.metadata = {
            'model': 'Qwen/Qwen3-1.7B',
            'requested_revision': PINNED_REVISION,
            'resolved_revision': PINNED_REVISION,
            'device': 'mlx_metal',
            'dtype': 'mlx_4bit_g64',
            'quantization': {'bits': 4, 'group_size': 64},
            'backend': 'mlx_lm.batch_generate',
            'mlx_model_path': model_path,
            'do_sample': False,
            'thinking': False,
            'context_limit': 8192,
            'max_new_tokens': max_new_tokens,
        }

    def render(self, prompt):
        text = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tokenizer.encode(text)

    def generate_batch(self, prompts, max_new_tokens=None):
        token_prompts = [self.render(p) for p in prompts]
        max_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        started = time.perf_counter()
        result = self.batch_generate(
            self.model, self.tokenizer, token_prompts,
            max_tokens=max_tokens, sampler=self.sampler, verbose=False)
        wall = time.perf_counter() - started
        texts = list(result.texts)
        prompt_tokens = [len(p) for p in token_prompts]
        gen_tokens = [len(self.tokenizer.encode(t, add_special_tokens=False)) for t in texts]
        return {
            'texts': texts,
            'prompt_tokens': prompt_tokens,
            'generation_tokens': gen_tokens,
            'wall_seconds': wall,
        }


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path, record):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def candidate_hash(ids):
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()


def salvage_truncated_ranking(text, allowed):
    """Best-effort recovery for truncated JSON arrays; returns (ranking, repaired) or raises."""
    import re
    matches = re.findall(r'"([A-Z0-9]{12})"', text)
    seen, ranking = set(), []
    for isin in matches:
        if isin in allowed and isin not in seen:
            seen.add(isin)
            ranking.append(isin)
        if len(ranking) == 10:
            return ranking, True
    raise ValueError('No complete ranking recovered from truncated response')


def build_manifest(starter, contexts, baseline, manifest, history_mode, short_term_days, history_events):
    return {
        'starter': starter, 'contexts': contexts, 'baseline': baseline,
        'manifest': manifest, 'history_mode': history_mode,
        'short_term_days': short_term_days, 'history_events': history_events,
    }


def prepare_prompts(bundle, assets, history, events, history_events, short_term_days, history_mode):
    starter, _a, _h, _t, contexts, baseline, _e, _p, manifest = bundle
    prompts, evidence_hashes = [], {}
    for uid in manifest['users']:
        evidence = ranker.canonical_evidence(contexts[uid], assets, history.get(uid, {}),
                                             events.get(uid, ()), history_events)
        evidence_hashes[uid] = evidence['sha256']
        prompts.append((uid, ranker.build_prompt(contexts[uid], assets, history.get(uid, {}),
                                                 events.get(uid, ()), history_events,
                                                 None, history_mode, short_term_days)))
    return prompts, evidence_hashes, manifest['users'][0] if manifest['users'] else None


def load_existing(output_dir):
    current = []
    by_uid = {}
    for record in read_jsonl(output_dir / 'ranker_responses.jsonl'):
        uid = record['user_id']
        if uid in by_uid:
            current[by_uid[uid]] = record
        else:
            by_uid[uid] = len(current)
            current.append(record)
    return current, by_uid


def evaluate_and_save(args, bundle, predictions, failures, method, runtime_metadata, output_dir):
    starter, assets, history, truth, contexts, baseline, events, pop, manifest = bundle
    records = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'user_id', 'rank', 'ISIN'])
        for uid in manifest['users']:
            combined = dict(baseline[uid])
            combined.update(predictions.get(uid, {}))
            for name, ranking in combined.items():
                starter.validate_ranking(ranking, contexts[uid]['eligible_asset_ids'])
                relevant = truth.get(uid, set()) & set(contexts[uid]['eligible_asset_ids'])
                records.append({'user_id': uid, 'method': name,
                                'relevant_count': len(relevant),
                                'ndcg_at_10': starter.ndcg10(ranking, relevant)})
                writer.writerows((name, uid, rank, isin)
                                 for rank, isin in enumerate(ranking, 1))
    names = sorted({r['method'] for r in records})
    metrics = {name: starter.aggregate([r for r in records if r['method'] == name]) for name in names}
    for name, data in (('metrics.json', metrics), ('per_user.json', records),
                       ('run_manifest.json', dict(manifest, method=method,
                                                  runtime=runtime_metadata,
                                                  failures=failures))):
        (output_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    return metrics


def run_arm(args, bundle, generator, history_mode, output_dir, batch_size):
    starter, assets, history, _truth, contexts, baseline, events, _pop, manifest = bundle
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts, evidence_hashes, first_user = prepare_prompts(
        bundle, assets, history, events, args.history_events, args.short_term_days, history_mode)
    boundary = (ranker.temporal_boundary(contexts[first_user]['cutoff'], args.short_term_days)
                if history_mode != 'legacy' and first_user else None)

    current, by_uid = load_existing(output_dir)
    done_uids = set(by_uid)
    pending = [(uid, prompt) for uid, prompt in prompts if uid not in done_uids]
    predictions = {}
    for record in current:
        uid = record['user_id']
        ids = ranker.candidate_ids(contexts[uid])
        ranking = record.get('ranking')
        if isinstance(ranking, list) and len(ranking) == 10:
            predictions[uid] = {'llmrank': ranking, 'candidate_order_first10': ids[:10]}
    failures = [dict(r) for r in current if r.get('fallback') or r.get('error')]

    generation_records = read_jsonl(output_dir / 'generations.jsonl')
    total_new = sum(not r.get('cache_hit', False) for r in generation_records)
    wall_gen = sum(float(r.get('response', {}).get('seconds', 0.0)) for r in generation_records)
    invalid_count = sum(1 for r in current if r.get('error') or r.get('fallback'))
    repair_count = sum(1 for r in current if r.get('parse_repaired'))

    start = time.perf_counter()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        batch_prompts = [p for _, p in batch]
        t0 = time.perf_counter()
        result = generator.generate_batch(batch_prompts, max_new_tokens=generator.max_new_tokens)
        batch_wall = time.perf_counter() - t0
        if batch_wall <= 0:
            batch_wall = result.get('wall_seconds', 0.0) or 0.001
        for i, (uid, prompt) in enumerate(batch):
            ids = ranker.candidate_ids(contexts[uid])
            text = result['texts'][i]
            pt = int(result['prompt_tokens'][i]) if i < len(result['prompt_tokens']) else 0
            gt = int(result['generation_tokens'][i]) if i < len(result['generation_tokens']) else 0
            seconds = batch_wall / max(1, len(batch))
            response = {'text': text, 'input_tokens': pt, 'output_tokens': gt, 'seconds': seconds}
            record = {'user_id': uid, 'raw_response': text, 'parse_repaired': False,
                      'fallback': False, 'error': None}
            try:
                ranking, repaired = ranker.parse_ranking(text, set(ids))
                record['parse_repaired'] = repaired
                repair_count += int(repaired)
                starter.validate_ranking(ranking, ids)
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                try:
                    ranking, repaired = salvage_truncated_ranking(text, set(ids))
                    record['parse_repaired'] = repaired
                    repair_count += int(repaired)
                    starter.validate_ranking(ranking, ids)
                    record['salvaged_truncated'] = True
                except Exception:
                    invalid_count += 1
                    record['error'] = f'{type(exc).__name__}: {exc}'
                    record['fallback'] = True
                    ranking = ids[:10]
                    failures.append(dict(record))
            record['ranking'] = ranking
            record['raw_invalid_count'] = invalid_count
            record['parse_repair_count'] = repair_count
            predictions[uid] = {'llmrank': ranking, 'candidate_order_first10': ids[:10]}
            append_jsonl(output_dir / 'ranker_responses.jsonl', record)
            append_jsonl(output_dir / 'final_responses.jsonl', record)
            gen_record = {
                'user_id': uid,
                'settings': generator.metadata,
                'cache_hit': False,
                'response': response,
            }
            append_jsonl(output_dir / 'generations.jsonl', gen_record)
            generation_records.append(gen_record)
            current.append(record)
            total_new += 1
            wall_gen += seconds

        metrics = evaluate_and_save(args, bundle, predictions, failures, 'llmrank',
                                    build_runtime_metadata(generator, args, history_mode,
                                                           boundary, evidence_hashes,
                                                           total_new, invalid_count,
                                                           repair_count),
                                    output_dir)
        done = len(predictions)
        elapsed = time.perf_counter() - start
        rate = done / elapsed * 60 if elapsed > 0 else 0
        print(json.dumps({
            'arm': history_mode, 'batch_done': done, 'total': len(prompts),
            'batch_wall_seconds': round(batch_wall, 2),
            'cumulative_wall_seconds': round(elapsed, 2),
            'rate_per_min': round(rate, 2),
            'llmrank_ndcg': metrics.get('llmrank', {}).get('ndcg_at_10'),
        }), flush=True)
        (output_dir / 'progress.json').write_text(json.dumps({
            'arm': history_mode, 'done': done, 'total': len(prompts),
            'elapsed_seconds': elapsed, 'rate_per_min': rate,
        }, indent=2) + '\n')

    runtime = build_runtime_metadata(generator, args, history_mode, boundary, evidence_hashes,
                                     total_new, invalid_count, repair_count)
    metrics = evaluate_and_save(args, bundle, predictions, failures, 'llmrank', runtime, output_dir)
    return {'metrics': metrics, 'runtime': runtime, 'predictions': predictions,
            'done': len(predictions), 'total': len(prompts)}


def build_runtime_metadata(generator, args, history_mode, boundary, evidence_hashes,
                           new_calls, invalid_count, repair_count):
    aggregate = hashlib.sha256(
        json.dumps(sorted(evidence_hashes.items()), sort_keys=True).encode()).hexdigest()
    return dict(generator.metadata,
                cache_hits=0, new_calls=new_calls,
                raw_invalid_count=invalid_count, parse_repair_count=repair_count,
                shuffle_seed=getattr(args, 'shuffle_seed', None),
                history_events=args.history_events,
                adaptation='sequential_recency_only',
                history_mode=history_mode,
                prompt_version=ranker.PROMPT_VERSIONS[history_mode],
                short_term_days=args.short_term_days if history_mode != 'legacy' else None,
                temporal_boundary=boundary,
                evidence_hashes=evidence_hashes,
                evidence_aggregate_sha256=aggregate,
                retry_invalid=0, retry_attempts=0, retry_recovered=0,
                final_invalid_after_retry=invalid_count,
                response_log='final_responses.jsonl',
                history_log='ranker_responses.jsonl',
                protocol_note=f'Fast path: mlx_lm.batch_generate, 4-bit quantized local conversion from pinned HF snapshot, temp=0, thinking=False, max_new_tokens={generator.max_new_tokens}; truncated-array salvage before documented fallback; applied identically to both arms.')


def parse_args(argv=None):
    parser = data_adapter.arguments('Fast HW8 smoke/full runner')
    parser.add_argument('--history-events', type=int, default=15)
    parser.add_argument('--history-mode', choices=['unified', 'split', 'both'], default='both')
    parser.add_argument('--short-term-days', type=int, default=90)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--mlx-model', default=DEFAULT_MLX_MODEL)
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--benchmark-users', type=int, default=3)
    return parser.parse_args(argv)


def summarize_arm(result, elapsed):
    runtime = result['runtime']
    users = result['total']
    return {
        'users': users, 'wall_seconds': round(elapsed, 2),
        'new_calls': runtime['new_calls'],
        'input_tokens': sum(r['response']['input_tokens'] for r in read_jsonl(Path(runtime.get('_output_dir', '')) / 'generations.jsonl')) if runtime.get('_output_dir') else None,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.benchmark_only:
        args.max_users = max(1, args.benchmark_users)
    bundle = data_adapter.load(args)
    arms = ['unified', 'split'] if args.history_mode == 'both' else [args.history_mode]
    generator = MLXBatchGenerator(args.mlx_model, max_new_tokens=args.max_new_tokens)

    summary = {}
    for arm in arms:
        output_dir = args.output_dir / arm
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result = run_arm(args, bundle, generator, arm, output_dir, args.batch_size)
        elapsed = time.perf_counter() - started
        gens = read_jsonl(output_dir / 'generations.jsonl')
        summary[arm] = {
            'users': result['total'], 'done': result['done'],
            'wall_seconds': round(elapsed, 2),
            'rate_per_min': round(result['done'] / elapsed * 60, 2) if elapsed else None,
            'new_calls': result['runtime']['new_calls'],
            'cache_hits': 0,
            'raw_invalid_count': result['runtime']['raw_invalid_count'],
            'parse_repair_count': result['runtime']['parse_repair_count'],
            'final_invalid_after_retry': result['runtime']['final_invalid_after_retry'],
            'input_tokens': sum(int(g['response']['input_tokens']) for g in gens),
            'output_tokens': sum(int(g['response']['output_tokens']) for g in gens),
            'llmrank_ndcg_at_10': result['metrics'].get('llmrank', {}).get('ndcg_at_10'),
            'output_dir': str(output_dir),
        }
        (output_dir / 'smoke_summary.json').write_text(json.dumps(summary[arm], indent=2) + '\n')

    (args.output_dir / 'fast_run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
