# RecSys-LLMs
Repository for Recommender Systems course in HSE : Code examples, assignments, and capstone project templates using JavaScript, Hugging Face LLM API, and GitHub Pages. From random lunch recommender to personalized AI systems.

## Project

The FAR-Trans team project lives in [`project/far-trans/`](project/far-trans/README.md):
a standard-library Python starter (common CSV loader, popularity, content-based and
recency baselines, nDCG@10 evaluation) plus the cohort and session instructions under
[`project/far-trans/instructions/`](project/far-trans/instructions/). The dataset
itself is public and is not redistributed here; download it from
<https://researchdata.gla.ac.uk/1658/> and pass the extracted folder with
`--data-dir`.

Runnable open-weight [LLM Encoder and Ranker baselines](project/far-trans-llm/README.md)
are also provided as optional starting points for the project.

## HW8 — Temporal History Separation for FAR-Trans LLM Ranking

**Status:** preliminary/incomplete for paired comparison. Unified fast-run arm completed 100/100 (fallback-inclusive); split arm was incomplete at submission snapshot. Do not cite as official paired 100-user result until split completes and audit passes.

### Artifacts

- Frozen protocol: `docs/research/hw8-protocol.md`
- Preliminary report: `docs/reports/hw8-preliminary-report.md`
- Submission PDF: `outputs/hw8/hw8-report.pdf`
- Classical baselines: `outputs/hw8/baseline/metrics.json`
- Smoke validation: `outputs/hw8/smoke/smoke_validation.json`
- Fast runner: `project/far-trans-llm/fast_ranker.py`

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers mlx mlx-lm reportlab pypdf
```

Model: `Qwen/Qwen3-1.7B`, pinned revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Do not download weights without explicit authorization; use cached/local snapshot.

### Reproduce smoke (3 users, both arms)

```bash
cd project/far-trans-llm
../../.venv/bin/python ranker.py --data-dir /path/to/FAR-Trans --output-dir ../../outputs/hw8/smoke/unified --max-users 3 --candidate-count 50 --history-events 15 --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device mps --on-invalid fallback --retry-invalid 0 --history-mode unified --short-term-days 90
../../.venv/bin/python ranker.py --data-dir /path/to/FAR-Trans --output-dir ../../outputs/hw8/smoke/split --max-users 3 --candidate-count 50 --history-events 15 --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device mps --on-invalid fallback --retry-invalid 0 --history-mode split --short-term-days 90
```

### Reproduce fast MLX run (protocol amendment, identical to both arms)

```bash
cd project/far-trans-llm
../../.venv/bin/python -m mlx_lm convert --hf-path /path/to/local-or-cached-Qwen3-1.7B-snapshot --mlx-path ../../outputs/hw8/mlx_qwen3_1p7b_4bit -q --q-bits 4 --q-group-size 64 --dtype bfloat16
../../.venv/bin/python fast_ranker.py --data-dir /path/to/FAR-Trans --output-dir ../../outputs/hw8/fast_full --max-users 100 --candidate-count 50 --history-events 15 --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device mps --on-invalid fallback --history-mode both --short-term-days 90 --max-new-tokens 160 --batch-size 3
```

Fast path amendment: MLX 4-bit local conversion, greedy temp=0, `enable_thinking=False`, batch generation, max_new_tokens=160, truncated-array salvage before documented fallback; applied identically to both arms.

### Refresh report from existing artifacts (no inference)

```bash
.venv/bin/python outputs/hw8/build_report.py
```
