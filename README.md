# LLM4Rec FAR-Trans project (HSE, GSB cohort)

**Student:** Kambetov David (BASBD-251). **Track:** Ranker.

**Research question (HW2).** We test whether a single retry with an error-specific
correction prompt improves nDCG@10 and output validity over the fallback-only LLM
ranker, for the 100 warm FAR-Trans users of the starter protocol.

## Contents
- `project/far-trans`: course starter (data loading, four baselines, nDCG@10), unmodified.
- `project/far-trans-llm`: course LLM baselines plus the HW2 change:
  - `ranker.py`: `--retry-invalid {0,1}` and `build_retry_prompt()`. When the first
    answer fails parsing/validation, the model gets one correction request containing
    the validator error and its previous answer; if that also fails, the original CBF
    fallback is used. One run writes both `llmrank` (control, first attempt only) and
    `llmrank_retry` (ours).
  - `data_adapter.py`: `mps` added to `--device` choices (Apple Silicon).
  - `test_retry.py`: CPU tests with a fake generator.

Upstream: https://github.com/dryjins/RecSys-LLMs at `f6850861ea20a51e1fb018ee4e5f840c2ac759e0`
(MIT licence, `LICENSE.upstream`). The first commit is the unmodified upstream copy, so
`git diff HEAD~1` after it shows exactly the HW2 change.

## Reproduce
Download `FAR-Trans.zip` from https://researchdata.gla.ac.uk/1658/ and unzip into
`data/FAR-Trans` at the repository root (not committed).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 transformers==4.57.6

# Baselines
cd project/far-trans
python starter.py run --data-dir ../../data/FAR-Trans --output-dir outputs/baseline

# Tests (no model needed)
cd ../far-trans-llm
python -m unittest discover -s . -p 'test_*.py'

# Control + retry in one run (first run: add --allow-download)
python ranker.py --data-dir ../../data/FAR-Trans --output-dir outputs/ranker_retry \
  --max-users 100 --candidate-count 50 \
  --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --device mps --on-invalid fallback --retry-invalid 1
```

Measured environment: MacBook Pro, Apple M1 Pro, 16 GB; Python 3.12 (arm64);
torch 2.11.0; transformers 4.57.6; MPS, float32. Not investment advice.
