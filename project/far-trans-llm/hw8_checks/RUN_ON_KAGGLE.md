# HW8: run mixed vs split (last_k) ranker inference on a Kaggle T4

Copy-paste cells for a Kaggle notebook with a **T4 GPU** accelerator and
**Internet on** (needed for `pip install` and the first model download).
Replace every `<...>` placeholder under `/kaggle/input/` with your dataset names.

> **Status: none of these commands has been run on Kaggle yet.** The commands
> match the local `--prepare-only` runs, which were checked. Package
> availability, T4 compatibility and runtime on Kaggle are **not verified**.
> See "Known risks" before starting.

Fixed settings (same as the local prepare-only checks): 100 warm users, cutoff
2021-06-01, 30-day horizon, 50 fixed CBF candidates, `Qwen/Qwen3-1.7B` at
revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, greedy, thinking off (both
fixed in `runtime.py`), `--on-invalid fallback`, `PYTHONHASHSEED=0`.

## 1. Paths (edit once)

```bash
%%bash
cat > /kaggle/working/hw8_env.sh <<'EOF'
export REPO_SRC=/kaggle/input/<repo-dataset>/RecSys-LLMs     # uploaded repo checkout
export DATA_DIR=/kaggle/input/<far-trans-dataset>/FAR-Trans  # transactions.csv, asset_information.csv, ...
export PKG=/kaggle/working/RecSys-LLMs/project/far-trans-llm
export PYTHONHASHSEED=0   # required: ranker.py refuses to run without a fixed seed
export DTYPE=auto         # auto = bfloat16 on CUDA; set float16 only if bf16 fails (see step 6)
EOF
cat /kaggle/working/hw8_env.sh
```

## 2. Copy the code to a writable folder

`/kaggle/input` is read-only and outputs are written relative to the package.

```bash
%%bash
set -euo pipefail
source /kaggle/working/hw8_env.sh
rm -rf /kaggle/working/RecSys-LLMs
cp -r "$REPO_SRC" /kaggle/working/RecSys-LLMs
cd "$PKG" && git log --oneline -1 2>/dev/null || echo "(no .git in the upload; record the commit hash by hand)"
ls "$DATA_DIR"
```

## 3. Install the pinned packages (from `requirements.txt`)

```bash
%%bash
set -euo pipefail
pip install -q 'torch==2.11.0' 'transformers==4.57.6'
python - <<'EOF'
import torch, transformers
print('torch', torch.__version__, '| transformers', transformers.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0))
    print('bf16 supported (torch reports)', torch.cuda.is_bf16_supported())
EOF
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
```

**Stop here** if `cuda available` is `False` or the install failed. Do not fall
back to CPU or to a different torch build without noting it as a deviation.

## 4. Data integrity and tests (no model)

```bash
%%bash
set -euo pipefail
source /kaggle/working/hw8_env.sh
cd "$DATA_DIR" && sha256sum transactions.csv asset_information.csv customer_information.csv close_prices.csv
cd "$PKG" && python -m unittest discover -s . -p 'test_*.py' 2>&1 | tail -3
```

Expected SHA256 (copied from the local `outputs/hw8_baselines/protocol.json`,
`source_sha256`):

| File | SHA256 |
|---|---|
| transactions.csv | `358871fa1311a802799df86e56c3d633513566965b01412d058e0f8fbe6cd54b` |
| asset_information.csv | `33f9897b14087068fba53a0bc9e24a0031b096bd3ad3e977e534153f29b7a590` |
| customer_information.csv | `c4ad52ebca90e32bf3df5eb64f362b7a3437b77d978b22627cc46927fe7064af` |
| close_prices.csv | `6ba1fe8390b31dfb3ab4e3e824910f9e8da4684cb6074cdb82eb6422a323656d` |

## 5. Pre-flight on Kaggle: prepare-only for both formats, then verify

This uses Kaggle's own Python, which may hash strings differently from the local
Python 3.10.4 even with the same `PYTHONHASHSEED`. So the candidate order must be
re-checked here, between the two formats, before spending GPU time.

```bash
%%bash
set -euo pipefail
source /kaggle/working/hw8_env.sh
cd "$PKG"
COMMON="--data-dir $DATA_DIR --max-users 100 --candidate-count 50 --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --on-invalid fallback --prepare-only"
python ranker.py $COMMON --output-dir outputs/hw8_prep_mixed --history-format mixed
python ranker.py $COMMON --output-dir outputs/hw8_prep_split --history-format split --short-term-mode last_k
python hw8_checks/hw8_verify_prep.py "$DATA_DIR" outputs/hw8_prep_mixed outputs/hw8_prep_split outputs/hw8_prep_checks/report.json \
  | python -c "import json,sys; r=json.load(sys.stdin); print({k: r[k] for k in ('same_user_order','b_candidates','c_long_term','last_k_regrouped_split_equals_mixed_bytes_users','e_truth')})"
python hw8_checks/hw8_candidate_stability.py "$DATA_DIR" outputs/hw8_prep_checks/candidates_seed0.json
```

Continue only if: `same_user_order` is `True`, `users_with_different_candidate_assets`
is `0`, both `c_long_term` mismatch counts are `0`,
`last_k_regrouped_split_equals_mixed_bytes_users` is `100`, and
`users_with_nonempty_truth` is `11`. Locally (Python 3.10.4, `PYTHONHASHSEED=0`) the
stability script printed `cohort_digest f7f33169daeb`, `candidates_digest 68a3b833a890`.
A different candidates digest on Kaggle means the candidate *order* differs from the
local prepare-only files. Mixed and split are still comparable with each other (same
run, same seed), but record it as a deviation.

## 6. Inference: mixed → `outputs/hw8_mixed`

**dtype:** try `--dtype auto` (bfloat16) first; switch `DTYPE=float16` in step 1 for BOTH mixed and split only if bf16 fails, rerun both into fresh output dirs, and record which dtype was used (`run_manifest.json` → `runtime.requested_dtype` and `runtime.dtype`).
A bf16 failure may not crash the run: errors during generation are caught by
`--on-invalid fallback` and show up as invalid outputs. Check `failures` in
`run_manifest.json` for `RuntimeError` messages before trusting a completed run.

```bash
%%bash
set -euo pipefail
source /kaggle/working/hw8_env.sh
cd "$PKG"
python ranker.py \
  --data-dir "$DATA_DIR" \
  --output-dir outputs/hw8_mixed \
  --max-users 100 --candidate-count 50 \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --allow-download --device cuda --on-invalid fallback \
  --dtype "$DTYPE" \
  --history-format mixed
```

## 7. Inference: split (last_k) → `outputs/hw8_split`

```bash
%%bash
set -euo pipefail
source /kaggle/working/hw8_env.sh
cd "$PKG"
python ranker.py \
  --data-dir "$DATA_DIR" \
  --output-dir outputs/hw8_split \
  --max-users 100 --candidate-count 50 \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --allow-download --device cuda --on-invalid fallback \
  --dtype "$DTYPE" \
  --history-format split --short-term-mode last_k
```

Use fresh output directories for any rerun. A partially completed directory holds
cached generations (`generation_cache/`) and will reuse them.

## 8. Collect numbers (copy from these files; do not retype)

```bash
%%bash
set -euo pipefail
source /kaggle/working/hw8_env.sh
cd "$PKG"
python - <<'EOF'
import json
for run in ('hw8_mixed', 'hw8_split'):
    m = json.load(open(f'outputs/{run}/metrics.json'))
    r = json.load(open(f'outputs/{run}/run_manifest.json'))
    rt = r['runtime']
    print(f'== outputs/{run}  method={r["method"]}')
    for name, v in sorted(m.items()):
        print(f'  {name:24s} nDCG@10 all={v["ndcg_at_10"]:.4f}  nonempty={v["ndcg_at_10_nonempty"]}')
    print('  raw_invalid_count', rt['raw_invalid_count'], '| parse_repair_count', rt['parse_repair_count'],
          '| failures listed', len(r['failures']), '| python_hash_seed', rt.get('python_hash_seed'))
    print('  torch', rt.get('torch_version'), '| transformers', rt.get('transformers_version'),
          '| requested_dtype', rt.get('requested_dtype'), '| dtype', rt.get('dtype'),
          '| resolved_revision', rt.get('resolved_revision'))
    print('  data_sha256', r['data_sha256'])
EOF
cd outputs && tar czf /kaggle/working/hw8_outputs.tgz hw8_prep_checks hw8_mixed hw8_split
ls -la /kaggle/working/hw8_outputs.tgz
```

Download `hw8_outputs.tgz` from the notebook's Output panel. Each `metrics.json`
also contains the four starter baselines (Popularity, CBF, Latest buy, Latest buy
or sell) and `candidate_order_first10` for the same users, so check that the baselines
match HW2 (0.0286 / 0.0201 / 0.0277 / 0.0271). Keep all of `outputs/` out of git.
Report invalid/fallback counts separately from nDCG; the nDCG of both LLM methods
includes the fallbacks.

## Known risks (not verified)

- **bfloat16 on T4 (precaution, not an observed failure).** The default `--dtype auto`
  loads bfloat16 on CUDA (`runtime.py`). The T4 (compute capability 7.5) has no native
  bf16 support; whether bf16 works and how fast it runs there with the pinned torch is
  unverified. `--dtype float16` exists only as a fallback: use it for BOTH runs or for
  neither, so mixed and split stay comparable. HW2 used bfloat16 on an RTX 5070 Ti, and
  the T4 and any float16 fallback may change generations, so compare HW8 mixed with
  HW8 split, not with the HW2 LLMRank number. Whether float16 causes numerical problems
  (e.g. empty or garbled output) with this model is also unverified; check `failures`
  and a few `raw_response` values in `final_responses.jsonl`. `run_manifest.json`
  records `runtime.requested_dtype` (the flag) and `runtime.dtype` (the torch dtype
  actually loaded).
- **torch 2.11.0 wheel.** The default PyPI wheel's CUDA build must support both the T4
  and Kaggle's driver. This was not checked; the local measurement was `2.11.0+cu128`.
- **Context budget.** Prompts are 12,628–19,045 characters locally (no tokenizer was
  available). A prompt over 8192 − 512 tokens raises an error that `--on-invalid
  fallback` turns into an invalid output (`runtime.py:45-46`); check `failures`.
- **Hash seed.** `PYTHONHASHSEED=0` must be set before Python starts (step 1 exports it).
  `ranker.py` exits with an error otherwise, and records the value in
  `run_manifest.json` → `runtime.python_hash_seed`.
- **`--allow-download` on both runs** follows the request; the second run should find
  the model already cached.
