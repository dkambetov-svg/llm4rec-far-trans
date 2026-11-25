# HW8 Frozen Experimental Protocol — Temporal History Separation in LLMRank

**Status:** FROZEN  
**Version:** 1.0  
**Date:** 2026-11-25  
**Course artifact:** Homework 2 / assignment No. 8, FAR-Trans project  
**Protocol owner:** individual submission; team-shared code must be disclosed in the PDF  
**Inference status:** NOT RUN. This protocol is fixed before any model call or evaluation result is inspected.

---

## 1. Research question and hypothesis

**Question:** Does explicitly separating long-term user preferences from short-term purchasing interests improve nDCG@10 in LLM-based financial asset ranking compared with a unified purchase history, while keeping historical transactions, asset metadata, and candidate sets identical?

- **H0:** The paired mean nDCG@10 difference between Arm B and Arm A is zero.
- **H1:** Arm B improves the paired mean nDCG@10 relative to Arm A.

The test is directional but the homework explicitly permits a negative result. A null or negative result will be reported as such.

**Predeclared design choice:** the short-term window is **90 days** before the cutoff. This value is fixed a priori and will **not** be adjusted using evaluation results.

---

## 2. Codebase anchors

All paths are relative to the repository root that contains `project/`:

- Official instructions: `project/far-trans/instructions/homework2.md`
- Classical starter and evaluation: `project/far-trans/starter.py`
- Identity extension seam (not an LLM): `project/far-trans/extensions.py`
- Starter contract tests: `project/far-trans/test_starter.py`
- Ranker baseline: `project/far-trans-llm/ranker.py`
- Shared data/evaluation adapter: `project/far-trans-llm/data_adapter.py`
- Local inference backend: `project/far-trans-llm/runtime.py`
- Ranker contract tests: `project/far-trans-llm/test_ranker.py`
- Published baseline results: `project/far-trans-llm/HW2_RESULTS.md`

Dataset directory: `/Users/davidkk/llm4rec-hw2/data/FAR-Trans` (contains `transactions.csv`, `asset_information.csv`, `customer_information.csv`, `close_prices.csv`).

---

## 3. Experimental arms

### 3.1 Frozen shared selection stage (applied before serialization)

Both arms receive exactly the same records. The selection stage is performed once per user and reused; no arm-specific filtering, truncation, scoring, or feature engineering is permitted.

1. **Users:** `data_adapter.load()` calls `starter.select_users(profiles, history, 'warm', 100, None)`. This is the official deterministic 100-user warm sample (SHA-256 seed `42` ordering), selected without using future truth.
2. **History events:** `starter.scan_recent_events()` returns pre-cutoff `(date, ISIN, transactionType)` rows. Both arms use only rows with `transactionType == 'Buy'`, `date < cutoff`, and `ISIN in assets`. Sell rows remain excluded from ranker history, as in the baseline.
3. **Full counts:** `starter.load_data()` produces the pre-cutoff Buy `Counter` per user. Restricted to eligible assets this is the ISIN-level summary of the same event multiset.
4. **Shared dated truncation:** from the filtered event list, sort by `(date, ISIN)` and retain the **last 15** dated Buy events (`--history-events 15`). This is the only truncation of dated events, and it is identical in both arms.
5. **Candidate set:** `starter.contexts_for()` builds CBF rankings and exposes the top **50** eligible ISINs as `context['candidate_assets']` (`--candidate-count 50`). `ranker.candidate_ids()` validates them. Candidate membership **and presentation order** are identical in both arms. `--shuffle-seed` is omitted in both arms; `--exclude-seen` is not used.
6. **Temporal boundary:** `B = 2021-03-03`, i.e. `cutoff (2021-06-01) − 90 days`. Day-level dates are inclusive on the left and exclusive at the cutoff:
   - **long-term:** `date < 2021-03-03`
   - **short-term:** `2021-03-03 <= date < 2021-06-01`

### 3.2 Arm A — unified purchase history (control)

Arm A is the **unchanged existing ranker serialization** in `ranker.prompt_payload()`:

- `buy_history_counts`: the full shared pre-cutoff Buy counts, each rendered with `asset_record()` fields (`ISIN`, as-of asset metadata, `buy_count`).
- `recent_buy_events_oldest_to_newest`: the shared last-15 dated Buy list.
- `most_recent_buy`: the last item of that shared list, or the string `"unknown"` when empty.
- Prompt instruction: the exact current `ranker.build_prompt()` instruction text.

No temporal labels or new fields are introduced in Arm A.

### 3.3 Arm B — explicit long-term and short-term sections (treatment)

Arm B uses the **same shared selection stage** and the **same per-record fields**; it only regroups and labels already-selected records.

- `long_term_history.buy_counts`: counts derived from the same full event multiset restricted to `date < 2021-03-03`, rendered with the same `asset_record()` + `buy_count` fields.
- `long_term_history.dated_events`: those entries of the shared last-15 dated list whose date is `< 2021-03-03`, rendered as `{'date': ..., 'ISIN': ...}`.
- `short_term_history.buy_counts`: counts derived from the same full event multiset restricted to `2021-03-03 <= date < 2021-06-01`, rendered with the same fields.
- `short_term_history.dated_events`: those entries of the shared last-15 dated list whose date is in the short-term window, rendered as `{'date': ..., 'ISIN': ...}`.
- `most_recent_buy`: identical top-level value to Arm A, derived from the same shared dated list. It is not a new feature and is not moved between sections.

The union of Arm B's two count groups equals Arm A's `buy_history_counts`. The union of Arm B's two dated-event groups equals Arm A's `recent_buy_events_oldest_to_newest`. Therefore no transaction, timestamp, count, metadata field, candidate, or ordering is added or removed.

Arm B's instruction is Arm A's instruction plus exactly one neutral sentence: *"The two history sections contain the same purchase records; an empty section means no observed purchases in that window."* No preference, risk, profitability, suitability, or external feature is added.

### 3.4 Explicit non-additions to Arm B

Arm B must **not** receive any of the following that Arm A does not receive:

- new metadata fields, asset text, prices, news, or external lookups;
- derived popularity, trend, momentum, volatility, or recency scores;
- different profile fields or different treatment of estimated/missing values;
- a different candidate-generation rule, candidate count, or candidate order;
- a different truncation of dated events (the last-15 rule is shared);
- future transactions, future metadata, truth labels, or evaluation artifacts.

---

## 4. Controlled variables

| Variable | Frozen value |
|---|---|
| Cohort | Official deterministic warm sample of 100 users, selected by `starter.select_users` with seed `42` |
| Cutoff | `2021-06-01`, exclusive for history and metadata |
| Evaluation window | 30 days: `2021-06-01` to `2021-07-01`, exclusive |
| Label | Future `Buy` intersected with eligible assets |
| Repeat purchases | Allowed; `--exclude-seen` not used |
| Eligibility | As-of metadata plus at least one positive pre-cutoff close price |
| History source | Same pre-cutoff Buy rows, same day-level timestamps, same last-15 truncation |
| Asset metadata | Same `latest_before` as-of rows and same `asset_record()` fields |
| Candidate generation | Existing ranker policy: CBF top 50 from `starter.contexts_for` |
| Candidate order | Identical list order in both arms; no shuffle seed |
| Model | `Qwen/Qwen3-1.7B` |
| Exact revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Generation | Greedy (`do_sample=False`), thinking disabled, 8192-token context, `max_new_tokens=512` for ranker |
| Device | `--device auto`; resolved device/dtype recorded identically for both arms |
| Demos | None; `--shots-file` omitted |
| Validation | `ranker.parse_ranking()` then `starter.validate_ranking()` |
| Invalid policy | `--on-invalid fallback`; `--retry-invalid 0` |
| Metric implementation | `starter.ndcg10()` and `starter.aggregate()` unchanged |
| Classical baselines | Popularity, CBF, Latest buy, Latest buy or sell retained in both runs |

---

## 5. Edge-case policy

### Empty history sections

- In Arm B, an empty count group or dated-event group is serialized as an empty list `[]`.
- The neutral instruction states that an empty section means no observed purchases in that window; no preference is inferred from absence.
- Cold-start handling is unchanged: if the shared selection has no usable history, `starter.contexts_for()` already supplies the popularity-derived candidate order; both arms inherit the same candidate list.

### Invalid outputs

- Fenced or surrounding-text JSON extraction by `ranker.parse_ranking()` is allowed and counted as `parse_repaired`, exactly as in the baseline.
- Any output that is not ten unique string ISINs from the fixed candidate list is invalid.
- With `--on-invalid fallback`, the invalid user's `llmrank` ranking is the original candidate order `ids[:10]`; the user is **not** removed.
- Every invalid event is recorded in `ranker_responses.jsonl` and summarized in `final_responses.jsonl` / `run_manifest.json`.
- Retries are disabled (`--retry-invalid 0`) so both arms use the same decision rule.

### Context overflow

- `runtime.LocalGenerator.generate()` raises if `input_tokens + 512 > 8192`.
- Overflow is a **hard failure**, not a reason to shrink candidates, drop old events, or change the 90-day window.
- If overflow occurs, the affected arm is reported as incomplete for that user; no per-user adaptive truncation is permitted.
- Any remediation must be written as a versioned protocol amendment **before** evaluation metrics are inspected and must be applied identically to both arms.

### Users without future purchases

- All selected users remain in the primary mean. Empty truth contributes `ndcg_at_10 = 0.0` via `starter.ndcg10()`.
- The secondary mean excludes empty-truth users and reports their count (expected 11 nonempty / 89 empty in the reference 100-user sample).
- No user is removed for empty truth, invalid output, cold start, or overflow.

---

## 6. Evaluation rules

### 6.1 Metrics

For each user `u` and arm `m ∈ {A, B}`:

- `nDCG_A(u)`, `nDCG_B(u)` are computed by the unchanged `starter.ndcg10()` on the ten returned ISINs and `truth[u] ∩ eligible_asset_ids`.
- **Primary metric:** `mean over all 100 users` of `nDCG@10`.
- **Secondary metric:** `mean over users with nonempty future truth` of `nDCG@10`, plus that user count.

### 6.2 Primary comparison

The primary comparison is the **paired difference**:

```
d(u)        = nDCG_B(u) − nDCG_A(u)
Delta_all   = (1 / 100) * sum_over_all_users d(u)
Delta_nonempty = (1 / |S+|) * sum_over_u_in_S+ d(u)
```

where `S+` is the set of users with nonempty future truth. Report `Delta_all` as the primary result and `Delta_nonempty` as secondary. Also report counts of users with `d(u) > 0`, `d(u) = 0`, and `d(u) < 0`.

No significance claim is required. Metrics are reported to **four decimal places** in the PDF table; full precision remains in artifact files.

### 6.3 Baselines and reference values

Both Arm A and Arm B runs must contain the four official classical baselines written by `data_adapter.save_results()`:

- `popularity`
- `cbf`
- `latest_buy`
- `latest_buy_or_sell`

They must be reported in the results table alongside `llmrank` (Arm A) and the Arm B method label. Values in `HW2_RESULTS.md` are reference points only; new results must come from the frozen runs described here.

### 6.4 Fairness checks before inference

- Run `ranker.py --prepare-only` for both arms and compare `ranker_prompts.jsonl`.
- Verify that prompts differ **only** in the history serialization and the single neutral section sentence.
- Verify from `contexts.jsonl` / `run_manifest.json` that cohort, candidate IDs, and candidate order are identical.
- Verify that both counts/dated-event partitions reconstruct the exact shared event multiset.

---

## 7. Expected artifacts

Per arm, in a fresh output directory:

- `ranker_prompts.jsonl` from `--prepare-only`
- `predictions.csv`
- `metrics.json`
- `per_user.json`
- `run_manifest.json`
- `final_responses.jsonl`
- `ranker_responses.jsonl`
- `generations.jsonl`
- `generation_cache/`

Cross-arm audit artifacts:

- a prompt diff report showing only the allowed history-section change;
- a candidate-equality check (IDs and order);
- a data SHA-256 / cohort equality check;
- a one-user manual nDCG reproduction record for the "personally verify" requirement.

Because `ranker.py` writes `run_manifest.json` rather than the starter's `protocol.json`, the manifest is the canonical frozen settings record. Additionally, one `starter.py export-prompts` invocation with the same data/cutoff/cohort flags may be used to produce a `protocol.json` whose data hashes and settings must match both ranker manifests; full hashes stay in artifacts, not in the report's main table.

---

## 8. Acceptance criteria

1. This protocol version is fixed before any inference or evaluation-result inspection.
2. Both arms complete the same 100 users; no user is dropped for empty truth, invalid output, cold start, or overflow.
3. Data SHA-256 values, user cohort, cutoff, horizon, eligible assets, candidate IDs, and candidate order are identical across arms.
4. The shared history selection and last-15 truncation are demonstrably identical; Arm B only partitions already-selected records.
5. No Arm-B-only preference feature or external information is present.
6. Model ID, exact revision, generation settings, validation, and fallback policy are identical and recorded in both manifests.
7. Both runs contain all four classical baselines and the LLM methods.
8. Failures, repairs, fallbacks, and any overflow are counted and reported; fallback-inclusive scores are not presented as bare LLM performance.
9. The PDF reports four-decimal metrics, candidate policy, invalid handling, runtime labeling, and a documented personal verification.
10. Any deviation is treated as an incomplete experiment or a newly versioned confirmatory protocol, not as hyperparameter tuning on evaluation results.

---

## 9. Change control

- The 90-day boundary, last-15 truncation, 50 candidates, model, revision, and decoding settings are frozen in this document.
- Evaluation results must not be used to change any of them.
- A change requires a new protocol version with a dated rationale, applied identically to both arms, and reported as a separate confirmatory experiment.

---

## 10. Planned commands (documented only; not executed)

```bash
cd /Users/davidkk/llm4rec-hw2/RecSys-LLMs/project/far-trans-llm

# Arm A: unified history (prompt inspection first).
python ranker.py --prepare-only \
  --data-dir /Users/davidkk/llm4rec-hw2/data/FAR-Trans \
  --output-dir outputs/hw8-arm-a-prompts \
  --max-users 100 --candidate-count 50 --history-events 15 \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --device auto --on-invalid fallback --retry-invalid 0 \
  --history-mode unified

# Arm B: 90-day separated history (prompt inspection first).
python ranker.py --prepare-only \
  --data-dir /Users/davidkk/llm4rec-hw2/data/FAR-Trans \
  --output-dir outputs/hw8-arm-b-prompts \
  --max-users 100 --candidate-count 50 --history-events 15 \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --device auto --on-invalid fallback --retry-invalid 0 \
  --history-mode separate --split-days 90
```

The final inference commands use the same flags without `--prepare-only`, in fresh output directories `outputs/hw8-arm-a` and `outputs/hw8-arm-b`. These commands are recorded here for reproducibility and were not run while creating this protocol.
