# HW8 Preliminary Report — Temporal History Separation in LLMRank (FAR-Trans)

**Status: PRELIMINARY / INCOMPLETE — NOT official paired 100-user comparison.**  
**Generated:** 2026-11-25 (submission deadline pressure).  
**Background inference:** active (PID in `outputs/hw8/fast_full_resume.pid`, log `outputs/hw8/fast_full_resume.log`). Unified arm reached 100/100; split arm still running at draft time. Do not treat as final until split completes 100/100 and paired audit is verified.

## 1. Research question

Does explicitly separating long-term user preferences from short-term purchasing interests improve nDCG@10 in LLM-based financial asset ranking compared with a unified purchase history, while keeping historical transactions, asset metadata, and candidate sets identical? (HW8 protocol v1.0, frozen before inference.)

- **H0:** Paired mean nDCG@10 difference (Split − Unified) is zero.
- **H1:** Split improves paired mean nDCG@10 over Unified.
- A null/negative result is acceptable and will be reported if confirmed.

## 2. Controlled protocol and dataset

| Item | Frozen value |
|---|---|
| Dataset | FAR-Trans `/Users/davidkk/llm4rec-hw2/data/FAR-Trans` (transactions, assets, customers, close prices) |
| Cohort | 100 warm users, deterministic `starter.select_users(..., seed=42)` |
| Cutoff / horizon | 2021-06-01 exclusive / 30-day future Buy window |
| Candidates | CBF top 50, identical IDs and order across arms |
| History | Pre-cutoff Buy rows only; last 15 dated Buy events, truncated before serialization |
| Temporal boundary | 90 days: long-term `< 2021-03-03`; short-term `>= 2021-03-03` and `< 2021-06-01` |
| Arm A (Unified) | One chronological history section |
| Arm B (Split) | Same records partitioned into long-term/short-term sections; no added evidence |
| Model | `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Decoding | Greedy, `do_sample=False`, thinking disabled, context 8192 |
| Invalid policy | Parse/repair allowed; invalid after repair -> documented fallback to original candidate order; user not dropped |
| Metrics | `starter.ndcg10` / `starter.aggregate`, unchanged; primary all-users mean, secondary nonempty-truth mean |

## 3. Methods

**Unified:** selected history presented as one chronological section (`history_mode=unified`).  
**Split:** same canonical evidence partitioned by 90-day boundary (`history_mode=split`); empty section allowed, no records added/removed.  
**Controls:** identical user cohort, candidates, candidate order, truncation, model/revision, decoding, validation, fallback, and classical baselines.

## 4. Evaluation metrics

- **Primary:** mean nDCG@10 over all 100 users, including empty-truth users as 0.
- **Secondary:** mean nDCG@10 over nonempty-truth users; report nonempty count.
- **Paired comparison (when complete):** `d(u)=nDCG_B(u)-nDCG_A(u)`, report `Delta_all`, `Delta_nonempty`, and counts where d>0/=0/<0.
- **Baselines:** Popularity, CBF, Latest buy, Latest buy or sell (official starter).

## 5. Verified classical baselines (100 users, official run)

Source: `outputs/hw8/baseline/metrics.json` (protocol hash `50efd4aeb415ba6f0c66dd0793a7c2033cdc32a42b86b58eebbca3e81eaf3417`).

| Method | Users | Nonempty | nDCG@10 (all) | nDCG@10 (nonempty) |
|---|---:|---:|---:|---:|
| Popularity | 100 | 11 | 0.0286 | 0.2599 |
| CBF | 100 | 11 | 0.0201 | 0.1827 |
| Latest buy | 100 | 11 | 0.0277 | 0.2520 |
| Latest buy or sell | 100 | 11 | 0.0271 | 0.2464 |

## 6. Smoke test (verified, 3 users, not scientific evidence)

Source: `outputs/hw8/smoke/smoke_validation.json`. Original MPS/Transformers smoke (before MLX optimization) completed both arms on the same 3 deterministic users with all checks passing: same users/evidence/candidate hashes, actual inference, identical settings, explicit MPS device, pinned revision resolved, 10 unique eligible IDs after documented fallback, raw responses/logs saved.

| Arm | Wall s | New calls | Input tokens | Output tokens | Invalid | Repaired | Fallback | Peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unified (MPS smoke) | 602.18 | 3 | 15,903 | 430 | 1 | 1 | 1 | 1095.0 |
| Split (MPS smoke) | 484.71 | 3 | 15,984 | 797 | 1 | 0 | 1 | 1458.6 |

Observed smoke failure cases: one out-of-candidate ISIN (unified) and one no-JSON-array/512-token-cap response (split); both handled by documented fallback.

## 7. MLX optimization (speed-critical path)

The original Transformers+MPS path measured ~173.8 s/user-arm (~4.83 h/100 users/arm), unacceptable for deadline. A fast path was implemented:

- **Backend:** `mlx_lm.batch_generate` on Apple Silicon Metal; model converted locally from cached pinned HF snapshot to `outputs/hw8/mlx_qwen3_1p7b_4bit` (4-bit, group size 64). No weight download.
- **Settings:** greedy (`temp=0`), `do_sample=False`, Qwen3 thinking disabled (`enable_thinking=False`), KV caching via MLX generation, model loaded once and reused for both arms.
- **Budget:** `max_new_tokens=160` (safe for 10-ISIN JSON array; 128 caused truncation).
- **Batching:** batch size 3 (batch 6 gave no meaningful throughput gain).
- **Salvage:** if JSON parse fails, extract first 10 unique eligible quoted ISINs from truncated output; if still invalid, documented fallback to original candidate order. Applied identically to both arms.
- **Throughput:** benchmark ~5.9 user-arm gens/min; 6-user test ~5.75/min; full-run observed ~5.8/min. Below 10/min target but far above original ~0.35/min.

## 8. Fast-run snapshot (PRELIMINARY — split incomplete)

Source: `outputs/hw8/fast_full/unified/metrics.json`, `outputs/hw8/fast_full/unified/run_manifest.json`, `outputs/hw8/fast_full/split/progress.json`. **Unified is single-arm complete; split is incomplete. No official paired 100-user comparison is reported.**

| Arm | Done / 100 | Status | nDCG@10 all | nDCG@10 nonempty | Raw invalid | Parse repaired | Final fallback |
|---|---:|---|---:|---:|---:|---:|---:|
| Unified (MLX 4-bit) | 100/100 | complete (unpaired) | 0.0201 | 0.1827 | 36 | 0 | 36 |
| Split (MLX 4-bit) | 3/100 (at snapshot) | running | PENDING | PENDING | PENDING | PENDING | PENDING |

**Caveat:** Unified 100-user result is reported as a preliminary single-arm artifact only. It must not be interpreted as official HW8 comparison until split completes 100/100, paired users are audited, and manifests/hashes are verified. The high fallback rate (36/100) means this is fallback-inclusive, not bare LLM performance.

## 9. Manual nDCG@10 verification example (actual artifact)

User: `0E72D1BF99E858EEE1AA` (from `outputs/hw8/fast_full/unified`, incomplete run; used only for verification example).  
Ground truth future Buy ∩ eligible: `['GRS372003004']` (relevant_count=1).  
Model top-10: `['GRS054061007','GRS070003009','GRS074083007','GRS074084005','GRS079103008','GRS083003012','GRS087103008','GRS090101007','GRS090106006','GRS095001004']`.

The relevant item is not in top-10, so DCG@10=0, IDCG@10=1/log2(1+1)=1, **nDCG@10=0.0**. This matches the recorded `ndcg_at_10=0.0`.

## 10. Limitations and pending work

- Fast-run is **incomplete for comparison**: unified 100/100 is complete but split was still running at draft time. No official paired 100-user comparison is reported.
- MLX 4-bit quantization and 160-token/salvage path differ from the original bfloat16 Transformers smoke; documented as a protocol amendment for speed, applied identically to both arms.
- 3-user smoke nDCG is not scientific evidence.
- Throughput below requested 10/min; estimated ~18 min remaining at observed rate.

## 11. Artifact paths

- Protocol: `docs/research/hw8-protocol.md`
- Classical baselines: `outputs/hw8/baseline/metrics.json`
- Smoke validation: `outputs/hw8/smoke/smoke_validation.json`
- Smoke arms: `outputs/hw8/smoke/unified/`, `outputs/hw8/smoke/split/`
- MLX model: `outputs/hw8/mlx_qwen3_1p7b_4bit/`
- Fast runner: `project/far-trans-llm/fast_ranker.py`
- Fast full run (active/incomplete): `outputs/hw8/fast_full/`, log `outputs/hw8/fast_full_resume.log`, PID `outputs/hw8/fast_full_resume.pid`
- Fast benchmarks: `outputs/hw8/fast_bench/`, `outputs/hw8/fast_bench_b6/`
- This draft: `docs/reports/hw8-preliminary-report.md`

## 12. Native session export instructions

Before submission, export the native OpenCode session/transcript used for this work (do not fabricate). In OpenCode, use the session export/share command supported by your installed version (for example `/export` or the session export action in the UI/CLI) and save the exported file alongside submission artifacts. Include the export path in the final PDF/package.

## References

- FAR-Trans HW2 instructions: `project/far-trans/instructions/homework2.md`
- Starter/evaluation: `project/far-trans/starter.py`
- Ranker: `project/far-trans-llm/ranker.py`
- Runtime: `project/far-trans-llm/runtime.py`
- Data adapter: `project/far-trans-llm/data_adapter.py`
