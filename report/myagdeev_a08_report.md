# Separating Long-Term and Short-Term Purchase History in an LLM Ranker on FAR-Trans: A Controlled Prompt-Structure Study

**Student:** Ruslan Myagdeev | **Team:** Miagdeev Ruslan, Alektorov Gleb, Kambetov David | **Date:** 2026-11-24
**Assignment:** Personalized Ranker (Week 8)
**GitHub repository:** https://github.com/dkambetov-svg/llm4rec-far-trans | **Experiment commit:** `e65a8a1` (e65a8a1b80dc6693971856993ccc1b8d0d4e9e8a); read-only analysis scripts added in `82c1943`

**Abstract** — I study whether labelling a customer's purchase history as long-term versus short-term helps a large language model (LLM) rank financial assets. I regroup the same past Buy facts into `long_term_preferences` and `short_term_interest` blocks and let Qwen3-1.7B rerank, zero-shot, 50 fixed content-based (CBF) candidates for 100 FAR-Trans users. The split prompt reaches nDCG@10 = 0.0231 against 0.0201 for the unseparated prompt (Table 1), but the whole difference comes from one of the 11 users with future purchases. Invalid outputs (22–33 of 100 responses) limit what any prompt change can show.

**Index Terms** — LLM reranking, financial asset recommendation, FAR-Trans, prompt structure, nDCG

*Sources.* Output paths are relative to `project/far-trans-llm/outputs/`, which is not in git. **M/**, **S/**, **W2/** and **W5/** stand for `kaggle/hw8_mixed/`, `kaggle/hw8_split/`, `kaggle/hw8_split_window_v2/` and `kaggle/hw8_split_window/`. The read-only scripts that read these files are in `project/far-trans-llm/hw8_checks/` (`hw8_kaggle_report.py`, `hw8_window_report.py`, `hw8_window_v2_report.py`).

## 1 Introduction

**Problem.** The course LLM ranker shows a customer's past purchases as one unlabelled history. My research question is: *"Does explicitly separating long-term user preferences from short-term purchase interests improve nDCG@10 of an LLM ranker of financial assets, compared with an unseparated purchase history, given identical historical transactions, asset metadata and candidate sets on FAR-Trans?"*

**Motivation.** Investors repeat purchases, but they also follow recent interests. An LLM may use both signals better if they are explicitly labelled.

**Concrete example.** User 6B1591 bought GRF000045002 on 2021-06-02, inside the evaluation window. The user had never bought it before the cutoff (pre-cutoff Buy count 0, `hw8_prep_checks/report.json`, field `example`). In every prompt it appears only as candidate #39 of 50 (prompts logged in `M/`, `S/` and `W2/generations.jsonl`). The split prompt ranks it 9th (`S/predictions.csv`), while the mixed prompt does not rank it in the top 10 (`M/predictions.csv`).

**Contributions.**
- A split prompt format whose default `last_k` mode is verified by tests to contain exactly the same facts as the original prompt. There is a unit test on a synthetic fixture (`test_ranker.py:466`) and a byte-level regrouping check on the real data, which passes for 100/100 users (`hw8_prep_checks/report.json` and `kaggle/hw8_prep_checks/report.json`, field `last_k_regrouped_split_equals_mixed_bytes_users`).
- A controlled comparison plus one iteration that changes a single variable, with all failures and fallbacks reported.

## 2 Related Work

FAR-Trans [1] is the investment dataset of customers, transactions and assets that I use. I took its four CSV files from the official archive [2]. LLMRank [3] ranks retrieved candidates zero-shot with an LLM, conditioned on the user's interaction history. The course ranker that I extend adapts its recency prompting without in-context examples (`project/far-trans-llm/HW1_TO_HW2.md:41-42`). The ranker model is Qwen3-1.7B [4].

**Alternatives considered.**
- **(a) A 90-day time window for short-term interest.** I rejected it for the main run because it also changes which events are shown. I used it later as the IMPROVE step.
- **(b) The Encoder route.** I rejected it because the question is about how the ranker reads history, not about asset representations.

## 3 Method

**Approach.** I keep the course ranker, its candidates and its validation, and change only how the past Buy history is presented in the prompt.

**Pipeline.**
1. The starter selects the cohort and the 50 CBF candidates.
2. The prompt is built in mixed or split format.
3. Qwen3-1.7B generates a ranking (greedy decoding, thinking off).
4. The output is validated as a JSON list of ten unique candidate ISINs (`ranker.py:177-206`).
5. An invalid output falls back to the CBF order (`ranker.py:268-277`).
6. The starter computes nDCG@10.

**New flags.**
- `--history-format {mixed,split}` (`ranker.py:308`). Mixed is the default, and mixed prompts stay byte-identical to the original (`test_ranker.py:348`).
- `--short-term-mode {last_k,window}` (`ranker.py:311`). `last_k` puts the same last 15 Buy events and the same most recent Buy into the labelled blocks (`ranker.py:128-139`). `window` shows the Buy events of the last N days (`ranker.py:70-90`).
- `--short-window-days`, default 90 (`ranker.py:315`).
- A `PYTHONHASHSEED` guard: `ranker.py` exits if the seed is unset or random, and records it in `run_manifest.json` (`ranker.py:325-328`).
- `--dtype {auto,bfloat16,float16}` (`ranker.py:319`, `runtime.py:10,32-35`). I added it as a precaution and only used `auto`.

**Tools and versions.**
- Python 3.13.16 on Kaggle (`kaggle/hw8_prep_checks/candidates_seed0.json`) and 3.10.4 locally.
- torch 2.11.0+cu130 and transformers 4.57.6 (`runtime` block of `M/`, `S/` and `W2/run_manifest.json`).
- Qwen/Qwen3-1.7B [4], revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e (requested and resolved revision in the same manifests).
- Tesla T4 on Kaggle (my notebook output), loaded in `torch.bfloat16` (`runtime.dtype` in the same manifests).

**Key design decisions.**
1. `last_k` is the default, so split differs from mixed only by structure.
2. A fixed `PYTHONHASHSEED` makes all runs see the same candidate order (§5, Failure case 1).
3. Invalid outputs fall back and are kept, never dropped, so every method is scored on the same 100 users.

**Configuration** (from each `run_manifest.json`):
- 100 users, cutoff 2021-06-01 exclusive, a 30-day evaluation window (`end_exclusive` 2021-07-01), and 50 candidates.
- Context 8192 tokens and at most 512 new tokens (`max_new_tokens` in `generations.jsonl`).
- `do_sample` false, `thinking` false, `python_hash_seed` "0", and `--on-invalid fallback`.

## 4 Experiments

**Setup.** I use FAR-Trans [2] with the starter's 100 warm users and binary nDCG@10. Users without future purchases count as 0 in the all-user mean, and 11 users have future purchases. All runs used a Kaggle Tesla T4 and the protocol above. LLM nDCG values include the fallbacks. I did not compare runtime between methods: per-call generation times are logged in `generations.jsonl`, but I did not analyse them.

**Table 1. Results on the same 100 users (4 decimals).** Each fallback count is `runtime.raw_invalid_count` in that run's `run_manifest.json`, which equals the length of its `failures` list.

| Method | Users | Users with future purchases | nDCG@10 all | nDCG@10 nonempty | Failed / fallback | Source |
|---|---:|---:|---:|---:|---:|---|
| Popularity | 100 | 11 | 0.0286 | 0.2599 | n/a | M/metrics.json |
| CBF | 100 | 11 | 0.0201 | 0.1827 | n/a | M/metrics.json |
| Latest buy | 100 | 11 | 0.0277 | 0.2520 | n/a | M/metrics.json |
| Latest buy or sell | 100 | 11 | 0.0271 | 0.2464 | n/a | M/metrics.json |
| candidate_order_first10 | 100 | 11 | 0.0201 | 0.1827 | n/a | M/metrics.json |
| llmrank (mixed) | 100 | 11 | 0.0201 | 0.1827 | 22 | M/metrics.json, M/run_manifest.json |
| llmrank_split (last_k) | 100 | 11 | 0.0231 | 0.2101 | 33 | S/metrics.json, S/run_manifest.json |
| llmrank_split_window (v2) | 100 | 11 | 0.0201 | 0.1827 | 27 | W2/metrics.json, W2/run_manifest.json |
| *Footnote: llmrank_split_window, excluded: transformers 5.20.2* | *100* | *11* | *0.0201* | *0.1827* | *27* | *W5/metrics.json, W5/run_manifest.json* |

The five non-LLM rows are identical in all four `metrics.json` files (`hw8_window_v2_report.py`, section 3).

**Table 2. The 11 users with future purchases** (nDCG from `*/per_user.json`; valid or fallback from the `fallback` field of `*/final_responses.jsonl`).

| User | Mixed | Split | Window v2 | nDCG mixed | nDCG split | nDCG window v2 | nDCG CBF |
|---|---|---|---|---:|---:|---:|---:|
| 870E81 | valid | valid | valid | 0.6131 | 0.6131 | 0.6131 | 0.6131 |
| 6BAA2D | fallback | fallback | fallback | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 94A16D | fallback | fallback | fallback | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 6FA5BF | fallback | fallback | fallback | 0.1078 | 0.1078 | 0.1078 | 0.1078 |
| 6B1591 | valid | valid | valid | 0.0000 | 0.3010 | 0.0000 | 0.0000 |
| 29C059 | fallback | fallback | fallback | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0E72D1 | fallback | fallback | fallback | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| D63DFA | valid | fallback | fallback | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 5087D2 | fallback | fallback | fallback | 0.2891 | 0.2891 | 0.2891 | 0.2891 |
| 8A220D | fallback | fallback | fallback | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 5180B5 | fallback | fallback | fallback | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Comparison vs baselines.** Neither LLM variant beats the simple baselines. Popularity (0.0286) and Latest buy (0.0277) stay on top in this sample (`M/metrics.json`), which suggests that repeat and popular purchases explain most of the 11 future purchases. Split (`last_k`) is above its own control, the original CBF order (0.0231 vs 0.0201), but this gain comes from a single user, 6B1591 (Table 2), so I do not treat it as evidence that separation helps. Mixed equals the CBF order exactly: 8 of the 11 users with future purchases fell back to that order, and the 3 valid responses did not move any hit. 870E81 has its hit at rank 1 in both lists, and 6B1591 and D63DFA have no hit in either (`M/predictions.csv`). The window run returns to 0.0201 (`W2/metrics.json`).

**Verification.**
- **Unit tests.** `python -m unittest discover -s . -p 'test_*.py'` ran 43 tests, all OK, locally and on Kaggle (Kaggle: my notebook output). They include golden byte checks of the mixed prompt (`test_ranker.py:348`), the same-facts test (`test_ranker.py:466`), future-data and Sell exclusion (`test_ranker.py:78,105,509`), the hash-seed guard (`test_ranker.py:246,262`) and dtype tests (`test_runtime.py`).
- **Manual mutation check.** Making `pre_cutoff_buys` count Sells fails the starter-consistency test. Removing its eligibility filter does not fail that test, because `starter.py:179` already drops those rows; a separate test with an ineligible ISIN covers that filter.
- **Leak scan.** No event or profile date is on or after 2021-06-01 in any prompt. The only such strings are the cutoff labels `cutoff_exclusive` and, in window mode, `window_end_exclusive`. Sources: `hw8_prep_checks/report.json` and `kaggle/hw8_prep_checks/report.json` (field `dated_fields_at_or_after_cutoff_excluding_cutoff_label` is empty), and the latest window event date, 2021-05-31 (`hw8_window_report.py`, section 6).
- **Identical candidates.** 0 users have different candidates between any two runs (`hw8_window_report.py` and `hw8_window_v2_report.py`, section 2).
- **Baselines reproduce HW2** to 4 decimals (Table 1 vs `project/far-trans-llm/HW2_RESULTS.md`).
- **Data integrity.** The data SHA256 is identical locally and on Kaggle (`data_sha256` in every `run_manifest.json`).
- **Manifests.** I compared the run manifests: W2 differs from S only in the window settings, `adaptation`, `history_events` and the outcome counts (`hw8_window_v2_report.py`, section 1).
- **Hand computation.** I computed nDCG@10 for user 6B1591 by hand. One relevant item at rank 9 gives DCG = 1/log2(10) = 0.3010 and IDCG = 1, so nDCG = 0.3010, which matches `S/per_user.json`. I also checked the averages. The per-user sums, 2.0100 (mixed) and 2.3110 (split), give 0.1827 and 0.2101 when divided by 11, and 0.0201 and 0.0231 when divided by 100, as in `M/` and `S/metrics.json`.

**Deviations from HW2.**
- **GPU:** Tesla T4 instead of the RTX 5070 Ti Laptop GPU.
- **torch build:** cu130 instead of cu128 (`HW2_RESULTS.md`).
- **Candidate digest:** `candidates_digest` is e7068ca54ca9 on Kaggle (Python 3.13.16, `kaggle/hw8_prep_checks/candidates_seed0.json`) but 68a3b833a890 locally (Python 3.10.4), because of tie order. The candidate sets are identical and the order differs for 5 of 100 users. Within my Kaggle runs the candidates are identical.

## 5 Discussion

**Failure case 1: candidate order changed between processes.** My first prepare-only check (no seed set) found 6 of 100 users with a different candidate order in mixed vs split:

```
outputs/hw8_prep_checks__hashseed_unset/report.json
"b_candidates": {"users_with_different_candidate_assets": 6, "candidate_list_sizes": {"50": 100},
                 "users_where_split_order_differs_from_starter_CBF_order": 6}
"last_k_regrouped_split_equals_mixed_bytes_users": 94
```

*Root cause.* `starter.py:41-42` builds the asset vectors from a `set`, so the order of the float sums depends on `PYTHONHASHSEED`. Ties in `order()` (`starter.py:69-73`) then break differently between processes.

*Fix and verification.* I set `PYTHONHASHSEED=0` and added a guard in `ranker.py` that refuses to run without it (`ranker.py:325-328`). I ran the candidate digest script (`hw8_checks/hw8_candidate_stability.py`) in 8 processes: seed 0 twice, seeds 1, 2 and 3, and seed unset three times (session log). Both seed-0 processes printed the same `candidates_digest` (68a3b833a890), and every other process printed a different one. The candidate sets never changed, but the order of 16 users differed, and CBF nDCG@10 stayed 0.020100 in all 8 processes. After the fix, 0 users differ and the byte-level regrouping holds for 100/100 users (`hw8_prep_checks/report.json`, `kaggle/hw8_prep_checks/report.json`).

**Failure case 2: library drift on Kaggle.** After a Kaggle session restart, the window run used transformers 5.20.2 instead of 4.57.6, the tokenizer class Qwen2Tokenizer instead of Qwen2TokenizerFast, and `resolved_revision` None (`W5/run_manifest.json`). I detected it from `run_manifest.json` and from a different model-loading log. *Fix:* I re-ran the pinned `pip install` and added a version check that stops the cell. *Verification:* the pinned v2 run matches the split manifest except for the window fields. The v2 run and the 5.20.2 run give identical raw responses, top-10 lists and valid/fallback statuses for 100/100 users (`hw8_window_v2_report.py`, section 5). So the drift had no effect here, but I report only the pinned run.

**IMPROVE (one variable: the short-term block is a 90-day window instead of the last 15 events).** My hypothesis before the run: "A time window should reflect short-term interest better than a fixed last-15 events, but for active users the prompt becomes longer, so invalid outputs may increase." The result supported neither part. The window block was empty for 44 of 100 users and shorter than `last_k` for 76 (median 1.0 events vs 2.5, maximum 23 vs 15; counted in `W2/` and `S/generations.jsonl`). So it gave the model less recent information, not more. Invalid outputs fell to 27, from 33 (`W2/` and `S/run_manifest.json`). nDCG@10 returned to 0.0201 because 6B1591 lost its hit (Table 2).

**What worked.**
- I designed split so that tests prove it carries exactly the same facts as mixed. This kept the comparison about structure only.
- I recorded the seed, versions and hashes in every `run_manifest.json`. This is how I found the second failure. The first was found by the prepare-only check, which led me to record the seed.
- I checked the prompts with `--prepare-only` before using GPU time.

**What surprised me or didn't work.**
- Labelled blocks increased invalid outputs (33 vs 22) instead of making the task easier. The increase came mostly from duplicate ISINs (9 vs 2; `failures` in `S/` and `M/run_manifest.json`).
- A detail in the starter (a `set` plus the hash seed) silently changed the candidate order between processes.
- A 90-day window is empty for almost half of the warm users (44 of 100), so recent activity in this sample is sparse.
- **Limitation.** `json.dumps(..., sort_keys=True)` (`ranker.py:174`) moves the history after `candidate_assets` in split, so its position in the prompt also changes (`hw8_prep_checks/report.json`, field `top_level_key_order`). I documented this instead of fixing it.

**Next improvement.** Reduce invalid outputs before testing prompt structure again:
- On an invalid response, send one retry that includes the validation error, identically for mixed and split.
- Keep the history before the candidates in both formats.
- Then rerun the same comparison.

With 8 of the 11 users with future purchases falling back, no prompt change can be measured reliably now.

### Team Discussion and Individual Contribution

**Team discussion and decisions.** Our team (Ruslan Miagdeev, Gleb Alektorov, David Kambetov) works on the same question about separating long-term and short-term purchase history on FAR-Trans. We decided that each member implements and evaluates their own variant of the temporal separation on a personal branch of the shared repository https://github.com/dkambetov-svg/llm4rec-far-trans. Each of us uses the course starter, the same 100-user protocol and the four provided baselines. Gleb's Week 8 experiment has already been merged into main (pull request #1). The team main branch also contains a retry-on-invalid ranker from HW2, which matches my next improvement. Unresolved questions:
- Which variant to carry into the final project.
- Whether to adopt retry-on-invalid as the common ranker, so that fallbacks do not dominate the comparison.
- How to fix the candidate order across machines (`PYTHONHASHSEED`), so that our results can be compared directly.

**Individual contribution.** I proposed the controlled design in which split carries exactly the same facts as mixed (`last_k` mode). I directed the implementation of the split format, the `PYTHONHASHSEED` guard and the `--dtype` option with tests. I ran all inference runs on Kaggle myself, found and documented both failure cases, ran the window IMPROVE iteration, and computed the nDCG check by hand.

**Shared vs own work.**
- *Shared:* the team repository, the course starter and baselines, and the general research direction.
- *My own:* the code on branch `ruslan-myagdeev/hw8-temporal-history-ranker`, all runs and numbers in this report, the verification, and this report.

My teammates' results are not used in this report.

## 6 AI Usage Disclosure

- **AI tools.** I used Claude Code (Claude Opus 5.5) as the coding agent, for code, tests, checks, analysis scripts and the report draft. I ran GPU inference myself in a Kaggle notebook.
- **How AI was used:** code generation, test writing, debugging, read-only analysis scripts, checking that the reference pages load and match their titles and authors, and drafting report text.
- **What I personally verified.**
    - I ran every Kaggle cell myself and checked the printed results before continuing: the SHA256 of all 4 data files equal to local, 43 tests OK, 0 differing candidates, 11 users with future purchases, and transformers 4.57.6 before the v2 run.
    - I computed nDCG for user 6B1591 and both averages by hand (§4).
    - I compared the baseline numbers with `HW2_RESULTS.md`.
    - I opened every reference URL and confirmed that it works.
- **What I trusted without verification.**
    - The per-user tables and the failure grouping produced by the agent's read-only scripts.
    - The agent's reading of `starter.py` for the hash-seed root cause: I checked the outcome, not every line.
    - That bfloat16 on the T4 does not change the results beyond what the runs show.
- **Session log:** the attached session.json of the agent session.

## References

[1] J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An Investment Dataset for Financial Asset Recommendation," arXiv:2407.08692, 2024 (accepted at the IJCAI-2024 Workshop on Recommender Systems in Finance). [Online]. Available: https://arxiv.org/abs/2407.08692. [Accessed: 2026-11-25].

[2] University of Glasgow, "FAR-Trans: An Investment Dataset for Financial Asset Recommendation," research data collection, 2024, doi: 10.5525/gla.researchdata.1658. [Online]. Available: https://researchdata.gla.ac.uk/1658/. [Accessed: 2026-11-25].

[3] Y. Hou et al., "Large Language Models are Zero-Shot Rankers for Recommender Systems," in Proc. European Conf. on Information Retrieval (ECIR), 2024. [Online]. Available: https://arxiv.org/abs/2305.08845. [Accessed: 2026-11-25].

[4] Qwen, "Qwen3-1.7B," Hugging Face model card. [Online]. Available: https://huggingface.co/Qwen/Qwen3-1.7B. [Accessed: 2026-11-25].
