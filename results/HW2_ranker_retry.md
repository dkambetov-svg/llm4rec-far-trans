# HW2 result: retry-on-invalid for the LLM ranker (aggregate only)

Code: commit `66e76ed`. Run: Kaggle, 1x Tesla T4, torch 2.11.0+cu130, transformers 4.57.6,
Qwen/Qwen3-1.7B @ 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, bfloat16, greedy, 50 CBF candidates,
`--on-invalid fallback --retry-invalid 1`. Starter defaults: 100 warm users, cutoff 2021-06-01,
30-day horizon; 11/100 users have future purchases. Wall time 46 min 36 s.

| Method | nDCG@10 all (100) | nDCG@10 nonempty (11) |
|---|---:|---:|
| Popularity | 0.0286 | 0.2599 |
| Latest buy | 0.0277 | 0.2520 |
| Latest buy or sell | 0.0271 | 0.2464 |
| CBF | 0.0201 | 0.1827 |
| Candidate order (first 10) | 0.0201 | 0.1827 |
| LLM ranker, fallback only (`llmrank`) | 0.0201 | 0.1827 |
| LLM ranker + retry (`llmrank_retry`) | 0.0201 | 0.1827 |

- First attempt invalid: 22/100 (19 outside candidate set, 2 duplicates, 1 wrong length).
- Retry: 22 attempted, 0 recovered; 19/22 retry answers identical to the first answer.
- 68 of 73 non-candidate ISINs in invalid answers are assets from the customer's own Buy history.
- 8 of the 11 users with future purchases had an invalid first answer (CBF fallback).
- Generation time: 22.4 s per first-attempt call, 27.0 s per retry call (594 s extra in total).

User-level outputs, prompts and data are not committed. Not investment advice.
