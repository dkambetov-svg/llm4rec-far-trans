# FAR-Trans temporal history comparison

Gleb Alektorov · Week 8 · Session 2

The same 100 users and 50 candidates per user were evaluated with a single chronological history and with headings marking purchases in the last 90 days. Mean nDCG@10 rose from 0.0102 to 0.0105. One user's relevant purchase moved from rank 7 to rank 5. Among the 11 users with eligible future purchases the means were 0.0929 and 0.0959.

- [Individual report](submission/alektorov_a08_report.pdf)
- [Simulated dialogue, JSONL](submission/alektorov_a08_session.jsonl) and [JSON](submission/alektorov_a08_session.json)
- [Measured results](results/) and [independent verification](work/verification.json)

## Reproduce

Run the commands below from the team repository root. Extract the official FAR-Trans archive under data/ so that data/FAR-Trans contains the four CSV files. Use Python 3.13 with torch==2.15.0, transformers==4.57.6 and safetensors==0.8.0. Download the model before inference.

```sh
hf download HuggingFaceTB/SmolLM2-135M-Instruct --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --local-dir data/model
starter=project/far-trans
data=data/FAR-Trans
python "$starter/starter.py" run --data-dir "$data" --output-dir data/baseline
python "gleb alektorov/work/rank.py" --starter "$starter" --data "$data" --baseline data/baseline --model data/model --out data/outputs
for arm in flat split; do
  python "$starter/starter.py" run --stage ranker --data-dir "$data" --responses "data/outputs/$arm.jsonl" --output-dir "data/$arm"
done
python "gleb alektorov/work/audit.py" data
```

The local inference log contains complete prompts, candidate-to-letter mappings, all scores, time and errors. Keep it private. The two starter output directories contain the original ISINs, full protocol and per-user evaluation. Each fresh experiment should use a new output directory.
