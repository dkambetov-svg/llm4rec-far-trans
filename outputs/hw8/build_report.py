"""Build/refresh HW8 submission PDF from latest available artifacts. No model inference."""
from __future__ import annotations
import json, datetime
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, ListFlowable, ListItem
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

ROOT = Path('/Users/davidkk/llm4rec-hw2/RecSys-LLMs')
OUT = ROOT / 'outputs/hw8/hw8-report.pdf'

def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None

def fmt(x, nd=4):
    return 'PENDING' if x is None else f'{x:.{nd}f}'

def get_snapshot():
    baseline = load_json(ROOT/'outputs/hw8/baseline/metrics.json') or {}
    smoke = load_json(ROOT/'outputs/hw8/smoke/smoke_validation.json') or {}
    uni_m = load_json(ROOT/'outputs/hw8/fast_full/unified/metrics.json')
    uni_r = load_json(ROOT/'outputs/hw8/fast_full/unified/run_manifest.json')
    uni_p = load_json(ROOT/'outputs/hw8/fast_full/unified/progress.json')
    spl_m = load_json(ROOT/'outputs/hw8/fast_full/split/metrics.json')
    spl_r = load_json(ROOT/'outputs/hw8/fast_full/split/run_manifest.json')
    spl_p = load_json(ROOT/'outputs/hw8/fast_full/split/progress.json')
    return dict(baseline=baseline, smoke=smoke, uni_m=uni_m, uni_r=uni_r, uni_p=uni_p,
                spl_m=spl_m, spl_r=spl_r, spl_p=spl_p)

def arm_row(name, m, r, p):
    if not m or not r:
        done = (p or {}).get('done', 0)
        return [name, f'{done}/100', 'running/pending', 'PENDING', 'PENDING', 'PENDING', 'PENDING', 'PENDING']
    rt = r.get('runtime', {})
    ll = m.get('llmrank', {})
    done = (p or {}).get('done', ll.get('users', 0))
    total = (p or {}).get('total', 100)
    status = 'complete (unpaired)' if done >= total else f'running {done}/{total}'
    return [name, f'{done}/{total}', status,
            fmt(ll.get('ndcg_at_10')), fmt(ll.get('ndcg_at_10_nonempty')),
            str(rt.get('raw_invalid_count','?')), str(rt.get('parse_repair_count','?')),
            str(rt.get('final_invalid_after_retry','?'))]

def build():
    snap = get_snapshot()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyJ', parent=styles['Normal'], fontSize=8.5, leading=10.2, alignment=TA_JUSTIFY, spaceAfter=3))
    styles.add(ParagraphStyle(name='BodyJSmall', parent=styles['Normal'], fontSize=7.5, leading=9, alignment=TA_JUSTIFY, spaceAfter=2))
    styles.add(ParagraphStyle(name='Section', parent=styles['Heading2'], fontSize=9.5, leading=11, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor('#111111')))
    styles.add(ParagraphStyle(name='Title2', parent=styles['Title'], fontSize=13, leading=15, alignment=TA_CENTER, spaceAfter=6))
    styles.add(ParagraphStyle(name='Meta', parent=styles['Normal'], fontSize=8, leading=9.5, alignment=TA_CENTER, textColor=colors.HexColor('#333333'), spaceAfter=6))
    styles.add(ParagraphStyle(name='Cell', parent=styles['Normal'], fontSize=7, leading=8.2))
    styles.add(ParagraphStyle(name='CellH', parent=styles['Normal'], fontSize=7, leading=8.2, textColor=colors.white))
    styles.add(ParagraphStyle(name='Warn', parent=styles['Normal'], fontSize=7.5, leading=9, textColor=colors.HexColor('#8B0000'), spaceAfter=3))

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.45*inch, rightMargin=0.45*inch,
                            topMargin=0.42*inch, bottomMargin=0.42*inch)
    story = []
    now = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    story.append(Paragraph('HW8 Preliminary Report: Temporal History Separation in LLMRank (FAR-Trans)', styles['Title2']))
    story.append(Paragraph(f'<b>PRELIMINARY / INCOMPLETE FOR PAIRED COMPARISON.</b> Generated {now}. Background inference may still be running; do not cite as official 100-user paired result until split completes and audit passes.', styles['Warn']))

    story.append(Paragraph('1. Research Question and Hypotheses', styles['Section']))
    story.append(Paragraph('<b>RQ:</b> Does explicitly separating long-term user preferences from short-term purchasing interests improve nDCG@10 in LLM-based financial asset ranking compared with a unified purchase history, while keeping historical transactions, asset metadata, and candidate sets identical? <b>H0:</b> paired mean nDCG@10 difference (Split−Unified)=0. <b>H1:</b> Split improves paired mean nDCG@10. Null/negative result is acceptable.', styles['BodyJ']))

    story.append(Paragraph('2. Controlled Experimental Protocol', styles['Section']))
    story.append(Paragraph('Dataset: FAR-Trans (transactions, asset_information, customer_information, close_prices), local path <font name="Courier" size="7">/Users/davidkk/llm4rec-hw2/data/FAR-Trans</font>. Cohort: 100 warm users, deterministic <font name="Courier" size="7">starter.select_users(...,seed=42)</font>. Cutoff 2021-06-01 exclusive; 30-day future Buy horizon. Candidates: CBF top 50, identical IDs/order. History: pre-cutoff Buy rows only; last 15 dated events truncated before serialization. Boundary: 90 days, long-term &lt; 2021-03-03, short-term &gt;= 2021-03-03 and &lt; cutoff. Unified: one chronological section. Split: same canonical records partitioned into long/short sections, no added evidence. Model: Qwen/Qwen3-1.7B, rev <font name="Courier" size="7">70d244cc86ccca08cf5af4e1e306ecf908b1ad5e</font>. Decoding: greedy, do_sample=False, thinking disabled, context 8192. Invalid: parse/repair allowed; invalid after repair receives documented fallback to original candidate order; no user dropped. Metrics: unchanged <font name="Courier" size="7">starter.ndcg10/aggregate</font>; primary all-user mean, secondary nonempty-truth mean.', styles['BodyJSmall']))

    story.append(Paragraph('3. Verified Classical Baselines (100 users)', styles['Section']))
    b = snap['baseline'].get('methods', {})
    base_rows = [['Method','Users','Nonempty','nDCG@10 (all)','nDCG@10 (nonempty)']]
    for k,label in [('popularity','Popularity'),('cbf','CBF'),('latest_buy','Latest buy'),('latest_buy_or_sell','Latest buy or sell')]:
        m=b.get(k,{})
        base_rows.append([label,str(m.get('users','')),str(m.get('nonempty_truth_users','')),fmt(m.get('ndcg_at_10')),fmt(m.get('ndcg_at_10_nonempty'))])
    t=Table([[Paragraph(c,styles['CellH']) for c in base_rows[0]]]+[[Paragraph(c,styles['Cell']) for c in r] for r in base_rows[1:]], colWidths=[1.5*inch,0.45*inch,0.55*inch,0.9*inch,1.0*inch])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#333333')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f2f2')]),('GRID',(0,0),(-1,-1),0.25,colors.grey),('PADDING',(0,0),(-1,-1),2)]))
    story.append(t)
    story.append(Paragraph('<font size="7">Source: outputs/hw8/baseline/metrics.json (protocol hash 50efd4ae...a3e81eaf3417).</font>', styles['BodyJSmall']))

    story.append(Paragraph('4. Available LLM Results (complete vs incomplete)', styles['Section']))
    ll_rows = [['Arm','Done','Status','nDCG@10 all','nDCG@10 nonempty','Raw invalid','Parse repaired','Final fallback']]
    ll_rows.append(arm_row('Unified MLX 4-bit', snap['uni_m'], snap['uni_r'], snap['uni_p']))
    ll_rows.append(arm_row('Split MLX 4-bit', snap['spl_m'], snap['spl_r'], snap['spl_p']))
    t2=Table([[Paragraph(c,styles['CellH']) for c in ll_rows[0]]]+[[Paragraph(c,styles['Cell']) for c in r] for r in ll_rows[1:]], colWidths=[1.05*inch,0.45*inch,0.95*inch,0.7*inch,0.8*inch,0.55*inch,0.55*inch,0.55*inch])
    t2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#333333')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f2f2')]),('GRID',(0,0),(-1,-1),0.25,colors.grey),('PADDING',(0,0),(-1,-1),2)]))
    story.append(t2)
    story.append(Paragraph('<b>Interpretation:</b> Unified 100/100 is single-arm complete but fallback-inclusive (36/100 fallback) and <b>not</b> a paired official comparison. Split was incomplete at generation time; no Delta_all/Delta_nonempty is reported. Any completed paired subset must be labeled with exact sample size and selection rule.', styles['Warn']))

    story.append(Paragraph('5. Smoke Test, Failure Analysis, and Limitations', styles['Section']))
    story.append(Paragraph('The initial 3-user MPS/Transformers smoke (outputs/hw8/smoke/smoke_validation.json) passed all checks: same users/evidence/candidate hashes, actual inference, identical settings, pinned revision resolved, explicit MPS, 10 unique eligible IDs after documented fallback, artifacts/logs saved. Observed failures: out-of-candidate ISIN (unified) and no-JSON-array/512-token-cap response (split), both handled by documented fallback. Fast MLX run shows high fallback rate (36/100 unified), indicating 4-bit/160-token generation often violated strict format; outputs remain fallback-inclusive. Limitations: split incomplete at draft time; no official paired 100-user result; 4-bit quantization and salvage change generation behavior; 3-user nDCG not scientific evidence; throughput ~5.8 user-arm/min below 10/min target.', styles['BodyJSmall']))

    story.append(Paragraph('6. MLX Inference Protocol Amendment', styles['Section']))
    story.append(Paragraph('Speed amendment (applied identically to both arms, documented before full fast run): converted cached pinned HF snapshot to local MLX 4-bit model (<font name="Courier" size="7">outputs/hw8/mlx_qwen3_1p7b_4bit</font>, bits=4, group=64), no weight download. Backend <font name="Courier" size="7">mlx_lm.batch_generate</font>, Apple Metal; model loaded once and reused. Greedy temp=0, do_sample=False, Qwen3 thinking disabled (<font name="Courier" size="7">enable_thinking=False</font>), KV cache enabled by backend, batch size 3, max_new_tokens=160. Truncated-array salvage extracts first 10 unique eligible quoted ISINs before documented fallback. Original MPS smoke measured ~173.8 s/user-arm; MLX measured ~5.8 user-arm/min.', styles['BodyJSmall']))

    story.append(Paragraph('7. Manual Verification Example (requires my personal confirmation)', styles['Section']))
    story.append(Paragraph('<b>I have not personally verified this example; user confirmation required.</b> User <font name="Courier" size="7">0E72D1BF99E858EEE1AA</font> (fast_full/unified artifact). Ground truth future Buy ∩ eligible: [GRS372003004] (relevant_count=1). Model top-10: [GRS054061007, GRS070003009, GRS074083007, GRS074084005, GRS079103008, GRS083003012, GRS087103008, GRS090101007, GRS090106006, GRS095001004]. Relevant item not in top-10: DCG@10=0, IDCG@10=1/log2(2)=1, nDCG@10=0.0, matching recorded artifact value.', styles['BodyJSmall']))

    story.append(Paragraph('8. Reproducibility and Artifacts', styles['Section']))
    story.append(Paragraph('Protocol: <font name="Courier" size="7">docs/research/hw8-protocol.md</font>. Baselines: <font name="Courier" size="7">outputs/hw8/baseline/metrics.json</font>. Smoke: <font name="Courier" size="7">outputs/hw8/smoke/smoke_validation.json</font>. Fast runner: <font name="Courier" size="7">project/far-trans-llm/fast_ranker.py</font>. Fast run: <font name="Courier" size="7">outputs/hw8/fast_full/</font> (resume log/pid/progress/manifests). Report refresh: <font name="Courier" size="7">cd /Users/davidkk/llm4rec-hw2/RecSys-LLMs &amp;&amp; /Users/davidkk/llm4rec-hw2/.venv/bin/python outputs/hw8/build_report.py</font>. Native OpenCode session export: run <font name="Courier" size="7">/export</font> in OpenCode and save the exported session file with submission artifacts.', styles['BodyJSmall']))

    story.append(Paragraph('<i>References (excluded from 5-page body limit):</i> FAR-Trans HW2 instructions; starter.py; ranker.py; runtime.py; data_adapter.py; HW8 protocol v1.0; smoke/fast manifests.', styles['BodyJSmall']))
    doc.build(story)

if __name__ == '__main__':
    build()
    print(OUT)
