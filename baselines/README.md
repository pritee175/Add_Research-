# Text baselines (2026-11-20)

Task: promotional vs preventive (binary; the single unclear_mixed ad is dropped). 101 ads: 63 promotional, 38 preventive.
Model: TF-IDF (word 1-2 grams + char 2-5 grams) + logistic regression, class-balanced. Run `python baselines/text_baseline.py`.
Never used as input: `main_label`, `final_reason`, `search_target`, `evidence.direction`, annotator fields.

| Input | Test F1 (n=15) | Val F1 | 5x5 grouped CV macro-F1 | CV acc |
|---|---|---|---|---|
| majority class | 0.38 | 0.38 | 0.38 ± 0.04 | 0.61 |
| transcript only | 0.86 | 0.56 | 0.76 ± 0.12 | 0.78 |
| screen text only | 0.93 | 0.87 | 0.88 ± 0.09 | 0.90 |
| screen text without warning/helpline/disclaimer/credit rows | 0.80 | 0.81 | 0.84 ± 0.09 | 0.84 |
| transcript + screen text | 0.93 | 1.00 | 0.93 ± 0.07 | 0.94 |

How to read this
- Trust the grouped CV (125 folds), not the 15-ad test split: one ad changes test accuracy by 7 points.
- The labels were adjudicated mostly from this same text (AI and `ann2` reasons quote transcript and screen text), so high text scores partly reflect label circularity. A model that beats these must be judged on the hard cases below, not on overall accuracy.
- Removing end-card cues (warnings, helplines, "issued in public interest") only costs about 4 CV points, so the model is not just detecting government end cards.
- Ads wrong most often: B_0044, B_0014, B_0028, B_0004, B_0042 (preventive films with little explicit warning text) and A_0047/A_0048, A_0003, A_0026, A_0038 (promotional ads with no product text or only a disclaimer).
- No frame/visual model yet; CPU only. Next: add frame features (e.g. a pretrained image encoder) and test whether they help on the hard cases above.
