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

# Frame and fusion baselines (2026-11-20)

Features: frozen CLIP ViT-L/14 (`openai/clip-vit-large-patch14`, commit `32bd642`) image embeddings of the 5 sampled frames per ad (512 frames, 102 ads), mean-pooled per ad. Weights cached on the project disk (`.hf_cache/`, git-ignored). Extraction: `extract_clip_features.py`; models: `image_fusion_baseline.py`; versions in `features/clip_vitl14_frames.json`. Same task, splits and grouped 5x5 CV as above. Fusion averages the text and image probabilities.

| Model | Test F1 (n=15) | Val F1 | 5x5 grouped CV macro-F1 | CV acc |
|---|---|---|---|---|
| image only | 1.00 | 0.87 | 0.94 ± 0.05 | 0.95 |
| text only (transcript + screen) | 0.93 | 1.00 | 0.93 ± 0.07 | 0.94 |
| text + image (late fusion) | 1.00 | 0.93 | 0.96 ± 0.05 | 0.96 |

How to read this
- Treat 0.93, 0.94 and 0.96 as roughly equal: the differences are smaller than the fold-to-fold spread (SD 0.05-0.07) on 101 ads. The test split has 15 ads, so its 1.00 is not a result to quote.
- The two modalities make different mistakes (text misses B_0044, B_0014, B_0028, B_0042; image misses B_0021, A_0017, B_0035, A_0005), which is why fusion helps a little. After fusion only B_0021, B_0044, B_0014, B_0004 are often wrong.
- Image features can pick up production style and end-card layout (glossy TV ads vs government PSAs), not only persuasive content. These baselines show the classes are separable; they do not show that a model understands promotion vs prevention. A harder check is to test on surrogate and "responsible drinking" ads only, where style and message disagree (too few ads for a reliable number so far).
- Labels were adjudicated mostly from the same text (see above), and the second annotator's agreement with the AI is not independent human agreement. Report these numbers as baselines on provisional labels.

# Style-versus-message check (2026-11-20)

Question: do the baselines understand the message, or the style of the source? `hard_subsets.py` scores out-of-fold accuracy (5x5 grouped CV, each ad's share of repeats classified correctly; bootstrap 95% CI over ads) on subsets defined from metadata that is never a model input. Run `cd baselines && python hard_subsets.py`.

| Subset | n | text | image | text+image |
|---|---|---|---|---|
| surrogate promotional (soda, water, music CDs, elaichi under restricted brands) | 22 | 0.99 [0.97,1.00] | 0.94 [0.83,1.00] | 1.00 |
| direct promotional (control) | 41 | 0.98 | 0.99 | 1.00 |
| preventive, non-institutional source (film-style, `source_type` other/brand) | 10 | **0.66 [0.42,0.86]** | 0.82 [0.60,1.00] | 0.76 [0.50,1.00] |
| preventive NGO/government PSAs (control) | 28 | 0.95 | 0.95 | 0.97 |
| all | 101 | 0.94 | 0.95 | 0.97 |

Findings
- Surrogate ads are not hard here. All three models classify them as promotional almost perfectly. The brand identity is the likely cue (brand names, "men will be men", logos), although CV is grouped by brand so the same brand never sits on both sides of a fold. Because every surrogate ad is promotional, this is a statement about the label, not about legal surrogate status.
- The weak spot is the preventive message delivered in a non-institutional style: short films and amateur-looking pieces. Accuracy drops from about 0.95 on NGO/government PSAs to 0.66-0.82 (text weakest). Ads mostly wrong: B_0028 and B_0044 (text), B_0004, B_0021 and A_0017 (image), B_0021 and B_0044 (fused).
- Limits: n=10 for the weak subset and its interval overlaps the control; treat this as a pointer, not a measured effect. All 10 preventive non-institutional ads are among the least typical of the preventive class, and no promotional ad has an institutional-looking style, so style and class are still confounded.
- To separate style from message the dataset needs promotional ads in an institutional/PSA style and preventive ads in a commercial style; neither exists yet.

# Final model and demo (2026-11-22)

**The model.** `fused_model.py`: TF-IDF + logistic regression on transcript + screen text, logistic regression on frozen CLIP ViT-L/14 image embeddings (5 frames/ad, mean-pooled), probabilities averaged. Either modality works alone. Saved in `models/` (`model_trainval.joblib` trained on train+val for honest test scores, `model_all.joblib` trained on all 101 binary ads for new videos). Train and score with `python baselines/final_model.py`.

**Held-out test split (15 ads, trained on train+val only):** fused 15/15 correct (95% Wilson interval 0.80-1.00), image only 15/15, text only 14/15. Per-ad probabilities: `results_final_test.csv`. This agrees with the grouped CV (macro-F1 0.96) but 15 ads is too few to report 1.00 as an accuracy; **quote the cross-validated numbers as the main result** and the test split as a sanity check.

**Informal out-of-sample check** (`external_check.py`): 15 round-4 "keep" candidates (6 drink-driving films, 8 old Indian cigarette TV ads, 1 Royal Stag short film) downloaded to `.tmp/` and never used for training or added to the dataset. Image-only predictions vs my expected labels (from titles and descriptions, NOT human-annotated): 12 videos were scored before the machine ran out of memory; **11 of 12 agreed** (5/6 drink-driving films, 6/6 cigarette ads). The one miss: "Chennai Traffic Police Advises (Do Not Drink and Drive)", predicted promotional at P=0.70. Three videos (Wills 96 World Cup, an old Doordarshan cigarette ad, the Royal Stag short film) were not scored. Treat this as a sanity check, not a result: n is tiny and the labels are unverified.

**Demo** (`demo.py`):
```
python baselines/demo.py --test-set                  # the 15 held-out ads: true vs predicted vs probability
python baselines/demo.py --ad_id B_0028              # one ad with explanation (warns if it was a training ad)
python baselines/demo.py --video my_ad.mp4           # new video, image-only (needs ~2 GB free RAM)
python baselines/demo.py --video my_ad.mp4 --transcript "..." --screen "..."   # adds the text model
```
It prints the label, P(promotional), each modality's vote, words pushing each way, and the 3 most similar dataset ads. `clip_embed.py` loads only the CLIP image tower in float16 (cosine >= 0.99997 vs the stored float32 features), because the full model did not fit in memory on a 7.5 GB laptop; the `--video` mode still needs roughly 2 GB free and can fail with a "paging file too small" error if other apps are using memory (close Chrome/VS Code windows and retry). `--test-set` and `--ad_id` do not load CLIP and always work.

**What the model is not.** It is not validated on videos from outside the collection, not a legal classifier of surrogate advertising, and the labels it learned are single-annotator labels with an AI cross-check. Style and class are confounded (see above), so it may be partly recognising PSA-vs-commercial style.

# Web demo (2026-11-22)

`python baselines/web_demo.py` then open http://127.0.0.1:8765 (standard library only, works offline, nothing leaves the computer). One page that shows: the pipeline, the data split and the checks that the test set is separate, cross-validated results, the subset analysis, a live demo on the 15 held-out test ads (watch the video, press Predict, then Reveal the human label), an optional upload for a new video, and the limitations. Files: `web_demo.py` (server), `web/index.html` (page).

The server is stdlib-only (about 30 MB) and reads precomputed `web/demo_data.json` (rebuild with `python baselines/build_demo_data.py` after retraining). The upload box runs `predict_video.py` in a separate process, so a memory failure there cannot take the page down. **If anything fails with a memory / "paging file too small" error:** the machine's C: drive was full (0 GB free), which stops Windows from growing its paging file; free space on C: or add a paging file on E: (System Properties > Advanced > Performance > Advanced > Virtual memory).

Verified: all API endpoints, video streaming with seeking, and that the page's JavaScript parses. It has not been checked visually in every browser. The upload box loads CLIP (about 2 GB free RAM); the held-out test demo needs almost nothing extra. Run it from the repo root; stop with Ctrl+C.

Split facts shown in the demo (binary task, 101 ads): train 70 / val 16 / test 15 (about 70/15/15), stratified by collection target and grouped by campaign/brand. Checked 2026-11-22: 0 shared groups, 0 shared YouTube videos, 0 identical transcripts between test and train+val; nearest training ad for any test ad has CLIP cosine at most 0.89 (median 0.83), below the 0.92 90th percentile seen between training ads. The test set uses different campaigns and brands but the same collection, era and sources.
