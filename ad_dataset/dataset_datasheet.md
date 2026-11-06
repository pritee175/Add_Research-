# Dataset Datasheet — Indian Addiction-Ad Dataset (v1, living document)

Fill each section as the dataset grows. Do not leave this for the end.

## 1. Motivation
Why created: no existing public dataset labels Indian ads (or any ads, in any country) as promotional vs. preventive toward addictive-substance use. [Expand with 2-3 sentences once drafting the paper intro.]

## 2. Composition
- Product category covered: pan_masala, tobacco (cigarettes/bidi), alcohol; one mouth-freshener/elaichi entry on the pan_masala boundary (A_0003, Vimal Elaichi) kept because of its real-world regulatory relevance to the pan masala/gutka product line.
- Classes: promotional / preventive / unclear_mixed / excluded
- Count for the first 70 ads (as of 2026-10-30, single-rater draft labels - see §4 caveat; round 2 below): 70 total ads (68 with a usable video file; 2 excluded because the YouTube videos have been removed: A_0020, B_0024). 35 promotional, 31 preventive, 4 excluded, 0 unclear_mixed.
- Round 2 (2026-11-01) added 36 third-party YouTube ads (A_0038-A_0073) found by brand search. Of these 36: 1 removed from YouTube (A_0057), 6 excluded after review (A_0039 VHS rip of unverified origin, A_0040 Pakistani Gold Flake pack, A_0063 likely Pakistani, A_0070/A_0071/A_0073 explainer videos), 29 usable. Round 3 (2026-11-06) added 13 preventive ads (B_0034-B_0046) chosen from a search-based candidate list; B_0034 and B_0036 were excluded as duplicates of B_0019 and B_0003. Total 119 rows, 108 usable (11 excluded or unavailable). Category mix of the 108 usable ads: tobacco 39, alcohol 36, pan masala 30, elaichi 3. By collection target: 65 promotional-targeted, 43 preventive-targeted (preventive: tobacco 35, alcohol 8, pan masala 0), so classes are imbalanced by design of the search; final labels are still pending. No preventive pan masala ads could be found on YouTube.
- Splits (`ads.csv` `split`, made by `tools/make_splits.py`, seed 42, excluded ads skipped): train 75 / val 17 / test 16, stratified by collection target and grouped so ads from the same campaign or brand, plus the duplicate pair B_0002/B_0017, never fall in different splits. Splits do not depend on final labels. They were regenerated when rounds 2 and 3 were added.
- Languages present: Hindi, English, Hindi-English code-mixed, Marathi, Bengali, Tamil; a few marked unknown.
- Known imbalance: promotional/preventive are roughly balanced at this draft stage (35 vs. 31), but this reflects collection targeting, not a verified ground-truth distribution — see §4.

## 2a. Scale vs. target (per Indian_Advertisement_Dataset_Complete_Guide.docx §11 build order)
- Build-order target was 150-200 pilot ads before computing agreement, scaling later to 300-500 ads/class. Current count (70) is well below both — more collection is still needed before this dataset is "pilot complete" by that plan.

## 3. Collection process
- Sources used: Google Ads Transparency Center, official brand YouTube channels, NTCP official YouTube channel, Internet Archive PSA collections.
- Method: manual, individually-reviewed selection + yt-dlp download per video. No automated bulk scraping.
- Date range of collection: 2026-10-29 to 2026-10-30 (all 70 ads collected within a 2-day window — a limitation in itself; see §8).
- Consent/IRB: annotators are labeling public ad content, not acting as human subjects themselves; no personal data collected from annotators beyond ID. [Confirm/revise with your institution if one is involved.]

## 4. Preprocessing / labeling
- Two independent annotators + adjudicator (see labeling_guide.md).
- **Current status (2026-10-30): this gate has NOT been met.** All 70 ads were labeled by Claude (AI) in a single review pass; the `ann1` and `ai_draft` columns in annotators.csv both originate from that same pass and are not independent. Pilot kappa and full-set agreement cannot be meaningfully computed from the current data — doing so would measure self-consistency, not inter-annotator reliability. A genuine second, independent human pass is the top-priority open task before these labels can be treated as validated ground truth. (User decision 2026-10-30: proceeding with other build-order steps in parallel rather than blocking on this; tracked as a known limitation, see §8.)
- Pilot kappa: [FILL IN once a real second independent human pass exists]
- Full-set agreement: [FILL IN once a real second independent human pass exists]
- ASR tool: Sarvam saaras:v4 (audio only uploaded), automatic and not yet human-corrected. Screen-text tool: a vision-language model (Claude) reads deduplicated keyframes (every 2 s for ads up to 60 s, every 3 s or more for longer ones, at most 90 frames per ad); EasyOCR was tried first and dropped because it misread logos, stylised and Devanagari text. Rows are `automatic` until human-corrected. Brief subtitles between sampled frames can be missed, so use speech.csv for spoken content; row end times are approximate where frames were skipped as near-duplicates. Built by scripts/extract_keyframes.py and scripts/step3b_screen_text_vlm.py.

## 5. Uses
- Intended use: training/evaluating promotional-vs-preventive classifiers for Indian video ads; studying shortcut-learning around health warnings.
- Not intended for: making legal/regulatory determinations about surrogate advertising (that requires sales-turnover data this dataset does not have).

## 6. Distribution
- Plan: release annotation CSVs + source URLs + re-download script. Raw video files not redistributed (copyright).
- License for annotations: [FILL IN, e.g. CC BY-NC-SA 4.0 to match PHAD precedent]

## 7. Maintenance
- Maintainer: [FILL IN]
- Update plan: [FILL IN]

## 8. Known limitations
- **Round-2 ads (A_0038-A_0073) have only the `ai_ann2` label** (no `ann1`), and 26 of the 29 usable ones were labeled from title, brand and transcript without watching the video. Their screen text was read from keyframes on 2026-11-06 (automatic, not corrected). The usable round-3 ads (B_0035, B_0037-B_0046) have no AI label. A_0062 may be a Pakistani production and A_0031 is a review video, not an ad; both need a decision. Their license is third-party re-upload, uploader named in `license_or_terms_note`.
- **A second AI pass (`ai_ann2`) exists** (kappa 0.866 vs `ann1`, 5 disagreements: A_0015, B_0004, B_0016, B_0017, B_0027), but it was made from transcript, screen text and metadata without frames and is not human agreement. Human second annotation and adjudication are still pending.
- **No verified inter-annotator reliability yet.** Current labels (annotators.csv `ann1`/`ai_draft`) are both AI-origin from one review pass, not two independent humans. Cohen's kappa has not been and cannot yet be meaningfully computed. Treat current `chosen_label` values as a strong draft, not validated ground truth, until a real second human pass is added.
- `surrogate_verified` is a content-based research proxy (shared brand name/logo), not a legal determination — ASCI's actual legal test relies on private sales-turnover data this dataset does not have.
- Small sample size (70 ads) relative to the build-order target of 150-500 ads; may not generalize across brands, years, or regions.
- Entire collection happened within a 2-day window (2026-10-29 to 2026-10-30), so there is no seasonal/temporal spread (e.g. festival-season ad surges, campaign refreshes over time).
- India-specific; label definitions and hard-case rules (surrogate advertising, "drink responsibly" disclaimers) are tuned to Indian regulatory context (COTPA, ASCI guidelines) and may not transfer directly to other countries.
- `frames.csv` currently holds frames extracted for AI review purposes (regular percentage-based sampling, 4-5 per ad) rather than frames specifically curated for model training (e.g. deliberately including brief logo/warning flashes) — may need resampling before use as model input.
- `speech.csv` and `screen_text.csv` are populated automatically and still need manual correction; speech segments come from Sarvam in chunks of up to 25 s, so long ads have few, long rows.
