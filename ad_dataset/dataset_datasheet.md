# Dataset Datasheet — Indian Addiction-Ad Dataset (v1, living document)

Fill each section as the dataset grows. Do not leave this for the end.

## 1. Motivation
Why created: no existing public dataset labels Indian ads (or any ads, in any country) as promotional vs. preventive toward addictive-substance use. [Expand with 2-3 sentences once drafting the paper intro.]

## 2. Composition
- Product category covered: pan_masala, tobacco (cigarettes/bidi), alcohol; one mouth-freshener/elaichi entry on the pan_masala boundary (A_0003, Vimal Elaichi) kept because of its real-world regulatory relevance to the pan masala/gutka product line.
- Classes: promotional / preventive / unclear_mixed / excluded
- Count so far (as of 2026-10-30, single-rater draft labels — see §4 caveat): 70 total ads (68 with a usable video file; 2 failed to download and are excluded). 35 promotional, 31 preventive, 4 excluded, 0 unclear_mixed.
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
- ASR tool: OpenAI Whisper, human-corrected. OCR tool: EasyOCR, human-corrected. [Not yet run — speech.csv/screen_text.csv still empty as of 2026-10-30.]

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
- **No verified inter-annotator reliability yet.** Current labels (annotators.csv `ann1`/`ai_draft`) are both AI-origin from one review pass, not two independent humans. Cohen's kappa has not been and cannot yet be meaningfully computed. Treat current `chosen_label` values as a strong draft, not validated ground truth, until a real second human pass is added.
- `surrogate_verified` is a content-based research proxy (shared brand name/logo), not a legal determination — ASCI's actual legal test relies on private sales-turnover data this dataset does not have.
- Small sample size (70 ads) relative to the build-order target of 150-500 ads; may not generalize across brands, years, or regions.
- Entire collection happened within a 2-day window (2026-10-29 to 2026-10-30), so there is no seasonal/temporal spread (e.g. festival-season ad surges, campaign refreshes over time).
- India-specific; label definitions and hard-case rules (surrogate advertising, "drink responsibly" disclaimers) are tuned to Indian regulatory context (COTPA, ASCI guidelines) and may not transfer directly to other countries.
- `frames.csv` currently holds frames extracted for AI review purposes (regular percentage-based sampling, 4-5 per ad) rather than frames specifically curated for model training (e.g. deliberately including brief logo/warning flashes) — may need resampling before use as model input.
- `speech.csv` and `screen_text.csv` are not yet populated (ASR/OCR step not yet run).
