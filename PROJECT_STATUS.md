# Project Status and Handover - Indian Addiction-Ad Dataset (v1)

Status as of 2026-11-06. Everything here was checked against the files in this repo on that date. If you only read one section, read **Section 1** and **Section 7**.

## 1. In one minute

**Goal.** Build a labeled video dataset of Indian ads for alcohol, tobacco and pan masala, so a model can learn to tell **promotional** ads (make the product attractive) from **preventive** ones (warn against use), including **surrogate advertising** (for example a "music CD" or "soda" ad that really sells whisky).

**Where we are.**
- 119 videos collected, **103 usable** (16 excluded: removed from YouTube, not Indian, not ads, amateur films or duplicates).
- Each usable video has speech transcripts. On-screen text exists for all 103 (round 1 earlier, rounds 2 and 3 on 2026-11-06).
- The 95 round 1-2 ads have an AI label; the 8 round-3 ads have none yet. **No human has labeled anything.** So the labels are drafts, not ground truth, and there is **no valid inter-annotator agreement number**.
- The data is **imbalanced**: 63 promotional-targeted vs 40 preventive-targeted. Preventive alcohol has 6 ads and preventive pan masala has **0**. Round 3 added 13 preventive ads; further YouTube searches found no more usable ones, so this gap is documented, not fixed.

**The three things that matter most next:**
1. Get real humans to label the ads (sheet is ready, rebuilt for the 103 usable ads).
2. Add AI labels for the 8 round-3 ads.
3. Do not train any model on the current labels or claim any agreement figure until step 1 is done.

## 2. The project in plain terms

- **Scope (v1).** Video ads only: commercial ads and public-health campaign videos. Excluded: static images, hoardings, movie scenes, news clips, ordinary user posts. India only. Product focus: alcohol, tobacco (cigarettes and smokeless) and pan masala.
- **Labels.**
  - `promotional` - main message makes the product or brand attractive, encourages trial or purchase, or normalizes use. A small required warning does not change this.
  - `preventive` - main message warns about harm, discourages use, or supports quitting.
  - `unclear_mixed` - substantial messages in both directions, or purpose genuinely unclear. Reason must be recorded.
  - `excluded` - not an ad or campaign, duplicate, incomplete, or too poor to judge. Reason must be recorded.
- **Hard cases** (full text in `ad_dataset/labeling_guide.md`): "drink responsibly" corporate ads, surrogate advertising (`surrogate_verified` vs `surrogate_possible`; our surrogate label is a content-based research proxy, not a legal finding), and pan masala ads that are not automatically gutka ads.
- **Intended process (from the project guide).** Two annotators label every ad independently and blind. A third person adjudicates disagreements. Cohen's kappa is computed on the two original labels before adjudication. Both original answers are always kept.

## 3. Repo map

| Path | What it is |
|---|---|
| `ad_dataset/annotations/ads.csv` | **Main table**, one row per ad (31 columns: metadata, product, brand, label fields, split). |
| `.../annotators.csv` | Every annotator's label per ad (`ann1`, `ai_draft`, `ai_ann2`). |
| `.../speech.csv` | Speech transcripts, timestamped (Sarvam ASR). |
| `.../screen_text.csv` | On-screen text read from keyframes, typed (brand, warning, helpline, tagline...). |
| `.../frames.csv`, `.../evidence.csv` | Sampled frame list and evidence notes (round 1 only). |
| `ad_dataset/annotation_sheet.html` | Blind labeling tool for human annotators (see Section 8). |
| `ad_dataset/candidates/candidates_all.csv` | Every candidate YouTube link found by search (rounds 1-3, 299 rows) with the review `decision`. Record of what was considered; downloaded ads are in `ads.csv`. |
| `ad_dataset/tools/` | `build_annotation_sheet.py`, `merge_ann2.py`, `make_splits.py`. |
| `scripts/` | Pipeline scripts for ASR, OCR and keyframes. |
| `ad_dataset/labeling_guide.md`, `dataset_datasheet.md` | Label rules, and the dataset description (a living document). |
| `research_notes/` | Background research (existing datasets, Indian regulation and surrogate ads, stance methods, venues). |
| `ad_dataset/audio/`, `ad_dataset/frames/` | 16 kHz wavs (243 MB) and sampled frames (40 MB). In git. |
| `ad_dataset/videos/` | **NOT in git** (1.7 GB, over GitHub's file limit). See Section 9. |

## 4. How the data was built (pipeline)

IDs: `A_xxxx` = commercial/promotional-targeted search, `B_xxxx` = public-health/preventive-targeted search. The prefix reflects how we searched, not the final label.

1. **Collect.** Manually found YouTube links (collectors "A" and "B"), recorded in the register, then in `ads.csv`. Round 1: 70 ads (`A_0001-A_0037`, `B_0001-B_0033`). Round 2: 36 ads (`A_0038-A_0073`) found by brand-name search. Round 3: 13 preventive ads (`B_0034-B_0046`) chosen from the preventive candidate list.
2. **Download.** `yt-dlp`, max 720p, to `ad_dataset/videos/<ad_id>.mp4`. Three videos had been removed from YouTube (A_0020, B_0024, A_0057).
3. **Audio.** `ffmpeg` to 16 kHz mono wav in `ad_dataset/audio/`.
4. **Speech to text.** Sarvam `saaras:v4` (cloud, **audio only is uploaded, never video**), in silence-aware chunks of 10-25 s, per-segment language. An API key is needed (`.secrets/speech_key.txt`, git-ignored). Script: `scripts/step2d_asr_sarvam.py`. Whisper variants exist as fallback. Output rows are marked `automatic`.
5. **Screen text.** Keyframes every 2 s (every 3 s or more for ads over 60 s), deduplicated by image difference (`scripts/extract_keyframes.py`), then read by a vision-language model (Claude) into typed rows. Output rows marked `automatic`. This replaced an earlier noisier EasyOCR pass.
6. **Labels (draft).** Claude labeled the first 70 ads in one pass (`ann1` and `ai_draft` are the same pass, not independent). A second Claude pass (`ai_ann2`) was run later on the 97 usable round 1-2 ads (before the latest exclusions).
7. **Pre-filled field.** `statutory_warning_present = yes` was set for 12 pan masala ads where a "chewing of pan masala is injurious to health" warning was read from frames. Marked as an automatic suggestion for annotators to confirm.
8. **Splits.** `make_splits.py` assigns train/val/test grouped by campaign or brand so near-copies never straddle splits, stratified by search target, fixed seed 42. Current: **train 72 / val 16 / test 15**.

## 5. Current numbers (103 usable ads)

| | Count |
|---|---|
| Total rows in `ads.csv` | 119 |
| Usable | 103 |
| Excluded | 16 (A_0020, A_0031, A_0039, A_0040, A_0057, A_0062, A_0063, A_0070, A_0071, A_0073, B_0024, B_0034, B_0036, B_0039, B_0043, B_0046) |
| By product | tobacco 38, alcohol 33, pan masala 29, elaichi 3 |
| By search target | promotional 63, preventive 40 |
| Preventive by product | tobacco 34, **alcohol 6**, pan masala 0 |
| Promotional by product | pan masala 30, alcohol 28, tobacco 4, elaichi 3 |
| Speech coverage | all usable ads except a few music-only ones |
| Screen-text coverage | all usable ads (644 rows incl. excluded ads, all `automatic`) |
| AI labels | `ai_ann2` covers the 97 round 1-2 ads only (63 promotional, 29 preventive, 3 unclear_mixed, 2 excluded). The 8 usable round-3 ads (B_0035, B_0037, B_0038, B_0040-B_0042, B_0044, B_0045) have **no AI label yet**. |
| Human labels | **0** |

Why the first nine were excluded: three videos were removed from YouTube; A_0040 is a Pakistani Gold Flake pack (Urdu script); A_0063 is likely a Pakistani ad; A_0039 is a VHS rip of unverified origin; A_0070, A_0071 and A_0073 are explainer videos, not ads. B_0034 and B_0036 are duplicates of B_0019 and B_0003 (transcript match). Rows were kept, not deleted, with `exclude_reason` filled.

## 6. What has been DONE (chronological)

1. Collected and downloaded round 1 (70 ads); extracted audio, ASR, keyframes and screen text.
2. Moved screen text from noisy OCR to VLM-read text (456 rows, 67 ads); documented the method in the datasheet.
3. Pre-filled `statutory_warning_present` for 12 ads.
4. Built the **blind human annotation sheet** and its generator.
5. Ran an AI second pass (`ai_ann2`), wrote `merge_ann2.py` (merge an annotator export and compute kappa).
6. Handled the two failed downloads; filled in the Product focus placeholder in the guide.
7. Built the grouped train/val/test splits.
8. Round 2: searched by brand, reviewed candidates, downloaded 35 ads, added metadata and speech, removed 6 doubtful ones, fixed metadata errors (for example A_0059 is Kamla Pasand pan masala, not alcohol), added `ai_ann2` labels.
9. Updated the datasheet counts and limitations.
10. Built the preventive candidate list; round 3 downloaded the 13 accepted ones (B_0034-B_0046), added speech, found 2 duplicates, rebuilt splits and the sheet.
11. Searched for more preventive alcohol and pan masala ads (111 candidates). Four were accepted, but on checking none was Indian and on-topic, so none was added.
12. Screen text for the 41 round-2 and round-3 ads (188 rows); `statutory_warning_present = yes` set for 9 pan masala ads where the warning text was read (A_0054, A_0059, A_0060, A_0064-A_0069).
13. Excluded A_0031 (a review, not an ad), A_0062 (likely Pakistani), B_0039 and B_0046 (amateur films), B_0043 (cut off). `build_annotation_sheet.py` now skips excluded ads (103 in the sheet).
14. Cleanup: merged the candidate lists into `candidates_all.csv`; removed the old register, the merged `ai_ann2` export, caches and logs.

## 7. What is REMAINING

**A. Needs a human (blocking for any credible result).**
1. **Real annotators.** The guide requires two independent human annotators and an adjudicator. Currently `ann1`, `ai_draft` and `ai_ann2` are all AI. Treat them as reference labels only.
2. **Adjudication** of disagreements. Known AI disagreements so far: A_0015, B_0004, B_0016, B_0017, B_0027.

**B. Data gaps I can do once decisions are made.**
4. **Class balance.** 63 promotional-targeted vs 40 preventive-targeted. Alcohol-preventive (6) is thin and pan masala preventive (0) does not exist. Preventive pan masala ads are almost absent on YouTube; state this in the paper. Promotional tobacco has only 4.
5. **Round 2 and 3 completeness.** Screen text is done for both (read from contact sheets of deduplicated keyframes, built with `scripts/make_contact_sheets.py`). They still have **no `frames.csv` or `evidence.csv` rows**, and the 8 usable round-3 ads have no AI label. A_0062 (likely Pakistani), A_0031 (a review, not an ad), B_0039 and B_0046 (amateur films) and B_0043 (cut off) were excluded on 2026-11-06. Four ads have no speech (A_0041, A_0047 and others), possibly music-only.
6. **Weak round-2 labels.** 26 of the 29 round-2 `ai_ann2` labels came from title, brand and transcript only, without watching the video (confidence set to medium and the reason says so). They are all "promotional", partly because we searched by commercial brand. Do not treat them as evidence.
7. **No `unclear_mixed` ground truth yet.** AI marked 3. Diageo and Pernod Ricard "responsible drinking" candidates are the best source of real mixed cases.
8. **Empty or unknown fields.** `surrogate_basis` is empty for all ads, `main_label` is filled only for the 9 excluded ads, and `celebrity_present`, `responsible_message_present`, `statutory_warning_present` are mostly `unknown`. Round-2 `promotion_type` is blank.
9. **Cleaning.** Speech and screen text are automatic and noisy (long ads have few, long speech rows). Priority: correct warning, brand and helpline rows.
10. **Register.** The original register was removed; `ads.csv` is now the only register.

**C. Dataset-level.**
11. Small test set (15 ads). Metrics will be very noisy.
12. Everything was collected over a few days (2026-10-29 to 2026-11-01), so no time spread.
13. The datasheet still has `[FILL IN]` placeholders for pilot kappa and full-set agreement.
14. **Licensing.** Videos are third-party YouTube uploads, many re-uploads of old TVCs. Plan to release annotations and links only, not the video files.

## 8. The plan, step by step

**Phase 1 - Rebalance (done for now; kept for reference).**
1. Open `ad_dataset/candidates/candidates_all.csv`. Change `keep?` to `keep it` or `reject`; decide the `maybe` rows (duplicates and language variants: take few; corporate "responsible drinking" ads: keep a handful). Target at least 15 alcohol-preventive.
2. Download the accepted ones: `yt-dlp -f "bv*[height<=720]+ba/b[height<=720]" --merge-output-format mp4 --ffmpeg-location .tmp/bin -o "ad_dataset/videos/<ad_id>.mp4" <url>`. New IDs continue as `B_0034...`. Add a row to `ads.csv` per ad (same columns as round 2; `search_target = preventive`).
3. Extract audio, run Sarvam ASR, extract keyframes, read screen text, and append (do **not** regenerate) the new rows in `speech.csv` and `screen_text.csv`, so existing corrections are not overwritten.
4. Re-run `python ad_dataset/tools/make_splits.py`. **Do this only before any model is trained.** After that, freeze the split.

**Phase 2 - Finish round-2 and round-3 data (screen text done; frames/evidence rows remain).** For the round-2 and round-3 ads, add `frames.csv`/`evidence.csv` rows, and confirm `statutory_warning_present` where a warning is visible.

**Phase 3 - Human labeling (about 1-2 hours per annotator per 100 ads).**
1. Rebuild the sheet after Phase 1: `python ad_dataset/tools/build_annotation_sheet.py`.
2. Annotator 1 and annotator 2 each open `ad_dataset/annotation_sheet.html` in Chrome or Edge, **without looking at `annotators.csv`**, label every ad (label, confidence, one-line reason) and click **Export CSV**. The sheet hides existing labels, source type, title and search target. Progress saves in the browser. It needs the `videos/` folder next to it.
3. Merge each export: `python ad_dataset/tools/merge_ann2.py <export.csv> <annotator_id>`. **Note:** the script currently computes kappa against `ann1`, which is an AI label. It needs a small change to compare two human ids before the number can be reported. Do that first.
4. Report kappa on the two original human labels, before adjudication.

**Phase 4 - Adjudication and final fields.**
1. Third person resolves every disagreement; record `adjudicator_id` and `final_label` in `annotators.csv`.
2. Copy final labels into `main_label`; fill `final_reason`, `exclude_reason`, `label_confidence`.
3. Fill `surrogate_basis` using the guide's rules (`surrogate_verified` only with a documented brand or logo link; otherwise `surrogate_possible`).
4. Fill `statutory_warning_present`, `celebrity_present`, `responsible_message_present`.

**Phase 5 - Clean and freeze.** Correct warning/brand/helpline rows in screen text and the worst speech rows. Fill the datasheet placeholders. Tag a release (annotations and links only).

**Phase 6 - Baselines.** Train simple classifiers on transcript, screen text and frames using the frozen splits, report per-class results, and be explicit about the small test set. Fields never to use as model input: `main_label`, `final_reason`, `exclude_reason`, annotator decisions, `evidence.direction`.

## 9. How to work with this repo

- **Videos are not in git.** Get them from the project owner's machine (`E:\Research_FF\ad_dataset\videos`, 1.6 GB), or re-download from the `source_url` column. Three are gone from YouTube.
- **Secrets.** Never commit `.secrets/` (Sarvam API key). It is git-ignored. Ask the owner for a key if you need to run ASR.
- **Environment.** Scripts expect a Python venv with `opencv-python`, `easyocr`, `requests`, `imageio-ffmpeg`, plus `yt-dlp` and `ffmpeg`. The venv itself is not in git. Paths in `scripts/` are hard-coded to `E:\Research_FF` and need adjusting on another machine.
- **Editing CSVs.** Use Python or a spreadsheet that keeps UTF-8; many fields are Hindi, Tamil, Bengali and Marathi. Files use CRLF line endings. Close a CSV in Excel before scripts write to it (a locked file causes write errors).
- **Do not regenerate `speech.csv` or `screen_text.csv` wholesale.** The assemble scripts rebuild them from scratch and would erase manual corrections. Append new ads only.

## 10. Risks and caveats to keep in mind

- **Label circularity.** All current labels come from the same AI family. Agreement between them (kappa 0.866 between `ann1` and `ai_ann2` on the 68 first-round ads with video) measures self-consistency, not reliability. Do not cite it.
- **Class and product skew.** Promotional ads are mostly pan masala and alcohol; preventive ads are almost all tobacco. A model could learn the product, not the stance. This is why alcohol-preventive and tobacco-promotional ads matter.
- **Small data.** 103 ads against a plan of 150-200 for the pilot and 300-500 per class later.
- **External processing.** Audio went to Sarvam, and frames were read by Claude. Mention this in any ethics or data statement.
- **Wellbeing.** Content includes addiction-promoting material. Annotators should take breaks and may flag distressing ads instead of forcing a label.

## 11. Open questions for the team

1. Who are the two human annotators and the adjudicator, and by when?
2. Accept that preventive pan masala ads are not available, or source them manually (state tobacco-control cell sites, Doordarshan archives)?
3. Do we collect more promotional tobacco ads (only 4 now) in the same round?
4. Release plan: annotations and links only, or request permission for videos?
