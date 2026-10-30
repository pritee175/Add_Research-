# Labeling Guide — Indian Addiction-Ad Dataset (v1)

## Scope
- Version 1 = video ads only (commercial ads + public-health campaign videos).
- Excluded from v1: static image ads, hoardings, movie scenes, news clips, ordinary user posts.
- Country: India. Product focus: [FILL IN — alcohol / tobacco-pan masala, pick one first].

## Label definitions

| Label | When to use it |
|---|---|
| promotional | The main message makes the product/brand attractive, encourages trial or purchase, or normalizes use. A small required warning does not change this. |
| preventive | The main message warns about harm, discourages use, or supports quitting. A product/brand may appear as something to avoid. |
| unclear_mixed | Substantial messages in both directions, neither dominates, or purpose is genuinely unclear. Record why in `final_reason`. |
| excluded | Not an ad/campaign, duplicate, incomplete, or too poor to judge. Fill `exclude_reason`. |

## Hard cases
- **"Drink/use responsibly" disclaimer ads:** if the ad mainly builds brand appeal → `promotional` (set `responsible_message_present = yes`). If it genuinely discourages harmful use with no meaningful brand appeal → `preventive`. If both are substantial → `unclear_mixed`, third reviewer decides. Never classify by the disclaimer phrase alone.
- **Surrogate advertising:** record `shown_product` and `implied_product` separately. Use `surrogate_verified` only when you can document a brand-name/logo link to the restricted product. Use `surrogate_possible` when plausible but unverified. A pan masala ad is not automatically a gutka ad. For preventive ads, `promotion_type` is normally `not_applicable`.
- Our surrogate label is a **content-based research proxy**, not a legal determination (ASCI's actual legal test is based on private sales-turnover data we don't have access to).

## Annotation process
1. Two annotators label each ad independently. Hide channel name/source_type where practical.
2. Third person (adjudicator) resolves disagreements; record `adjudicator_id`.
3. Keep both original annotator answers even after the final decision.
4. Compute Cohen's kappa on the pilot set from the two original labels, *before* adjudication.
5. Use `unknown` when a fact could apply but can't be determined; leave blank only when it doesn't apply.

## Fields never given to the classifier as input
`main_label`, `final_reason`, `exclude_reason`, annotator decisions, `evidence.direction` — these are the answer key, not features.

## Annotator wellbeing note
Content includes addiction-promoting material. Annotators should take breaks between sessions; flag any ad that is personally distressing to the adjudicator rather than forcing a label.
