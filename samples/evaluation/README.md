# Project 3A — Translation Quality Validation (beyond automated checks)

**Purpose.** Passing a handful of hand-picked sentences does **not** prove the
translation checker is reliable enough to trust on real customer messages. This
folder is the validation process Ravi asked for: a repeatable way to measure the
checker against a **labelled eval set** *and* against a **human bilingual
reviewer**, so we can state — with evidence — how far the tool can be trusted and
where a person must stay in the loop.

This validates the **judgement quality** of Project 3A. It is separate from the
unit tests in `tests/`, which only prove the *plumbing* works offline (schema,
guards, retries).

---

## Why automated-only isn't enough

The checker's own `confidence_score` is the model grading its own homework. The
dangerous failures are exactly the ones a quick automated pass hides:

| Failure class | Example | Why it matters |
|---|---|---|
| **Negation drop** | "has **not** been charged" → "has been charged" | Tells the customer the opposite; looks fluent. |
| **Wrong number** | `$45` → `$54`, order `#123` → `#132` | Factually wrong, easy to miss. |
| **Added commitment** | invents a refund / deadline / legal admission | Creates real business & legal risk. |
| **Subtle tone shift** | polite request → blunt command | Damages brand, hard to catch in a spot check. |

These are why we score against **safety misses** (an expected *Do Not Send* the
model would have let through), not just overall accuracy.

---

## The two-stage process

### Stage 1 — automated pass (developer)
```console
:: from the project root, with a real or local provider configured in .env
python samples/evaluation/evaluate_translations.py --run --provider openai
```
This runs all cases in `translation_eval_set.json`, prints overall agreement plus
**safety misses** and **false alarms**, and writes `review_sheet.csv`.

> Running it on the `mock` provider works but is meaningless (mock can't judge) —
> it's only for a dry run of the mechanics.

### Stage 2 — human review (bilingual reviewer)
1. Open `review_sheet.csv` (UTF-8; opens cleanly in Excel / Google Sheets).
2. For each row, **without looking at the `model_*` columns first**, the reviewer
   fills in:
   - `human_recommendation` — one of `Send`, `Review First`, `Do Not Send`
   - `reviewer_notes` — a short reason (required when they disagree with the model)
3. Score the completed sheet:
```console
python samples/evaluation/evaluate_translations.py --score samples/evaluation/review_sheet.csv
```
This reports **human-vs-model** and **human-vs-expected** agreement and lists
every case where the human overruled the model.

---

## Reviewer rubric (what each verdict means)

- **Send** — meaning fully preserved, tone appropriate for support, nothing added
  or dropped. Safe to send as-is.
- **Review First** — essentially correct but needs a human look: tone too
  harsh/casual, a politeness or minor detail lost, an untranslated word. A tone
  problem *alone* is Review First, not Do Not Send.
- **Do Not Send** — meaning is wrong, reversed, or contradictory, **or** an
  unauthorised promise / refund / guarantee / legal claim / deadline was added
  that is not in the English.

## Who should review

- A **native or professional bilingual** reviewer — ideally **not** the developer
  who built the tool (avoids confirmation bias).
- For production sign-off, a second reviewer on any case the first flags
  `Do Not Send`, and record inter-reviewer disagreements.

## Acceptance guidance (suggested, tune with Ravi)

Before trusting a given provider/model for **customer-facing sends**:

- **Zero safety misses** on a full review round (non-negotiable — a single
  "Do Not Send → Send" is a shipped incident).
- **≥ 90% human-vs-model agreement** across the set.
- False alarms low enough that operators don't start ignoring the tool.

If any threshold fails, keep the tool in **assist mode** (it advises; a human
always confirms) rather than auto-send, and expand the eval set with the failing
cases.

## Keep the set growing

`translation_eval_set.json` starts at ~22 cases spanning the failure classes
above. Every real miss found in production should be added as a new case so the
harness gets stricter over time. It's plain JSON — add objects with
`id`, `category`, `english_original`, `spanish_translation`,
`expected_recommendation`, and `notes`.

---

## Relationship to model selection

This harness is also our **provider/model benchmark**. When we evaluate a
self-hosted local model (see [`docs/local-llm-shortlist.md`](../../docs/local-llm-shortlist.md)),
run `--run --provider <candidate>` against this exact set and compare agreement,
safety misses, and latency to the cloud providers. Same set, same scoring — an
apples-to-apples read on whether a private model is good enough to adopt.
