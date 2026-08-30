# Completeness Audit — Projects 2A & 3A

**Author:** Shrey · **For:** Ravi · **Date:** July 2026

A pre-integration pass to make sure both prototypes are complete and internally
consistent before they get wired into the app. Scope was **the two projects as
they stand in this repo** — I did **not** pull in or merge the earlier
text-corrector / universal-translator / Jira tools (they stay standalone per the
"keep 2A and 3A separate" instruction).

## What I checked
- Both CLIs, business logic, schemas, formatters, and the provider adapter.
- The spec field lists in the README vs. what the code actually produces.
- Sample inputs / expected outputs for internal consistency.
- Tests: coverage and that they pass offline.
- The provider-agnostic contract (keys via env only, one adapter file, strict-JSON validation).

## Findings & resolutions

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | **3A had no UI-facing summary field.** 2A exposed `short_summary`; 3A only had `send_recommendation` + `explanation`, so a UI had nothing concise to show. | Medium | Added an AI-produced `short_summary` to the 3A schema, prompt, offline stub, and formatter (mirrors 2A). |
| 2 | **No clean "summary-only" output.** Both CLIs led with the full JSON dump — the "verbose debug" Ravi flagged for the UI. | Medium | Added `--summary-only` to both CLIs: prints just the one-line summary on stdout, nothing else. |
| 3 | **Unused import.** `IssueType` was imported in `service_request_cleanup.py` but never used. | Low | Removed. |
| 4 | **Inaccurate sample notes.** Both `case_1_target.json` files claimed the neighbouring `case_1.json` "is the offline mock placeholder", but both `case_1.json` files are actually **real-provider** outputs (e.g. confidence 97, `Send`) — the mock never produces those. | Low | Reworded both `_note`s to describe the target's role without the false claim (non-destructive — sample data left intact). |
| 5 | **Generated artifact could be committed.** The new evaluation harness writes `review_sheet.csv`. | Low | Added it to `.gitignore`. |

## Gaps closed by the other tasks in this batch
- **Validation depth (3A):** added a human-in-the-loop evaluation harness and
  methodology — see [`samples/evaluation/README.md`](../samples/evaluation/README.md).
- **Model independence:** added the local-LLM shortlist —
  see [`local-llm-shortlist.md`](local-llm-shortlist.md).

## Verified healthy (no change needed)
- **Provider-agnostic design intact:** all vendor contact is isolated in
  `src/provider_adapter.py`; business logic validates every response against the
  Pydantic schemas before use; API keys come only from env / `.env`.
- **Offline-safe tests:** the full suite runs on the `mock` provider (no key, no
  network) and passes.
- **2A and 3A remain standalone** — not merged, per instruction.

## Deliberately left as-is (with rationale)
- **Sample outputs for cases 2–7 are real-provider outputs**, not the `mock`
  baseline, so they can't be reproduced without an API key. I left them because
  they're richer examples of real analysis. *If you'd prefer a key-free,
  reproducible committed baseline, regenerate with the default `mock` provider:*
  `python samples/generate_sample_outputs.py` and
  `python samples/generate_translation_outputs.py`.
- **Mock 2A `short_summary`** just echoes the first ~200 chars (the offline stub
  can't summarise). This is honest placeholder behaviour; a real provider returns
  a true one-sentence summary. Not worth special-casing.

## Recommended before integration (not done here — needs your call)
- Point the `openai` branch at a local model and run the 3A eval harness to pick
  the private model (the local-API task you're taking next).
- Decide the acceptance thresholds in the evaluation README with Ravi (safety
  misses / agreement %) before any auto-send.
