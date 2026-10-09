# Service-Desk AI Utilities (Projects 2A & 3A)

Two standalone Python utilities that turn raw service-desk text into **structured,
validated JSON** for downstream systems. They are model-provider–agnostic: the
same code runs against a local self-hosted model (the default), OpenAI, Anthropic
(Claude), or Google (Gemini) - chosen by configuration, not by rewriting the logic.

| Project | What it does | Status |
|---|---|---|
| **2A - Service Request Cleanup** | Turns a messy support/technical note into a clean, professional, structured record. | ✅ Built |
| **3A - Translation Quality Checker** | Checks whether an English→Spanish customer message is accurate and safe to send. | ✅ Built |

> These are **prototypes** built per the assignment. They are deliberately kept
> separate from the earlier translator / corrector / Jira tools and are **not**
> merged into them.

---

## Overview - what was built, and how it works

Both tools take raw service-desk text and return **structured, validated JSON**
through one shared, provider-agnostic layer (`src/provider_adapter.py`), so the
same code runs against a **local self-hosted model** (the default - nothing
leaves the machine, no API key) or a cloud provider (OpenAI / Anthropic / Google)
by changing a setting, never the logic. Every model response is checked against a
strict schema (`src/schemas.py`) before it is shown; a bad shape is re-asked, and
if the model still fails, the tool errors cleanly instead of emitting junk. The
whole thing is designed to run on a **CPU-only Linux box in under ~15 seconds per
request** using a small local model - no GPU required.

**Project 2A (cleanup)** is a single pass: a small local model (default
`granite3.3:2b` via [Ollama](https://ollama.com)) rewrites a messy note into a
clean, structured record and never invents facts (anything unstated becomes
`"Not stated"` or a `missing_information` entry).

**Project 3A (translation check)** is where the real engineering is - a small,
fast model is not trustworthy enough on its own for a message a customer will
read, so it runs a **layered, escalate-only safety pipeline**. Each layer may
only make the verdict *stricter* (`Send → Review First → Do Not Send`), never
looser:

1. **The model** (`granite3.3:2b`) gives a first judgement of accuracy, tone, and
   send-recommendation, as validated JSON.
2. **A deterministic safety gate** (`src/safety_gate.py`, pure Python - no model,
   ~zero latency) re-checks the exact high-risk error classes small models miss:
   changed/added numbers and currencies, unauthorised commitments (guarantees,
   refunds, discounts, legal admissions, deadlines, free-of-charge, blame-shifting
   onto the customer), and dropped/added negations that reverse meaning.
3. **An optional semantic check** (`src/nli_check.py`) uses a small open-source
   cross-lingual model - **mDeBERTa-v3 XNLI** (MIT licence, ~280M params, runs on
   CPU in under a second) - to catch pure meaning-swaps that have no trigger words
   ("your subscription is active" → "has been suspended"). It only hard-blocks
   when *both* reading directions agree it's a contradiction.

**What I did:** built both tools and their strict schemas; built the 3A safety
net (the deterministic gate and the NLI layer) and the shared escalate-only
merge; wrapped both in a local web UI (`app.py`); built the evaluation harness
and a **152-case labelled test set** and used it to validate the design. On that
set the final pipeline produced **0 dangerous messages rated "Send" and 0 false
alarms** (good messages wrongly blocked); the remaining disagreements are all
one-step-milder judgement calls on borderline messages, never the dangerous kind.
It runs on CPU under the 15-second budget, ships **83 passing tests**, and
deploys to a Linux box with a single command (`python3 deploy.py`).

---

## Project 2A - Service Request & Technical Comment Cleanup Assistant

**Input:** one raw note (an argument, a file, or piped via stdin), e.g.

```
cust says cant login been trying since morning says password reset not working
maybe email wrong needs help asap very upset
```

**Output:** machine-readable JSON first, then a human-readable summary. The JSON
fields (validated against a strict schema before anything is shown):

| Field | Meaning |
|---|---|
| `cleaned_description` | professional rewrite of the issue |
| `short_summary` | one-sentence summary |
| `issue_type` | one of: Login, Payment, Refund, Order, Loyalty, Technical, Data, Integration, Other |
| `customer_impact` | what the customer/user is experiencing |
| `technical_details` | list of systems/errors/APIs/screens mentioned |
| `missing_information` | list of info needed to resolve it |
| `suggested_next_step` | recommended next action |
| `tone_adjustment` | note on emotional/unclear wording that was neutralised |
| `ml_tags` | simple, reusable tags for future classification |
| `confidence_score` | 0–100, based on clarity/completeness of the input |

**It never invents facts** - anything not stated becomes `"Not stated"` or goes
into `missing_information`.

---

## Project 3A - Translation Quality Checker

**Input:** two texts - the original English message and its Spanish translation.
Pass them as two arguments, as `--english`/`--spanish`, or as a JSON object
(`--file` or stdin):

```json
{
  "english_original": "We are sorry your order was delayed. Please send us your order number so we can review the issue.",
  "spanish_translation": "Lamentamos que su pedido se haya retrasado. Envíenos su número de pedido para revisar el problema."
}
```

**Output:** machine-readable JSON first, then a human-readable summary. The JSON
fields (validated against a strict schema before anything is shown):

| Field | Meaning |
|---|---|
| `short_summary` | one-sentence, UI-facing verdict (e.g. "Do not send: adds an unauthorised refund promise") |
| `accuracy_rating` | `High` / `Medium` / `Low` - how faithfully the Spanish conveys the English |
| `confidence_score` | 0–100 confidence in the assessment |
| `tone_check` | `Friendly` / `Professional` / `Too Harsh` / `Too Casual` / `Unclear` |
| `risky_phrases` | wording that could confuse, offend, or create business risk |
| `missing_meaning` | meaning in the English that is absent from the Spanish |
| `added_meaning` | meaning in the Spanish that was **not** in the English (e.g. an invented refund promise) |
| `suggested_correction` | improved Spanish - only if a correction is needed, else empty |
| `back_translation` | the Spanish translated back into English, for operator review |
| `send_recommendation` | `Send` / `Review First` / `Do Not Send` |
| `explanation` | short justification of the recommendation |

It **won't make the message more binding than the original**, flags added
promises/claims, and **returns a clear error if the Spanish is blank or isn't
Spanish** (e.g. the English was pasted into the Spanish field).

> **Validating quality (beyond the unit tests):** a labelled eval set, a
> bilingual-reviewer workflow, and agreement scoring live in
> [`samples/evaluation/README.md`](samples/evaluation/README.md).

---

## Quick start

> **Deploying to a Linux box?** Skip straight to
> [Deploy on a Linux (CPU) box](#deploy-on-a-linux-cpu-box) - one command
> (`python3 deploy.py`) does the whole setup and starts the web UI.

Paths differ by OS: on **Windows** the venv Python is `.venv\Scripts\python`; on
**Linux/Mac** it's `.venv/bin/python`. Both are shown below.

```console
# 1. one-time setup (creates a virtual environment, installs core deps)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Windows
.venv/bin/pip install -r requirements.txt          # Linux / Mac

# 2. run it
#    The DEFAULT provider is "local" (a self-hosted model - see "Local model" below).
#    To try instantly with NO setup, add --provider mock to any command.

# Project 2A - clean up a messy note   (Windows shown; on Linux use .venv/bin/python)
.venv\Scripts\python cleanup.py "cust cant login password reset not working very upset" --provider mock

# Project 3A - check a translation (English first, Spanish second)
.venv\Scripts\python check_translation.py "We are sorry your order was delayed." "Lamentamos que su pedido se haya retrasado." --provider mock

# Or try them interactively (prompts you in a loop):
.venv\Scripts\python try_2a.py
.venv\Scripts\python try_3a.py
```

**Prefer a browser?** There's a local web UI with both tools on one page:

```console
.venv\Scripts\python app.py            # Windows  → open http://127.0.0.1:8765
.venv/bin/python app.py                # Linux / Mac
```

You can also just **double-click `try_2a.py` / `try_3a.py`** in File Explorer -
they auto-detect and relaunch under the project's virtual-env, so the interactive
tester opens even without a terminal. Leave the prompt blank (or Ctrl+C) to quit.

You'll get the JSON record plus a readable summary. The **mock** provider is a
placeholder for testing/demo - for real analysis, configure a provider below.

> **Heads-up on mock mode:** the offline `mock` provider can't actually reason.
> For 2A it just tidies the text; for 3A it always returns `Review First` with a
> low score. To see real cleaning / real translation judgement (and to meet the
> assignment's quality criteria), configure a real provider below.

---

## Local model (the default provider)

By default the tools call a **local, self-hosted model** through an
OpenAI-compatible server - no API key, no cloud, nothing leaves your machine. The
easiest server is [Ollama](https://ollama.com):

```console
# one-time: install Ollama, then pull a model
ollama pull granite3.3:2b        # recommended small model - fast on CPU, used for both 2A and 3A

# that's it - Ollama serves on http://127.0.0.1:11434 automatically
.venv\Scripts\python cleanup.py "cust cant login, password reset not working"
```

`granite3.3:2b` is the model we validated on CPU (the 152-case 3A results above).
`phi4-mini` and other small models also work - see
[`docs/local-llm-shortlist.md`](docs/local-llm-shortlist.md).

Settings (in `.env`):

| Setting | Default | Meaning |
|---|---|---|
| `MODEL_PROVIDER` | `local` | use the local server |
| `LOCAL_MODEL` | `phi4-mini` | fallback model if the per-tool ones below are unset (set to `granite3.3:2b` on your CPU box) |
| `CLEANUP_MODEL` | *(falls back to `LOCAL_MODEL`)* | model for 2A |
| `TRANSLATION_MODEL` | *(falls back to `LOCAL_MODEL`)* | model for 3A |
| `LOCAL_BASE_URL` | `http://127.0.0.1:11434/v1` | Ollama; vLLM `:8000/v1`, LM Studio `:1234/v1` |
| `NLI_CHECK` | `on` | 3A semantic check; set to `off` to disable it |

The local provider needs **no extra `pip install`** - it uses only the standard
library, and works with any OpenAI-compatible server (Ollama, vLLM, LM Studio,
llama.cpp, TGI). Which model to choose for your hardware is covered in
[`docs/local-llm-shortlist.md`](docs/local-llm-shortlist.md) - benchmark candidates
with the 3A eval harness. If the server isn't running you get a clear error telling
you what to start (or switch to `--provider mock`).

## Using a cloud AI provider (optional)

1. Copy `.env.example` to `.env`.
2. Set `MODEL_PROVIDER` and the matching key, then install that provider's SDK:

   | Provider | `MODEL_PROVIDER` | Key in `.env` | Install |
   |---|---|---|---|
   | OpenAI | `openai` | `OPENAI_API_KEY` | `pip install openai` |
   | Anthropic (Claude) | `anthropic` | `ANTHROPIC_API_KEY` | `pip install anthropic` |
   | Google (Gemini) | `google` | `GOOGLE_API_KEY` | `pip install google-generativeai` |

   (Or `pip install -r requirements-providers.txt` to get all three.)

3. Run the same command - or override per run with `--provider` / `--model`:

   ```console
   .venv\Scripts\python cleanup.py "..." --provider openai
   .venv\Scripts\python cleanup.py "..." --provider anthropic --model claude-sonnet-4-6
   ```

Keys are read from environment variables only - **never hard-coded**, never
committed (`.env` is git-ignored). To compare providers on the same input (the
instructor's suggestion), just change `--provider` and re-run.

> **Planning a private / self-hosted model** (to reduce reliance on paid API
> keys)? The shortlist and how it plugs into `provider_adapter.py` with ~zero
> code change is in [`docs/local-llm-shortlist.md`](docs/local-llm-shortlist.md).

---

## Deploy on a Linux (CPU) box

The whole project runs on a plain CPU-only Linux machine - no GPU. There's a
single Python command that does the entire setup and starts the web UI. (It's
`deploy.py`, kept in Python on purpose - no shell scripts.)

### One command

```console
# from the project root, on the Linux box:
python3 deploy.py
```

That runs, in order: create the virtual environment → install core packages →
install the 3A semantic-check packages → write a safe `.env` (local model, **no
cloud API key**) → check the model server and pull the model if missing →
pre-download the semantic-check model → **start the web UI** at
`http://127.0.0.1:8765`.

Useful variations:

```console
python3 deploy.py --host 0.0.0.0     # let other machines on the network reach the UI
python3 deploy.py --setup-only       # install + configure everything, but don't start the server
python3 deploy.py --no-nli           # lighter box: skip the semantic-check stack entirely
python3 deploy.py --model llama3.2:3b # use a different local model
```

> On Linux, `deploy.py` installs the official **CPU-only PyTorch build** for the
> semantic check (~200MB download) instead of the default build, which drags in
> several GB of CUDA/GPU libraries a CPU box can never use.

> `--host 0.0.0.0` exposes the UI to anyone who can reach that machine on the
> network. Use it only on a trusted network / when testers need to reach the box;
> the default `127.0.0.1` keeps it to the box itself.

### Prerequisite: the model server (Ollama)

`deploy.py` needs a local model server running. On Linux that's a one-time
install of [Ollama](https://ollama.com); the installer sets it up as a background
service that starts automatically:

```console
curl -fsSL https://ollama.com/install.sh | sh
```

If Ollama isn't installed or isn't running, `deploy.py` stops with a clear
message telling you exactly what to run, then you re-run `deploy.py`. It pulls
the model itself, so you don't need to `ollama pull` by hand. On a CPU-only box
you use the **stock** model (`granite3.3:2b`) - the GPU-forcing model variants
from local testing aren't needed, because a box with no GPU already runs on CPU.

### What to do by hand (if you'd rather not use the script)

`deploy.py` just automates these steps - you can run them yourself:

```console
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt                                    # core
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu # CPU-only PyTorch (small)
.venv/bin/pip install -r requirements-nli.txt                                # 3A semantic check (optional)
cp .env.example .env                               # then edit: MODEL_PROVIDER=local, LOCAL_MODEL=granite3.3:2b
ollama pull granite3.3:2b
.venv/bin/python app.py --host 0.0.0.0             # start the web UI
```

### Notes for a shared box

- **Never copy a `.env` that contains a real API key** onto a shared machine.
  `deploy.py` writes a key-free `.env` for you and warns if it finds keys in an
  existing one.
- The first request after startup can be a little slower while the local model
  loads into memory. Setting `OLLAMA_KEEP_ALIVE` (e.g. `export
  OLLAMA_KEEP_ALIVE=30m`) keeps it warm between requests.
- The semantic-check model is cached under `~/.cache/huggingface` after the first
  download (~1.1GB). To avoid re-downloading it on another box, copy that folder
  over, or just let `deploy.py` fetch it once.
- To turn the semantic check off at runtime without reinstalling anything, set
  `NLI_CHECK=off` in the environment or `.env`.

---

## Command-line options

**Project 2A - `cleanup.py`**

```
python cleanup.py [text] [options]

  text                 the note to clean (or use --file / stdin)
  -f, --file FILE      read the note from a text file
  -p, --provider NAME  local | mock | openai | anthropic | google   (default: from .env)
  -m, --model NAME     override the model for the provider
  --json-only          print JSON only (no human-readable summary)
  --summary-only       print ONLY the one-line summary (clean output for a UI)
  --dev-mode           enable logging of the input text (LOCAL dev only)
```

**Project 3A - `check_translation.py`**

```
python check_translation.py [english] [spanish] [options]

  english spanish      the two texts as positional args
  -e, --english TEXT   the English original
  -s, --spanish TEXT   the Spanish translation
  -f, --file FILE      JSON file: {"english_original":..., "spanish_translation":...}
  -p, --provider NAME  local | mock | openai | anthropic | google   (default: from .env)
  -m, --model NAME     override the model for the provider
  --json-only          print JSON only (no human-readable summary)
  --summary-only       print ONLY the one-line summary (clean output for a UI)
  --dev-mode           enable logging of the input text (LOCAL dev only)
```

You can also pipe a JSON object in via stdin (same shape as `--file`).

> **Privacy:** customer text is **not** logged unless you pass `--dev-mode`.

---

## Project layout

```
shrey-projects-2a3a/
  deploy.py                      one-command Linux setup + launch (creates venv, installs, starts UI)
  app.py                         local web UI - both tools on one page (stdlib http.server)
  cleanup.py                     launcher for Project 2A
  check_translation.py           launcher for Project 3A
  try_2a.py                      interactive prompt-loop demo for 2A
  try_3a.py                      interactive prompt-loop demo for 3A
  README.md  .env.example
  requirements.txt  requirements-providers.txt  requirements-nli.txt
  src/
    schemas.py                   Pydantic schemas for 2A AND 3A (the validated contract)
    provider_adapter.py          the only file that talks to model vendors (shared)
    service_request_cleanup.py   2A logic + command-line interface
    translation_quality_checker.py  3A logic + command-line interface
    safety_gate.py               3A layer 2: deterministic, escalate-only safety checks
    nli_check.py                 3A layer 3: optional cross-lingual semantic check (mDeBERTa-XNLI)
    formatters.py                human-readable output (both projects)
  tests/
    test_service_request_cleanup.py   test_translation_quality_checker.py
    test_safety_gate.py               test_nli_check.py
    test_provider_adapter.py          test_app.py
  samples/
    service_request_inputs.json     the 7 spec test cases for 2A
    translation_inputs.json         the 7 spec test cases for 3A
    generate_sample_outputs.py      regenerates the 2A outputs
    generate_translation_outputs.py regenerates the 3A outputs
    expected_outputs/               2A: one JSON per case + a "golden target" for case 1
      translation/                  3A: one JSON per case + a "golden target" for case 1
    evaluation/                     3A quality validation (beyond automated checks)
      translation_eval_set.json       152 labelled cases across the failure classes
      evaluate_translations.py        human-in-the-loop harness (CSV export + scoring)
      README.md                       validation methodology & acceptance guidance
  docs/
    local-llm-shortlist.md          open-source LLMs for private, cost-controlled hosting
```

---

## Running the tests

```console
.venv\Scripts\python -m pytest -q          # Windows
.venv/bin/python -m pytest -q              # Linux / Mac
```

Tests run fully offline - no key, no network, no model download needed (the
`mock` provider is used, and the NLI layer's wiring is tested with the real model
mocked out). They cover both projects: the spec scenarios, schema validation
(enum normalisation, score clamping, list coercion), the blank / not-Spanish
guards, the model-retry path, the 3A **safety gate** (every error class, plus the
escalate-only merge and the regression cases from the 152-set validation), the
**semantic-check** wiring (on/off switch, both-direction contradiction rule,
never-downgrades, fail-open), and the **web UI** handlers. Current suite:
**83 tests**.

> The semantic layer's real accuracy (not just its wiring) is measured by the
> eval harness on the 152-case set, not the unit tests - see
> [`samples/evaluation/README.md`](samples/evaluation/README.md).

---

## How the provider-agnostic design works

- **`provider_adapter.py`** is the *only* place that knows about a vendor's SDK.
  The business logic calls `complete(system, user, provider=...)` and gets text
  back. Swapping or adding a provider touches this one file.
- **`schemas.py`** validates every model response *before* it's displayed or
  saved. If the model returns a bad shape, the cleanup module re-asks it with the
  error (up to a retry limit), then fails cleanly rather than emitting junk.
- This is exactly how the **default `local` provider** works: one branch in
  `provider_adapter.py` calls a self-hosted OpenAI-compatible server - no change to
  the utilities or the workflow. Swapping between local, OpenAI, Anthropic, and
  Google is a config change, never a code rewrite.
