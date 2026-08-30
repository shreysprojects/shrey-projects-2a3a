# Local / Self-Hosted LLM Shortlist

**Author:** Shrey  ·  **For:** Ravi  ·  **Date:** July 2026

## 1. Goal & evaluation criteria

We currently depend on paid cloud API keys (OpenAI / Anthropic / Google). The goal of this doc is to shortlist **open-weight LLMs we can self-host privately** so we can:

- **Control cost** — no per-token API bill; pay for hardware/compute we already control.
- **Keep data private** — prompts and outputs stay on our infrastructure.
- **Stay provider-agnostic** — swap models without rewriting business logic.

Ravi named **Llama** and **GLM 5.2** as candidates and raised a **geopolitical concern about GLM being from China**. He wants something **fast, scalable, and extensible across multiple projects without rewriting business logic**. This doc addresses all three.

I evaluated candidates against:

| Criterion | Why it matters for us |
|---|---|
| **License / openness** | Must permit commercial self-hosting with no MAU trap or field-of-use limits. |
| **Quality / reasoning tier** | Our tasks are moderate, not frontier — but output must be correct. |
| **Speed / latency** | Ravi wants "fast"; our jobs are short and high-volume. |
| **Hardware / VRAM footprint** | Determines whether one GPU box suffices or we need a cluster. |
| **JSON / structured-output reliability** | **Critical** — our Pydantic schemas require STRICT JSON (see §7). |
| **Ease of self-hosting** | Prototype speed vs. production ops. |
| **Extensibility across projects** | One serving endpoint reused by all projects, zero per-project rewrites. |
| **Data privacy / governance** | Provenance, supply-chain, and policy optics (see §4). |

---

## 2. How a local model slots into THIS codebase (the key insight)

Our repo already isolates all vendor contact in **one file**: `src/provider_adapter.py`. Business logic (2A service-request cleanup, 3A translation quality checker) never talks to a vendor directly — it calls `complete(system, user, provider=..., model=...)` and validates the returned text against strict Pydantic JSON schemas. The provider is chosen by the `MODEL_PROVIDER` env var.

The `openai` branch (lines 84–100) does this:

```python
client = OpenAI()                          # honors OPENAI_BASE_URL / base_url
resp = client.chat.completions.create(
    model=model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
    temperature=temperature,
    response_format={"type": "json_object"},   # <-- we require valid JSON
    messages=[{"role": "system", ...}, {"role": "user", ...}],
)
```

**Every mainstream local serving stack (Ollama, vLLM, llama.cpp server, LM Studio, TGI) exposes an OpenAI-compatible `/v1/chat/completions` endpoint.** That means we can point the **existing `openai` branch at a local server** and add a private model with **essentially zero business-logic change**:

```bash
# .env — point the SAME openai code path at a local box
MODEL_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama; vLLM=:8000/v1, etc.
OPENAI_API_KEY=local-not-checked            # SDK requires a value; local servers ignore it
OPENAI_MODEL=qwen3.6:27b                    # whatever model the server has loaded
```

The `openai` SDK's `OpenAI()` client reads `OPENAI_BASE_URL` from the environment, so **no code edit is strictly required** to prototype — just env vars. (If we want it explicit and self-documenting, a one-line `base_url=os.getenv("OPENAI_BASE_URL")` on the client is the only change we'd ever make.)

**Why structured-output support matters here:** our schemas reject anything that isn't valid JSON. The adapter already sends `response_format={"type": "json_object"}`. A local server must honor that (or an equivalent guided-decoding / grammar mode) or the schema validation will fail. This single requirement drives the serving-stack choice in §5 — some stacks handle `response_format` cleanly, one historically does not.

---

## 3. Comparison of leading open-weight model families (mid-2026)

Versions and licenses verified via web search (see Sources). Quality tiers are qualitative — I have **not** fabricated benchmark numbers; where I cite a benchmark it's from a linked source.

| Family (latest) | Approx sizes | License | Quality tier | Strengths | Weaknesses / caveats | Provenance |
|---|---|---|---|---|---|---|
| **Meta Llama 4** (Scout, Maverick; Behemoth larger) | MoE; Scout & Maverick are the shipped open tiers | **Llama 4 Community License** — commercial OK **under 700M MAU**; "Built with Llama" attribution required | Strong general | Huge ecosystem, tooling, fine-tunes everywhere; natively multimodal MoE | Not a pure OSI license (MAU cap + attribution); Meta's newest *frontier* model (Muse Spark, Apr 2026) is **closed/API-only** | USA (Meta) |
| **Alibaba Qwen 3.5 / 3.6** | Dense 0.6B–32B, plus 27B & MoE (e.g. 35B-A3B, 235B-A22B); 3.6-27B is newest dense | **Apache 2.0** | Strong, top open tier | Clean permissive license, wide size ladder, excellent multilingual (100+ langs), strong JSON/instruction following, active releases | Chinese provenance (same policy optics as GLM — see §4); many SKUs to track | China (Alibaba) |
| **Zhipu / Z.ai GLM 5.2** | ~744–753B MoE, ~40B active; 1M context | **MIT** | Frontier-class (esp. coding/agentic) | Fully open MIT weights, top-tier coding (2nd on Code Arena behind Claude Opus 4.8 per VentureBeat), 1M context | **Enormous** — full precision ~1.5 TB GPU memory; impractical to self-host for us; Chinese provenance | China (Z.ai / Zhipu) |
| **Mistral 3 / Small 4** | Ministral dense 3B/8B/14B; Mistral Large 3 MoE 675B/41B-active; Small 4 unified | **Apache 2.0** | Strong (small–mid), frontier (Large 3) | Clean Apache license, EU provenance, base/instruct/reasoning variants, good efficiency | Large 3 is very big; smaller models slightly behind Qwen on some tasks | France (EU) |
| **Google Gemma 4** | E2B (~2.3B eff), E4B (~4.5B eff), 12B, 26B-A4B MoE, 31B dense | **Apache 2.0** (new for Gemma 4; older Gemmas had custom terms) | Good–strong | Now truly permissive, multimodal, 256K context on larger sizes, runs phone→server | Slightly behind Qwen/Llama at the top end; brand-new license so less field history | USA (Google) |
| **DeepSeek V4** | V4-Pro 1.6T/49B-active; V4-Flash 284B/13B-active; 1M context | **MIT** | Frontier-class | Very strong reasoning/coding, cheap hosted API, MIT weights | Both variants are **too large to self-host** on a modest box; Chinese provenance | China (DeepSeek) |
| **Microsoft Phi-4-reasoning-vision-15B** | ~15B (plus smaller Phi-4 line) | Permissive (MIT-style) | Good "small-but-smart" | Compact, strong reasoning-for-size, decides when to "think", multimodal, easy to host | Not frontier; smaller world-knowledge; best as an efficiency play | USA (Microsoft) |

**Read for our use case:** our workloads are **short, JSON-structured, moderate-reasoning** tasks — not 1M-context frontier coding. So the giant MoE models (GLM 5.2, DeepSeek V4, Mistral Large 3, Llama 4 Behemoth) are **overkill and unhostable** on a modest box. The sweet spot is a **strong dense model in the 8B–32B range** (Qwen 3.6-27B / Qwen3 8B–32B, Llama, Gemma 4, Mistral/Ministral, Phi).

---

## 4. On the GLM / geopolitical concern Ravi raised

**What GLM is.** GLM ("General Language Model") is the model family from **Z.ai (formerly Zhipu AI)**, a Beijing-based AI lab. **GLM 5.2** (open weights released June 17, 2026 on Hugging Face and ModelScope under the **MIT license**) is a ~744–753B-parameter mixture-of-experts model with ~40B active parameters and a 1M-token context window. It is genuinely strong — VentureBeat and others report it ranking second on Code Arena behind Claude Opus 4.8 and beating GPT-5.5 on several long-horizon coding benchmarks at a fraction of the cost.

**The realistic governance picture — be precise, not alarmist.** The concern people cite (e.g. DHS warnings, TechTimes) is that a Chinese company is subject to China's **National Intelligence Law (Article 7)**, which can compel Chinese organizations to cooperate with state intelligence. **That risk is about the hosted API** — where your prompts travel to Z.ai's servers in China. It is a real reason to avoid the *hosted GLM API* for sensitive data.

**Crucially, that runtime data-exfiltration risk largely evaporates when you self-host open weights.** Open weights are just a big file of numbers. When you download them and run them on **your own hardware**, inference happens locally: **prompts and outputs never leave your infrastructure**, there is no phone-home, and the model file cannot exfiltrate data on its own. So for **self-hosted** open weights, the concern is **not runtime data leakage** — it's more about **provenance, supply-chain, and policy optics**:

- **Supply-chain / integrity:** is the weight file what it claims to be, unmodified?
- **Policy / procurement optics:** some orgs have policies or customer commitments that disfavor Chinese-origin models regardless of hosting.
- **Model behavior/bias:** training data and alignment reflect the origin; validate on our own tasks.

**How to de-risk if we ever did use a Chinese-origin model (GLM or Qwen):**

1. **Self-host only** — never route sensitive prompts through the vendor's hosted API.
2. **Verify weight checksums** against the official Hugging Face hashes before loading.
3. **Run with no outbound network / no telemetry** — air-gap the inference box or firewall egress; the serving stacks we use (§5) don't require internet at inference time.
4. **Evaluate on our own benchmark** — run it against our 3A translation-QA eval set and 2A cleanup samples; judge on our data, not marketing benchmarks.
5. **Pin and archive the exact weight version** so behavior is reproducible.

**Bottom line for Ravi:** GLM 5.2 is technically excellent, but (a) at ~744B/~1.5 TB of GPU memory it is **impractical for us to self-host** on any modest hardware, and (b) if provenance optics matter to the team, an equally-open **Apache-2.0** alternative avoids the debate entirely. Note the same China-provenance point applies to **Qwen** — but Qwen ships in **self-hostable 8B–32B sizes**, so it stays on the table with the de-risking steps above. My primary recommendation (§6) sidesteps the concern by default.

---

## 5. Serving stacks: which server hosts the model

All five below expose an **OpenAI-compatible `/v1/chat/completions`** endpoint, so all five plug into our existing `openai` branch. They differ on ease vs. throughput and — importantly for us — on how cleanly they honor structured JSON output.

| Stack | Ease | Throughput / scale | OpenAI-compatible API | Structured JSON | Best for |
|---|---|---|---|---|---|
| **Ollama** | Easiest — one command to pull & run | Modest (single-node, good for low/mid concurrency) | Yes (`:11434/v1`) | Supports `format`/JSON mode; good enough for prototyping | **Prototype**, dev boxes, quick internal use |
| **vLLM** | Moderate (Python/server setup) | **Highest** — paged-attention, continuous batching, multi-GPU | Yes (`:8000/v1`) | **Strong** — guided decoding (xgrammar default in 2026, plus outlines / lm-format-enforcer) for JSON-schema-constrained output | **Production**, high concurrency, strict JSON |
| **llama.cpp (server)** | Easy–moderate | Good on CPU/GPU/Apple; single-node | Yes | **Caveat:** historically buggy `response_format` on the OpenAI-compat route (`json_schema` vs `grammar` conflict reported) — verify before relying on strict JSON | Edge / CPU / Mac, cost-minimal hosting |
| **LM Studio** | Easiest GUI | Modest (desktop-oriented) | Yes | JSON mode via GUI | Individual dev experimentation, demos |
| **TGI (Hugging Face Text Generation Inference)** | Moderate | High (production-grade) | Yes | Guidance / grammar support | Production if standardizing on HF stack |

**Takeaway for us:** **Ollama for the prototype** (fastest to a working local endpoint behind the `openai` branch), **vLLM for production** (throughput + the most reliable structured-JSON support, which our Pydantic schemas depend on). Avoid leaning on llama.cpp's OpenAI `response_format` for strict JSON until we've verified the current build handles it.

---

## 6. Hardware guidance (rough, single-GPU reality check)

Quantization (esp. **Q4_K_M**, 4-bit) cuts VRAM ~75% vs FP16 with little quality loss for our task type. Reserve ~20–30% extra for context/overhead. Rough figures from current guides:

| Model size | FP16 VRAM | 4-bit (Q4) VRAM | Fits on…? |
|---|---|---|---|
| **7–8B** | ~14–16 GB | **~5–7 GB** | Any modern GPU (even 8–12 GB), or CPU/Mac |
| **32–34B** | ~64–68 GB | **~22–24 GB** | **A single 24 GB card (RTX 3090/4090)** — the sweet spot |
| **70B** | ~140 GB | ~35–40 GB | Needs **2× 24 GB** for full-GPU; or partial offload to RAM (slower) |
| **~700B MoE (GLM 5.2 / DeepSeek V4-Pro)** | ~1.5 TB | still very large | **Multi-GPU cluster only — not a single box** |

**What a modest single-GPU box (one 24 GB GPU) can realistically run:** comfortably an **8B model at high speed**, or a **32B model at Q4** with good throughput. That is exactly the range our workloads need. A 70B needs two cards; the ~700B frontier MoEs are out of scope for us.

---

## 7. Recommendation for THIS team

Our two live workloads — **2A service-request cleanup** and **3A translation quality checker** — are **short-context, moderate-reasoning tasks whose output must be STRICT JSON** (validated by Pydantic). They do **not** need 1M context or frontier coding ability. That profile favors a **strong, permissively-licensed dense model in the 8B–32B range on a stack with rock-solid JSON support** — not a giant Chinese MoE.

### Primary pick: **Qwen 3.6-27B** (Apache 2.0), served via **Ollama → vLLM**

- **Quality/size fit:** top-tier open quality at a size that runs on a single 24 GB GPU at Q4. Strong instruction-following and JSON reliability, excellent multilingual — directly useful for the **3A translation** work.
- **License:** **Apache 2.0** — clean commercial self-hosting, no MAU trap, no attribution string.
- **Extensibility:** one vLLM endpoint serves *all* projects through the existing `openai` branch. New project = new prompts, **zero adapter rewrite**.
- **Governance note:** Qwen is Chinese-origin like GLM, but we **self-host** it (weights on our hardware, no API), and it comes in a **hostable size** GLM does not. Apply the §4 de-risking (checksum, no egress, eval on our data). If provenance optics are a hard blocker for the team, use the fallback below instead — the migration is one env var.

### Fallback pick (non-China provenance): **Meta Llama (latest, ~8B) or Google Gemma 4 (12B/31B)** — or **Mistral/Ministral 3** (EU)

- Choose this if the **geopolitical concern is a hard rule**. Llama (USA), Gemma 4 (USA, now **Apache 2.0**), and Mistral (France, **Apache 2.0**) all avoid the China-provenance debate.
- **Llama** carries the Community License MAU cap (fine for us, well under 700M) plus a "Built with Llama" attribution requirement; **Gemma 4 and Mistral are cleaner (Apache 2.0)**.
- Swapping is trivial: same serving stack, same `openai` branch — just change `OPENAI_MODEL`.

### Why NOT GLM 5.2 / DeepSeek V4 as the pick
Both are genuinely excellent, but at **~700B+ / ~1.5 TB GPU memory** they're **unhostable on our hardware** and overkill for short JSON tasks. GLM also sits at the center of Ravi's provenance concern. Keep them noted as "if we ever need frontier coding and have a GPU cluster" options.

### Concrete first step (this sprint)
1. On a box with one 24 GB GPU, install **Ollama** and `ollama pull qwen3.6:27b` (or an 8B first for speed).
2. Set env only — **no code change**:
   ```
   MODEL_PROVIDER=openai
   OPENAI_BASE_URL=http://localhost:11434/v1
   OPENAI_API_KEY=local
   OPENAI_MODEL=qwen3.6:27b
   ```
3. **Benchmark on our 3A translation-QA eval set** (and 2A cleanup samples): measure (a) JSON-schema pass rate against our Pydantic models, (b) latency, (c) answer quality vs. our current cloud provider.
4. If JSON pass-rate or throughput needs hardening for production, move the same model behind **vLLM** with guided decoding (xgrammar) — still the same `openai` branch, still zero business-logic change.
5. Verify weight checksums and firewall the inference box's egress before any sensitive data.

If the benchmark passes, we have a private, cost-controlled, provider-agnostic inference path that any current or future project inherits for free.

---

## Sources

- [Llama (language model) — Wikipedia](https://en.wikipedia.org/wiki/Llama_(language_model))
- [The Llama 4 herd — Meta AI](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Llama 4 LICENSE — meta-llama/llama-models (GitHub)](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE)
- [Llama 4 Complete Guide 2026 (incl. Muse Spark) — Codersera](https://codersera.com/blog/llama-4-complete-guide-2026/)
- [Qwen — Wikipedia](https://en.wikipedia.org/wiki/Qwen)
- [Qwen3.6 Series: Alibaba's Open-Source LLM 2026 — AI/ML API](https://aimlapi.com/blog/qwen-3-6-series-alibabas-open-source-llm-revolution-in-2026)
- [How to Use Qwen3.6-27B — Tosea.ai](https://tosea.ai/blog/qwen-3-6-27b-complete-guide)
- [Introducing Mistral 3 — Mistral AI](https://mistral.ai/news/mistral-3/)
- [Mistral Large 3: 675B Open-Weight MoE — Effloow](https://effloow.com/articles/mistral-large-3-moe-work-mode-developer-guide-2026)
- [Gemma 4 — Google Blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Gemma 4 Complete Guide 2026 — Codersera](https://codersera.com/blog/gemma-4-complete-guide-2026/)
- [Z.ai's GLM-5.2 beats GPT-5.5 on coding benchmarks — VentureBeat](https://venturebeat.com/technology/z-ais-open-weights-glm-5-2-beats-gpt-5-5-on-multiple-long-horizon-coding-benchmarks-for-1-6th-the-cost)
- [Zhipu AI Releases GLM-5.2 Open-Weights (1M context) — AI Intel Report](https://aiintelreport.com/frontier-models/zhipu-ai-glm-5-2-open-weights)
- [GLM-5.2 Open Weights Live: China Data Risk on API — TechTimes](https://www.techtimes.com/articles/318543/20260617/glm-52-open-weights-live-top-coding-benchmark-api-use-carries-china-data-risk.htm)
- [Self-Host GLM 5.2: Open Weights & vLLM Guide — Lushbinary](https://lushbinary.com/blog/glm-5-2-self-hosting-open-weights-vllm-guide/)
- [DeepSeek V4 Open-Weights Launch — WinBuzzer](https://winbuzzer.com/2026/04/27/deepseek-v4-open-weights-launch-xcxwbn/)
- [DeepSeek-V4-Pro — Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [Microsoft releases Phi-4-reasoning-vision-15B — MLQ News](https://mlq.ai/news/microsoft-releases-phi-4-reasoning-vision-15b-open-weight-multimodal-ai-model/)
- [OpenAI compatibility — Ollama Docs](https://docs.ollama.com/api/openai-compatibility)
- [response_format issue on llama.cpp v1/chat/completions — GitHub #11847](https://github.com/ggml-org/llama.cpp/issues/11847)
- [Structured Decoding in vLLM: A Gentle Introduction — BentoML](https://www.bentoml.com/blog/structured-decoding-in-vllm-a-gentle-introduction)
- [Structured outputs in vLLM — Red Hat Developer](https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses)
- [Local LLM Deployment on 24GB GPUs — IntuitionLabs](https://intuitionlabs.ai/articles/local-llm-deployment-24gb-gpu-optimization)
- [LLM Quantization: Q4, Q8, FP16 and VRAM Tradeoffs (2026) — llmhardware.io](https://llmhardware.io/guides/llm-quantization-guide)
- [Run 70B LLMs on Consumer GPU — EaseCloud](https://blog.easecloud.io/ai-cloud/run-70b-models-on-consumer-gpus/)
