# Qwen3.8 27B (Q4_K_M) vs. Ternary-Bonsai-2 27B (PQ2_0) — Head-to-Head on a Single RTX 4090

Measured comparison of two ~27B local LLMs served over OpenAI-compatible APIs on a single machine, tested for use as an agent model (tool calling, code, math). All numbers are raw measurements from the same hardware, same day.

## Setup

| | **Qwen3.8 27B** | **Ternary-Bonsai-2 27B** |
|---|---|---|
| Quantization | GGUF Q4_K_M (17 GB file) | GGUF PQ2_0 ternary (7.2 GB file) |
| Runtime | Ollama 0.34.4 (`localhost:11434`) | llama-server (PrismML fork, win-cuda-12.4) on port 18435 |
| Context | 32,768 (Ollama default) | 65,536 (`-c 65536`) |
| GPU | RTX 4090, 24 GB, CUDA 12.4 (both) | |
| VRAM when loaded | ~18.5 GB | ~13.1 GB |

Both endpoints speak the OpenAI `chat/completions` API and were driven with the same benchmark script (streaming requests, sequential runs, since the two models cannot share the GPU — see below).

## Measured Results

### Speed

| Metric | Qwen3.8 | Bonsai 2 |
|---|---|---|
| Generation throughput (warm) | **~44 tok/s** (1,500-token generation, 33.8 s) | **~80 tok/s** (1,024-token generation, 12.7 s) |
| Prompt processing | ~120–220 tok/s (Ollama-reported) | ~2,800 tok/s for a 10,279-token prompt (llama-server log) |
| Cold start (model load into VRAM) | ~25–28 s | ~20 s |
| Time-to-first-token (warm, math task) | ~19 s (short thinking prefix) | ~1.5 s |

### Task accuracy (same prompts, both models)

| Task | Qwen3.8 | Bonsai 2 |
|---|---|---|
| Math (bat-and-ball trap: $1.10 total, bat $1.00 more) | ✅ `0.05` with brief reasoning | ✅ `0.05` with brief reasoning |
| Code (order-preserving dict dedupe, "keep last occurrence") | ✅ correct, compact one-shot answer | ❌ **empty response** — spent the entire 4,096-token budget on reasoning, `finish_reason: length`, zero content (51 s of wall time) |
| Tool calling (weather tool) | ✅ clean `tool_calls` payload, 1.7 s | ✅ clean `tool_calls` payload, 1.3 s |
| Simple chat end-to-end (agent harness, one turn) | ✅ 35 s incl. cold load | ✅ 8 s warm |

## Key Findings

### 1. Bonsai 2 can burn its entire output budget on thinking and return nothing

The PrismML llama-server build exposes the model's chain-of-thought in a `reasoning_content` field (not `reasoning`) on the chat-completion message. On the code task — which is genuinely ambiguous ("keep last occurrence" + "preserve order" has two defensible interpretations) — Bonsai 2 kept deliberating and hit the token cap at both 2,048 and 4,096 `max_tokens`, returning `content: ""` with `finish_reason: length`. A client that only reads `content` gets a total failure after 25–51 seconds of waiting.

Qwen3.8 also thinks, but its thinking is brief (a few hundred tokens) and it always produces a final answer.

### 2. The two models cannot run concurrently on one 24 GB card

18.5 GB + 13.1 GB > 24 GB. Whichever loads second dies with `cudaMalloc failed: out of memory`:

- If Bonsai loads while Qwen is resident: llama-server exits at model load.
- If Qwen loads while Bonsai is resident: Ollama returns HTTP 500 with `llama-server startup failed ... out of memory` and the client must retry.

Ollama auto-evicts its model after ~5 minutes of idle, so switching models eventually works on its own, but a clean switch is explicit:

```bash
ollama stop qwen3.8:27b-q4_K_M     # free ~18.5 GB before loading Bonsai
# kill the Bonsai llama-server before loading Qwen
```

### 3. Bonsai 2 is much faster — when it produces output

~80 tok/s vs. ~44 tok/s is a real 1.8× advantage, and prompt processing is an order of magnitude faster. The PQ2_0 ternary file also fits the whole model in half the file size. The speed win is real but conditional on the model actually finishing.

## Verdict

**Qwen3.8 27B Q4_K_M is the safer agent model.** Bounded thinking, always produces content, reliable tool calls, and it's the more recent base model.

**Bonsai 2 27B PQ2_0 is faster but has a hard reliability flaw** for agentic use: unbounded thinking that can consume the whole token budget and emit nothing. That failure mode is triggered by exactly the kind of ambiguous, multi-constraint prompts that agent harnesses generate constantly, so it isn't an edge case.

Rules of thumb:

- Default agent model on a 24 GB card: **Qwen3.8 27B**.
- Bonsai 2 is a good choice for short, unambiguous, latency-sensitive prompts (Q&A, completions) where the caller can detect empty `content` and retry with a larger budget or a non-thinking prompt.
- If using Bonsai 2, always check `reasoning_content`/`content` and treat `finish_reason: length` with empty `content` as a retryable failure, not a result.

## Reproducing

```bash
# Qwen3.8 via Ollama
ollama pull qwen3.8:27b-q4_K_M
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "qwen3.8:27b-q4_K_M",
  "messages": [{"role": "user", "content": "hello"}]
}'

# Bonsai 2 via PrismML llama-server (stock llama.cpp/Ollama cannot load PQ2_0)
./llama-server.exe -m Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 --n-cpu-moe 0 -c 65536 --port 18435 --host 127.0.0.1 \
  --alias Ternary-Bonsai-2-27B
```

Bench script: streaming `chat/completions` requests against both endpoints, three tasks (math, code, tool call) plus a 1,500/1,024-token generation for throughput; TTFT measured to first non-reasoning content token.
