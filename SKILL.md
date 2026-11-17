---
name: before-you-pull
description: Decide, before downloading tens of gigabytes, whether a local LLM will fit on this machine and how fast it can possibly run. Builds a machine card and a model card (straight from the model's config.json), then a fit sheet across quantizations and context lengths with memory need, FITS/TIGHT/SPILLS/NO, decode and prefill speed ceilings, and a single recommended pick; after download, one short run is checked against the ceiling to catch CPU fallback or broken setups. Use when someone asks "can I run model X on my GPU/Mac/laptop?", "which quant should I download?", "how much context can I afford?", "why is it so slow?", or is choosing between local models for Ollama, LM Studio, llama.cpp, MLX, vLLM or similar.
---

# Before You Pull

## What this is

People download a 40 GB model, wait an hour, and then find it either does not load, loads
but crawls because half of it spilled into system RAM, or fits only with a context too short
for their work. All of that is predictable from a few numbers you already have. This skill
does the arithmetic first and hands over one clear recommendation, with its assumptions shown.

It rests on three facts about running language models locally:

1. **Memory need = weights + KV cache + runtime overhead.** Weights depend on the quantization;
   the KV cache grows with context length and with the number of simultaneous users; the runtime
   needs a little on top. Long context is often what breaks the budget, not the model itself.
2. **Generating text is limited by memory bandwidth.** Every new token reads the active weights
   (and the KV cache so far) once. So the speed ceiling is roughly
   `bandwidth / bytes read per token`. Mixture-of-experts models read only their active experts,
   which is why a 30B MoE can out-run a dense 8B on the same machine.
3. **Reading the prompt (prefill) is limited by compute.** About `2 x active parameters` FLOPs per
   token; a practical ceiling is half of the hardware's measured FP16 throughput.

The result is a ceiling, not a promise: healthy setups usually land at 50-85% of it. Far below
means something is wrong; above means the measurement is wrong.

## When to use it

- Choosing a model or quantization to download for Ollama, LM Studio, llama.cpp, MLX, vLLM,
  SGLang, ExLlama, or a similar runtime.
- Deciding how much context (or how many parallel users) a machine can afford.
- Comparing machines for a purchase: "is 32 GB unified enough for this model?"
- Diagnosing a slow local model after the fact (`check`).

Not for: hosted APIs; training or fine-tuning memory; squeezing the last 10% out of a server
that already runs well (that is tuning work, done by measured experiments one change at a time).

## Step by step

### 1. Fill in the machine card
`python3 scripts/fitsheet.py machine > machine.json`

- `memory_gb`: GPU memory, or total RAM on unified-memory machines (Apple silicon, many new laptops).
- `unified`: true when the GPU shares system memory. The OS and apps keep a share, so the default
  usable fraction is 70% (92% on a dedicated GPU). Override `usable_fraction` if you know better.
- `bandwidth_gbs`: **measured** copy bandwidth if you can (a large device-to-device copy, or a
  memory benchmark); otherwise the spec value x 0.8, and set `bandwidth_source` to say so.
- `fp16_tflops` (optional): measured large-matmul throughput, for prefill estimates.
- `system_ram_gb`, `ram_bandwidth_gbs`: used to estimate how slow a partial CPU spill would be.
- `disk_free_gb`: catches the "it won't even download" case.

Ask only for what you can't detect. On the user's machine you may read it yourself
(`nvidia-smi`, `system_profiler SPHardwareDataType`, `free -g`, `df -h`), read-only.

### 2. Build the model card
Best: download only the model's `config.json` (a few KB) and run
`python3 scripts/fitsheet.py model --from-config config.json > model.json`.
It estimates total and active parameters from the shapes (attention, MLP or experts,
embeddings) and fills layers, KV heads, head size, sliding-window layers and max context.
Then add real file sizes from the download page to `file_gb`, e.g. `{"Q4_K_M": 18.6}`: a real
size always beats the bits-per-weight guess, and the sheet marks guesses with `~`.

Sanity-check the parameter estimate against the model's name ("8B", "30B-A3B"). If it is far off,
the config uses a layout the estimator doesn't know (hybrid/linear attention, unusual experts);
type the numbers in by hand.

### 3. Produce the fit sheet
`python3 scripts/fitsheet.py sheet machine.json model.json --contexts 8192,32768 --users 1`

Options: `--kv q8` or `--kv q4` (quantized KV cache, if the runtime supports it), `--quants` to
choose rows, `--min-tps` for the slowest acceptable speed (default 10 tok/s), `--allow-tight`.

Each row reads:

| fit | meaning |
|---|---|
| FITS | needs at most 90% of usable memory |
| TIGHT | fits, but with little margin: close other apps, expect trouble at peak |
| SPILLS | only fits if part of the model runs from system RAM: works, but slowly |
| NO | does not fit at all |

Flags: `LOSSY-RISK` (under 4 bits per weight, where quality loss becomes noticeable, especially
for small models), `OVER-MAX-CONTEXT`, `DISK`.

### 4. Make the pick and explain it
The sheet suggests the highest-precision quant that fits, meets the speed floor and has no flags,
at the longest requested context that allows one. Present it in three lines:
**what to download**, **what it costs** (memory left over, expected speed range = 50-85% of the
ceiling), and **what was traded** (shorter context, KV quantization, smaller model). If there is
no safe pick, say which lever helps most: shorter context, `--kv q8`, a smaller or MoE model,
or a different machine. Don't recommend SPILLS rows for interactive use without saying they
will be slow.

### 5. After download: check one real run
Run one short generation in the user's runtime, note the decode tok/s it reports, and run
`python3 scripts/fitsheet.py check --predicted <ceiling> --measured <tok/s>`.

- HEALTHY (50-105%): done.
- LOW (25-50%): partial CPU offload, wrong backend (e.g. CPU build), laptop power-saving, another GPU job.
- BROKEN (<25%): running on CPU or swapping. Fix before anything else.
- ABOVE CEILING: suspect the measurement (cached prompt, tokens counted per chunk, speculative
  decoding not in the card).

## Examples

**Long context on a 24 GB desktop GPU (dense 8B).** At 131,072 tokens the fp16 KV cache alone is
about 17 GB, so Q8_0 needs ~27 GB and SPILLS (ceiling drops to a few tok/s). Re-running with
`--kv q8` halves the cache: Q8_0 at 131,072 now FITS (18.6 GB of 22.1 usable) with a ceiling
of ~48 tok/s at full context and ~100 tok/s on short chats. Pick: Q8_0 + q8 KV cache.
(`examples/sheet-desktop-8b-*.md`; example machine values.)

**A 30B mixture-of-experts on a 32 GB unified-memory laptop.** Only ~22 GB is usable. Q5 and
above are NO; Q4_K_M at 32k context is TIGHT (0.4 GB spare) with a ceiling of ~49 tok/s short /
~28 tok/s at full context, fast because only ~3B parameters are active. Q3 FITS comfortably but
carries LOSSY-RISK. Recommendation: Q4_K_M with q8 KV, close the browser, or limit context to 8k.
(`examples/sheet-laptop-moe30b.md`.)

**"It runs at 6 tok/s, the sheet said 60."** `check` says BROKEN: under 25% of the ceiling.
The usual causes are a CPU-only build, layers not offloaded to the GPU, or memory swapping.

## Tips

- Real file sizes > bits-per-weight guesses; measured bandwidth > spec sheets. Say which you used.
- Context is a budget: ask what the user actually needs (chat 8k, coding 32k, whole-repo 128k).
- Concurrency multiplies the KV cache: `--users 4` needs four caches.
- Hybrid models with sliding-window or linear-attention layers need much less KV; set
  `kv_layers`, `sliding_layers` and `sliding_window` correctly or the sheet will be too pessimistic.
- Unified memory: the GPU can't use all of it; leave room for the OS and the apps the user keeps open.
- Speculative decoding, batching and runtime tricks can beat the simple ceiling; treat a big
  measured win with suspicion until the output is checked.
- Never ask for or print access tokens to gated model repos; downloading `config.json` usually
  needs none.
