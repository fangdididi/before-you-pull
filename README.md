# before-you-pull

**English** | [简体中文](README.zh-CN.md)

> Will this local LLM fit, and how fast can it possibly go? Find out from a 5 KB `config.json`
> before you pull 40 GB of weights.

`before-you-pull` is an agent skill for Claude Code, Codex, Cursor and any agent that reads
`SKILL.md` files. It turns "can I run model X on my machine?" into a one-page fit sheet:
memory needed per quantization and context length, whether it FITS, is TIGHT, SPILLS into
system RAM or doesn't fit at all, the decode and prefill speed ceilings, and one recommended
download. After download, a single short run is checked against the ceiling to catch CPU
fallback and broken setups.

## Why

Most "local model" disappointment is predictable: the KV cache for a long context outgrows the
GPU, a unified-memory laptop can't hand all its RAM to the GPU, half the layers quietly run on
the CPU. The physics is simple (memory = weights + KV + overhead; generation speed ≈ bandwidth ÷
bytes read per token), but nobody does the sums before downloading. This skill does.

## Features

- **Model card from config.json**: estimates total and active parameters (dense and
  mixture-of-experts), KV heads, head size, sliding-window layers, max context.
- **Machine card**: dedicated vs unified memory (with a sensible OS reserve), measured vs spec
  bandwidth, optional FP16 throughput, system RAM for spill estimates, free disk.
- **Fit sheet**: quant x context matrix with weights, KV, total need, FITS / TIGHT / SPILLS / NO,
  decode ceilings (short chat and full context), prefill ceiling, flags (LOSSY-RISK,
  OVER-MAX-CONTEXT, DISK).
- **KV cache options**: fp16, q8, q4; concurrency with `--users`.
- **One pick**: highest-precision safe quant at the longest context that allows it.
- **Reality check**: `check` grades a measured tok/s as HEALTHY / LOW / BROKEN / ABOVE CEILING.
- Real file sizes override estimates and are marked; pure Python standard library; 12 tests.

## Install

```bash
git clone https://github.com/fangdididi/before-you-pull.git
cp -r before-you-pull ~/.claude/skills/        # Claude Code (personal)
cp -r before-you-pull .claude/skills/          # Claude Code (one project)
cp -r before-you-pull .agents/skills/          # Codex and other .agents/skills readers
# Cursor: place it in .cursor/skills/ or reference SKILL.md from a rule
```

Then just ask: *"I have a 16 GB MacBook Air. Can I run a 14B model with 32k context, and which
quant should I download?"*

## Helper script

```bash
python3 scripts/fitsheet.py machine > machine.json                  # fill it in
python3 scripts/fitsheet.py model --from-config config.json > model.json
python3 scripts/fitsheet.py sheet machine.json model.json --contexts 8192,32768 --kv q8
python3 scripts/fitsheet.py check --predicted 48 --measured 35      # after one real run
python3 -m unittest discover -s tests
```

## Example

`examples/sheet-desktop-8b-q8kv.md` (example machine values):

```
| quant | context | weights GB | KV GB | need GB | fit  | decode tok/s (short / full ctx) |
| Q8_0  | 131072  | ~8.53      | 8.59  | 18.55   | FITS | 99.5 / 48.3                      |
**Pick: Q8_0 at 131072 tokens** (FITS): 3.5 GB spare, ceiling ~48.3 tok/s at full context.
```

The same model with an fp16 KV cache needs ~27 GB at that context and SPILLS. Quantizing the
cache is what makes 128k practical on a 24 GB card.

## File structure

```
before-you-pull/
├── SKILL.md / SKILL.zh-CN.md     # skill instructions (English / Chinese)
├── README.md / README.zh-CN.md
├── scripts/fitsheet.py           # machine / model / sheet / check
├── examples/                     # example configs, cards and generated sheets
├── tests/test_fitsheet.py
└── LICENSE
```

## Limits

Ceilings are arithmetic, not benchmarks. Hybrid linear-attention models, speculative decoding and
unusual runtimes need hand-entered card values; real speed usually lands at 50-85% of the ceiling.

## License

MIT © fangdididi
