# before-you-pull（下载之前）

[English](README.md) | **简体中文**

> 这个本地大模型装得下吗？最快能跑多快？在拉取 40 GB 权重之前，用一个 5 KB 的 `config.json` 就能知道。

`before-you-pull` 是一个智能体技能，适用于 Claude Code、Codex、Cursor 以及任何能读取 `SKILL.md` 的智能体。
它把"我的机器能跑 X 模型吗？"变成一页"装得下表"：每种量化和上下文长度需要多少内存，是 FITS（装得下）、TIGHT（勉强）、
SPILLS（溢出到系统内存）还是完全装不下，解码和预填充的速度上限，以及唯一推荐的下载版本。
下载后，只需跑一次短测试与上限对比，就能抓出退回 CPU 和配置错误。

## 为什么需要它

大多数"本地模型"的失望都是可以预见的：长上下文的 KV 缓存撑爆显存；统一内存笔记本没法把全部内存交给 GPU；一半的层悄悄跑在 CPU 上。
背后的原理很简单（内存 = 权重 + KV + 开销；生成速度 ≈ 带宽 ÷ 每个 token 读取的字节数），只是没人在下载前算一算。这个技能替你算。

## 功能

- **从 config.json 生成模型卡**：估算总参数与活跃参数（稠密和混合专家模型）、KV 头数、头维度、滑动窗口层、最大上下文。
- **机器卡**：独立显存与统一内存（含合理的系统预留）、实测与标称带宽、可选 FP16 算力、用于估算溢出的系统内存、剩余磁盘。
- **装得下表**：量化 × 上下文矩阵，包含权重、KV、总需求、FITS / TIGHT / SPILLS / NO、解码上限（短对话和满上下文）、预填充上限，以及标记（LOSSY-RISK、OVER-MAX-CONTEXT、DISK）。
- **KV 缓存选项**：fp16、q8、q4；用 `--users` 设置并发。
- **唯一推荐**：在允许的最长上下文下，精度最高且安全的量化。
- **现实检查**：`check` 把实测 tok/s 评为 HEALTHY / LOW / BROKEN / ABOVE CEILING。
- 真实文件大小会覆盖估算值并标注；纯 Python 标准库；12 个测试。

## 安装

```bash
git clone https://github.com/fangdididi/before-you-pull.git
cp -r before-you-pull ~/.claude/skills/        # Claude Code（个人）
cp -r before-you-pull .claude/skills/          # Claude Code（单个项目）
cp -r before-you-pull .agents/skills/          # Codex 及其他读取 .agents/skills 的智能体
# Cursor：放到 .cursor/skills/，或在规则里引用 SKILL.md
```

然后直接问：*"我有一台 16 GB 的 MacBook Air，能跑 14B 模型配 32k 上下文吗？该下载哪个量化？"*

## 辅助脚本

```bash
python3 scripts/fitsheet.py machine > machine.json                  # 填写机器卡
python3 scripts/fitsheet.py model --from-config config.json > model.json
python3 scripts/fitsheet.py sheet machine.json model.json --contexts 8192,32768 --kv q8
python3 scripts/fitsheet.py check --predicted 48 --measured 35      # 真实跑一次之后
python3 -m unittest discover -s tests
```

## 示例

`examples/sheet-desktop-8b-q8kv.md`（机器数值为示例值）：

```
| quant | context | weights GB | KV GB | need GB | fit  | decode tok/s (short / full ctx) |
| Q8_0  | 131072  | ~8.53      | 8.59  | 18.55   | FITS | 99.5 / 48.3                      |
**Pick: Q8_0 at 131072 tokens** (FITS): 3.5 GB spare, ceiling ~48.3 tok/s at full context.
```

同一个模型如果用 fp16 KV 缓存，在这个上下文下需要约 27 GB，会溢出（SPILLS）。正是量化缓存让 24 GB 显卡跑 128k 变得可行。

## 文件结构

```
before-you-pull/
├── SKILL.md / SKILL.zh-CN.md     # 技能说明（英文 / 中文）
├── README.md / README.zh-CN.md
├── scripts/fitsheet.py           # machine / model / sheet / check
├── examples/                     # 示例配置、卡片和生成的表格
├── tests/test_fitsheet.py
└── LICENSE
```

## 局限

上限是算出来的，不是跑分。混合线性注意力模型、投机解码和特殊运行时需要手动填写卡片数值；真实速度通常落在上限的 50–85%。

## 许可证

MIT © fangdididi
