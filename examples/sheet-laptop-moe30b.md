# Fit sheet: example-moe-30b-a3b on example laptop, 32 GB unified memory

- Usable memory: 22.4 GB of 32 GB (unified)
- Bandwidth: 100 GB/s (measured copy (example value)); users: 1; KV cache: q8
- Speeds are single-user ceilings, not promises; real runs usually land at 50-85% of them.

| quant | context | weights GB | KV GB | need GB | fit | decode tok/s (short / full ctx) | prefill tok/s | flags |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | 8192 | ~32.43 | 0.4 | 35.45 | NO | - | 898 |  |
| Q8_0 | 32768 | ~32.43 | 1.61 | 36.66 | NO | - | 898 |  |
| Q6_K | 8192 | ~25.18 | 0.4 | 27.84 | NO | - | 898 |  |
| Q6_K | 32768 | ~25.18 | 1.61 | 29.05 | NO | - | 898 |  |
| Q5_K_M | 8192 | ~21.75 | 0.4 | 24.24 | NO | - | 898 |  |
| Q5_K_M | 32768 | ~21.75 | 1.61 | 25.44 | NO | - | 898 |  |
| Q4_K_M | 8192 | ~18.5 | 0.4 | 20.83 | TIGHT | 48.7 / 41.5 | 898 |  |
| Q4_K_M | 32768 | ~18.5 | 1.61 | 22.04 | TIGHT | 48.7 / 27.7 | 898 |  |
| Q3_K_M | 8192 | ~14.88 | 0.4 | 17.03 | FITS | 60.2 / 49.6 | 898 | LOSSY-RISK |
| Q3_K_M | 32768 | ~14.88 | 1.61 | 18.23 | FITS | 60.2 / 31.0 | 898 | LOSSY-RISK |

`~` = estimated from bits per weight; put the real file size in `file_gb` to replace it.

**Pick: Q4_K_M at 32768 tokens** (TIGHT): 0.4 GB spare, ceiling ~27.7 tok/s at full context.
