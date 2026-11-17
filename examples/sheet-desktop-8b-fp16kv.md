# Fit sheet: example-dense-8b on example desktop, 24 GB GPU

- Usable memory: 22.1 GB of 24 GB (dedicated)
- Bandwidth: 800 GB/s (spec x 0.8 (example value)); users: 1; KV cache: fp16
- Speeds are single-user ceilings, not promises; real runs usually land at 50-85% of them.

| quant | context | weights GB | KV GB | need GB | fit | decode tok/s (short / full ctx) | prefill tok/s | flags |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | 32768 | ~8.53 | 4.29 | 14.25 | FITS | 98.7 / 65.2 | 2491 |  |
| Q8_0 | 131072 | ~8.53 | 17.18 | 27.14 | SPILLS | 11.9 / 3.8 | 2491 |  |
| Q6_K | 32768 | ~6.62 | 4.29 | 12.25 | FITS | 126.5 / 76.3 | 2491 |  |
| Q6_K | 131072 | ~6.62 | 17.18 | 25.14 | SPILLS | 18.9 / 5.1 | 2491 |  |
| Q5_K_M | 32768 | ~5.72 | 4.29 | 11.3 | FITS | 146.0 / 83.0 | 2491 |  |
| Q5_K_M | 131072 | ~5.72 | 17.18 | 24.19 | SPILLS | 26.3 / 6.4 | 2491 |  |
| Q4_K_M | 32768 | ~4.87 | 4.29 | 10.41 | FITS | 170.8 / 90.4 | 2491 |  |
| Q4_K_M | 131072 | ~4.87 | 17.18 | 23.29 | SPILLS | 42.0 / 9.0 | 2491 |  |
| Q3_K_M | 32768 | ~3.91 | 4.29 | 9.41 | FITS | 210.9 / 100.6 | 2491 | LOSSY-RISK |
| Q3_K_M | 131072 | ~3.91 | 17.18 | 22.29 | SPILLS | 126.9 / 23.1 | 2491 | LOSSY-RISK |

`~` = estimated from bits per weight; put the real file size in `file_gb` to replace it.

Nothing safe fits at 131072 tokens; the longest context with a safe pick is 32768.

**Pick: Q8_0 at 32768 tokens** (FITS): 7.8 GB spare, ceiling ~65.2 tok/s at full context.
