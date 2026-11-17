# Fit sheet: example-dense-8b on example desktop, 24 GB GPU

- Usable memory: 22.1 GB of 24 GB (dedicated)
- Bandwidth: 800 GB/s (spec x 0.8 (example value)); users: 1; KV cache: q8
- Speeds are single-user ceilings, not promises; real runs usually land at 50-85% of them.

| quant | context | weights GB | KV GB | need GB | fit | decode tok/s (short / full ctx) | prefill tok/s | flags |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | 32768 | ~8.53 | 2.15 | 12.11 | FITS | 99.5 / 79.0 | 2491 |  |
| Q8_0 | 131072 | ~8.53 | 8.59 | 18.55 | FITS | 99.5 / 48.3 | 2491 |  |
| Q6_K | 32768 | ~6.62 | 2.15 | 10.1 | FITS | 127.8 / 95.9 | 2491 |  |
| Q6_K | 131072 | ~6.62 | 8.59 | 16.55 | FITS | 127.8 / 54.1 | 2491 |  |
| Q5_K_M | 32768 | ~5.72 | 2.15 | 9.15 | FITS | 147.8 / 106.7 | 2491 |  |
| Q5_K_M | 131072 | ~5.72 | 8.59 | 15.6 | FITS | 147.8 / 57.4 | 2491 |  |
| Q4_K_M | 32768 | ~4.87 | 2.15 | 8.26 | FITS | 173.3 / 119.5 | 2491 |  |
| Q4_K_M | 131072 | ~4.87 | 8.59 | 14.7 | FITS | 173.3 / 60.9 | 2491 |  |
| Q3_K_M | 32768 | ~3.91 | 2.15 | 7.26 | FITS | 214.7 / 137.8 | 2491 | LOSSY-RISK |
| Q3_K_M | 131072 | ~3.91 | 8.59 | 13.7 | FITS | 214.7 / 65.3 | 2491 | LOSSY-RISK |

`~` = estimated from bits per weight; put the real file size in `file_gb` to replace it.

**Pick: Q8_0 at 131072 tokens** (FITS): 3.5 GB spare, ceiling ~48.3 tok/s at full context.
