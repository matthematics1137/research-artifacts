# Interleaved speed repeats: final1

Decode tok/s, mean over the prompts (code, math, prose), per round; no profiler.

| arm | build | round 1 | round 2 | round 3 | mean | ms/token |
|---|---|---|---|---|---|---|
| stock-0 | 1.6.5 {} | 24.50 | 23.00 | 22.69 | 23.40 | 42.74 |
| adopt-0 | 1.6.5+w11b {"EXL3_FD75D_LIVE": "0", "EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 42.29 | 44.37 | 40.33 | 42.33 | 23.62 |
| stock-mtp-0 | 1.6.5 {} | 46.59 | 51.55 | 42.37 | 46.84 | 21.35 |
| adopt-mtp-0 | 1.6.5+w11b {"EXL3_FD75D_LIVE": "0", "EXL3_INT8_SQ_MAX_M": "2", "EXL3_MGEMM_N_THRESHOLD": "0"} | 52.49 | 58.23 | 49.00 | 53.24 | 18.78 |
| stock-28k | 1.6.5 {} | 19.37 | 19.96 | 17.00 | 18.78 | 53.26 |
| adopt-28k | 1.6.5+w11b {"EXL3_FD75D_LIVE": "0", "EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 39.83 | 41.19 | 39.02 | 40.01 | 24.99 |
| stock-mtp-28k | 1.6.5 {} | 37.14 | 37.10 | 36.09 | 36.78 | 27.19 |
| adopt-mtp-28k | 1.6.5+w11b {"EXL3_FD75D_LIVE": "0", "EXL3_INT8_SQ_MAX_M": "2", "EXL3_MGEMM_N_THRESHOLD": "0"} | 51.36 | 50.02 | 50.93 | 50.77 | 19.70 |

**adopt-0 against stock-0** (tok/s change per round):

round 1: +72.6%, round 2: +92.9%, round 3: +77.7%; faster in 3 of 3 rounds; mean +81.1%

**adopt-28k against stock-28k** (tok/s change per round):

round 1: +105.6%, round 2: +106.3%, round 3: +129.5%; faster in 3 of 3 rounds; mean +113.8%

**adopt-mtp-0 against stock-mtp-0** (tok/s change per round):

round 1: +12.6%, round 2: +13.0%, round 3: +15.7%; faster in 3 of 3 rounds; mean +13.8%

**adopt-mtp-28k against stock-mtp-28k** (tok/s change per round):

round 1: +38.3%, round 2: +34.8%, round 3: +41.1%; faster in 3 of 3 rounds; mean +38.1%
