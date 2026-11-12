# Interleaved speed repeats: depth2

Decode tok/s, mean over the prompts (code, math, prose), per round; no profiler.

| arm | build | round 1 | round 2 | round 3 | mean | ms/token |
|---|---|---|---|---|---|---|
| f1-0 | 1.6.5+w9 {"EXL3_MGEMM_N_THRESHOLD": "0"} | 43.82 | 38.99 | 41.93 | 41.58 | 24.05 |
| w11-0 | 1.6.5+w11b {"EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 43.33 | 43.59 | 42.78 | 43.24 | 23.13 |
| w11L-0 | 1.6.5+w11b {"EXL3_FD75D_LIVE": "1", "EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 44.57 | 43.81 | 39.55 | 42.65 | 23.45 |
| f1-8k | 1.6.5+w9 {"EXL3_MGEMM_N_THRESHOLD": "0"} | 37.00 | 34.52 | 33.29 | 34.94 | 28.62 |
| w11-8k | 1.6.5+w11b {"EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 42.92 | 40.38 | 39.50 | 40.93 | 24.43 |
| w11L-8k | 1.6.5+w11b {"EXL3_FD75D_LIVE": "1", "EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 42.80 | 40.30 | 39.52 | 40.87 | 24.47 |
| f1-28k | 1.6.5+w9 {"EXL3_MGEMM_N_THRESHOLD": "0"} | 28.45 | 29.24 | 28.99 | 28.90 | 34.61 |
| w11-28k | 1.6.5+w11b {"EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 38.75 | 42.41 | 38.57 | 39.91 | 25.06 |
| w11L-28k | 1.6.5+w11b {"EXL3_FD75D_LIVE": "1", "EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 39.22 | 40.50 | 39.29 | 39.67 | 25.21 |
| w11off-28k | 1.6.5+w11b {"EXL3_FD75D": "0", "EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 29.11 | 30.29 | 29.52 | 29.64 | 33.74 |
| f1m-28k | 1.6.5+w9 {"EXL3_MGEMM_N_THRESHOLD": "0"} | 32.18 | 38.02 | 42.19 | 37.46 | 26.69 |
| w11Lm-28k | 1.6.5+w11b {"EXL3_FD75D_LIVE": "1", "EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 53.58 | 50.25 | 52.56 | 52.13 | 19.18 |

**w11-0 against f1-0** (tok/s change per round):

round 1: -1.1%, round 2: +11.8%, round 3: +2.0%; faster in 2 of 3 rounds; mean +4.2%

**w11-8k against f1-8k** (tok/s change per round):

round 1: +16.0%, round 2: +17.0%, round 3: +18.6%; faster in 3 of 3 rounds; mean +17.2%

**w11-28k against f1-28k** (tok/s change per round):

round 1: +36.2%, round 2: +45.0%, round 3: +33.0%; faster in 3 of 3 rounds; mean +38.1%

**w11L-0 against w11-0** (tok/s change per round):

round 1: +2.9%, round 2: +0.5%, round 3: -7.5%; faster in 2 of 3 rounds; mean -1.4%

**w11L-8k against w11-8k** (tok/s change per round):

round 1: -0.3%, round 2: -0.2%, round 3: +0.1%; faster in 1 of 3 rounds; mean -0.1%

**w11L-28k against w11-28k** (tok/s change per round):

round 1: +1.2%, round 2: -4.5%, round 3: +1.8%; faster in 2 of 3 rounds; mean -0.5%

**w11off-28k against f1-28k** (tok/s change per round):

round 1: +2.3%, round 2: +3.6%, round 3: +1.8%; faster in 3 of 3 rounds; mean +2.6%

**w11Lm-28k against f1m-28k** (tok/s change per round):

round 1: +66.5%, round 2: +32.2%, round 3: +24.6%; faster in 3 of 3 rounds; mean +41.1%
