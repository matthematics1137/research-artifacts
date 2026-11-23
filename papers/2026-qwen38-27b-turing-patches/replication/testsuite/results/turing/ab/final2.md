# Interleaved speed repeats: final2

Decode tok/s, mean over the prompts (code, math, prose), per round; no profiler.

| arm | build | round 1 | round 2 | round 3 | mean | ms/token | pooled tok/s | pooled ms/token |
|---|---|---|---|---|---|---|---|---|
| stock-0 | 1.6.5 {} | 25.71 | 25.16 | 25.12 | 25.33 | 39.47 | 25.33 | 39.47 |
| adopt-0 | 1.6.5+w11b {"EXL3_FD75D_LIVE": "0", "EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 45.76 | 45.20 | 45.39 | 45.45 | 22.00 | 45.45 | 22.00 |
| w13-0 | 1.6.5+w13 {"EXL3_FD75D_LIVE": "0", "EXL3_GEMV_SK": "1", "EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 47.85 | 47.44 | 47.31 | 47.53 | 21.04 | 47.53 | 21.04 |
| adopt-28k | 1.6.5+w11b {"EXL3_FD75D_LIVE": "0", "EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 41.97 | 41.91 | 41.84 | 41.91 | 23.86 | 41.91 | 23.86 |
| w13-28k | 1.6.5+w13 {"EXL3_FD75D_LIVE": "0", "EXL3_GEMV_SK": "1", "EXL3_INT8_SQ_MAX_M": "1", "EXL3_MGEMM_N_THRESHOLD": "0"} | 43.98 | 43.96 | 43.80 | 43.91 | 22.77 | 43.88 | 22.79 |
| stock-mtp-0 | 1.6.5 {} | 50.16 | 50.27 | 51.64 | 50.69 | 19.73 | 46.40 | 21.55 |
| adopt-mtp-0 | 1.6.5+w11b {"EXL3_FD75D_LIVE": "0", "EXL3_INT8_SQ_MAX_M": "2", "EXL3_MGEMM_N_THRESHOLD": "0"} | 57.58 | 57.49 | 59.04 | 58.04 | 17.23 | 54.60 | 18.32 |
| w13s1-mtp-0 | 1.6.5+w13 {"EXL3_FD75D_LIVE": "0", "EXL3_GEMV_SK": "1", "EXL3_INT8_SQ_MAX_M": "2", "EXL3_MGEMM_N_THRESHOLD": "0"} | 70.85 | 70.63 | 68.74 | 70.08 | 14.27 | 65.50 | 15.27 |
| w13s2-mtp-0 | 1.6.5+w13 {"EXL3_FD75D_LIVE": "0", "EXL3_GEMV_SK": "2", "EXL3_INT8_SQ_MAX_M": "2", "EXL3_MGEMM_N_THRESHOLD": "0"} | 79.54 | 75.58 | 75.84 | 76.99 | 12.99 | 71.61 | 13.97 |
| raf-mtp-0 | 1.5.3+rafatxf {} | 71.20 | 73.24 | 69.52 | 71.32 | 14.02 | 66.18 | 15.11 |
| adopt-mtp-28k | 1.6.5+w11b {"EXL3_FD75D_LIVE": "0", "EXL3_INT8_SQ_MAX_M": "2", "EXL3_MGEMM_N_THRESHOLD": "0"} | 54.12 | 54.78 | 54.27 | 54.39 | 18.39 | 50.51 | 19.80 |
| w13s1-mtp-28k | 1.6.5+w13 {"EXL3_FD75D_LIVE": "0", "EXL3_GEMV_SK": "1", "EXL3_INT8_SQ_MAX_M": "2", "EXL3_MGEMM_N_THRESHOLD": "0"} | 67.02 | 64.63 | 65.79 | 65.81 | 15.20 | 61.33 | 16.31 |
| w13s2-mtp-28k | 1.6.5+w13 {"EXL3_FD75D_LIVE": "0", "EXL3_GEMV_SK": "2", "EXL3_INT8_SQ_MAX_M": "2", "EXL3_MGEMM_N_THRESHOLD": "0"} | 72.67 | 72.37 | 69.58 | 71.54 | 13.98 | 66.57 | 15.02 |
| raf-mtp-28k | 1.5.3+rafatxf {} | 66.79 | 67.83 | 68.87 | 67.83 | 14.74 | 63.09 | 15.85 |

**w13-0 against adopt-0** (tok/s change per round):

round 1: +4.5%, round 2: +5.0%, round 3: +4.3%; faster in 3 of 3 rounds; mean +4.6%
pooled: round 1: +4.5%, round 2: +5.0%, round 3: +4.3%; faster in 3 of 3 rounds; all rounds +4.6%

**w13-28k against adopt-28k** (tok/s change per round):

round 1: +4.8%, round 2: +4.9%, round 3: +4.7%; faster in 3 of 3 rounds; mean +4.8%
pooled: round 1: +4.6%, round 2: +4.9%, round 3: +4.7%; faster in 3 of 3 rounds; all rounds +4.7%

**w13s1-mtp-0 against adopt-mtp-0** (tok/s change per round):

round 1: +23.0%, round 2: +22.9%, round 3: +16.4%; faster in 3 of 3 rounds; mean +20.8%
pooled: round 1: +22.8%, round 2: +21.0%, round 3: +16.1%; faster in 3 of 3 rounds; all rounds +20.0%

**w13s1-mtp-28k against adopt-mtp-28k** (tok/s change per round):

round 1: +23.8%, round 2: +18.0%, round 3: +21.2%; faster in 3 of 3 rounds; mean +21.0%
pooled: round 1: +22.6%, round 2: +19.9%, round 3: +21.8%; faster in 3 of 3 rounds; all rounds +21.4%

**w13s1-mtp-0 against raf-mtp-0** (tok/s change per round):

round 1: -0.5%, round 2: -3.6%, round 3: -1.1%; faster in 0 of 3 rounds; mean -1.7%
pooled: round 1: -0.8%, round 2: -1.3%, round 3: -1.1%; faster in 0 of 3 rounds; all rounds -1.0%

**w13s2-mtp-0 against raf-mtp-0** (tok/s change per round):

round 1: +11.7%, round 2: +3.2%, round 3: +9.1%; faster in 3 of 3 rounds; mean +8.0%
pooled: round 1: +11.7%, round 2: +5.2%, round 3: +7.6%; faster in 3 of 3 rounds; all rounds +8.2%

**w13s1-mtp-28k against raf-mtp-28k** (tok/s change per round):

round 1: +0.3%, round 2: -4.7%, round 3: -4.5%; faster in 1 of 3 rounds; mean -3.0%
pooled: round 1: +2.1%, round 2: -5.3%, round 3: -5.2%; faster in 1 of 3 rounds; all rounds -2.8%

**w13s2-mtp-28k against raf-mtp-28k** (tok/s change per round):

round 1: +8.8%, round 2: +6.7%, round 3: +1.0%; faster in 3 of 3 rounds; mean +5.5%
pooled: round 1: +9.2%, round 2: +6.5%, round 3: +1.0%; faster in 3 of 3 rounds; all rounds +5.5%

**w13s2-mtp-0 against w13s1-mtp-0** (tok/s change per round):

round 1: +12.3%, round 2: +7.0%, round 3: +10.3%; faster in 3 of 3 rounds; mean +9.9%
pooled: round 1: +12.6%, round 2: +6.6%, round 3: +8.8%; faster in 3 of 3 rounds; all rounds +9.3%

**w13s2-mtp-28k against w13s1-mtp-28k** (tok/s change per round):

round 1: +8.4%, round 2: +12.0%, round 3: +5.8%; faster in 3 of 3 rounds; mean +8.7%
pooled: round 1: +7.0%, round 2: +12.5%, round 3: +6.5%; faster in 3 of 3 rounds; all rounds +8.5%

**adopt-0 against stock-0** (tok/s change per round):

round 1: +78.0%, round 2: +79.6%, round 3: +80.6%; faster in 3 of 3 rounds; mean +79.4%
pooled: round 1: +78.0%, round 2: +79.7%, round 3: +80.6%; faster in 3 of 3 rounds; all rounds +79.4%

**w13-0 against stock-0** (tok/s change per round):

round 1: +86.1%, round 2: +88.5%, round 3: +88.3%; faster in 3 of 3 rounds; mean +87.6%
pooled: round 1: +86.1%, round 2: +88.6%, round 3: +88.3%; faster in 3 of 3 rounds; all rounds +87.6%

**adopt-mtp-0 against stock-mtp-0** (tok/s change per round):

round 1: +14.8%, round 2: +14.4%, round 3: +14.3%; faster in 3 of 3 rounds; mean +14.5%
pooled: round 1: +16.9%, round 2: +19.6%, round 3: +16.7%; faster in 3 of 3 rounds; all rounds +17.7%

**w13s1-mtp-0 against stock-mtp-0** (tok/s change per round):

round 1: +41.2%, round 2: +40.5%, round 3: +33.1%; faster in 3 of 3 rounds; mean +38.3%
pooled: round 1: +43.5%, round 2: +44.7%, round 3: +35.5%; faster in 3 of 3 rounds; all rounds +41.2%

**w13s2-mtp-0 against stock-mtp-0** (tok/s change per round):

round 1: +58.6%, round 2: +50.3%, round 3: +46.9%; faster in 3 of 3 rounds; mean +51.9%
pooled: round 1: +61.7%, round 2: +54.3%, round 3: +47.4%; faster in 3 of 3 rounds; all rounds +54.3%

**w13s1-mtp-0 against w13-0** (tok/s change per round):

round 1: +48.1%, round 2: +48.9%, round 3: +45.3%; faster in 3 of 3 rounds; mean +47.4%
pooled: round 1: +37.8%, round 2: +39.7%, round 3: +36.0%; faster in 3 of 3 rounds; all rounds +37.8%

**w13s1-mtp-28k against w13-28k** (tok/s change per round):

round 1: +52.4%, round 2: +47.0%, round 3: +50.2%; faster in 3 of 3 rounds; mean +49.9%
pooled: round 1: +43.6%, round 2: +37.4%, round 3: +38.2%; faster in 3 of 3 rounds; all rounds +39.8%
