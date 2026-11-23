# Test 2: bytes-only against the priced graph (2026-11-01 18:54:34 EST)

Pre-registration: `research/prereg/2026-11-01-four-resources-tests-1-3.md`, test 2, with Amendment 1. Predictions: `predictions.md`, written 2026-11-01 18:49:21 EST before any held-out number was read. Code: `tools/four_resources/test2_priced_graph.py`. CPU only; nothing was edited or deleted.

**Cross-build prediction (Amendment 1).** The 2080 Ti training census `d0-f1-mgemmn0` is build 1.6.5+w9. final2's adopted arms run w11b and its w13 arms run w13, so the priced graph, which knows only the w9 kernels, predicts those arms across builds. The held-out 28K census `d28-f1-mgemmn0` is w9, the same build.

Predictions files (unchanged since written): `predictions.md` sha256 991b18f606131ded, `predictions.json` sha256 de5eb3e040584856.

## What the comparison shows

- **Bytes do not price a 2-bit token.** Bytes-only misses every held-out 2-bit configuration, by -60% to -22%. Kill 1 is not triggered.
- **Across builds the priced graph holds on the 2080 Ti.** Built from the w9 census, it predicts final2's undrafted w11b and w13 arms within +3.1% to +9.2%; it overpredicts w13 most, whose kernels are faster than the w9 constants know.
- **The attention kernel, not the cache bytes, sets the 28K cost.** The w9 census at 28K measured 32.75 ms/token against 24.60 predicted (-24.9%): attention took 11.35 ms/token against 2.84 ms priced from bytes. The w11b and w13 builds pay only +1.86 and +1.75 ms at 28K, close to bytes-only's +1.63; the w9 censuses pay +9.78.
- **On the laptop the cache type reverses the byte ordering.** Per cache value read on the GPU, q8_0 costs 7.4, f16 10.6 and q4_0 12.1 µs per million values, so the smallest cache is the slowest. The priced graph gets the 8B's contrast right in sign (q4_0 slower); bytes-only gets it wrong.
- **The priced graph overprices the fully resident 8B** by +22.6%, +27.6%, +15.0%, +7.7% (primary). The fixed cost borrowed from the 2080 Ti census is too large for it; the declared zero-fixed-cost variant gives +15.9%, +21.8%, +6.4%, +0.7%.
- **CPU attention with an f16 cache is priced from one training cell.** Q4_K_XL with f16, eight attention layers on the CPU, is overpredicted by +48.7%; with q4_0, seven layers, by +3.5%.
- **Kill 2 is triggered** and the ±12% claim is refuted (details below).
- **Host RAM:** a MiB of weights on the host costs 0.0294 ms per token, 7.0x an effective VRAM-resident MiB (registered 0.021-0.026 ms and 4.5-5.5x).
- **Energy:** capped arms draw 96.0-98.8% of the cap, so energy is cap × time less 1-4%; 19 of 23 arms are within the registered ±3%.

Deviations and interpretations, all stated in `predictions.md` before comparing: drafted arms are left to test 3; equal clocks are assumed on the 2080 Ti; the laptop's host cache constant was split by type after the first structure missed a training cell by -20.2% (the first structure's predictions are reported too); the laptop's fixed cost is borrowed from the 2080 Ti census (a zero variant is reported); laptop speeds are pooled per decode step, the engine's own definition. The 80 W laptop census is deferred.

## Verdicts on the registered predictions and kill criteria

- **Kill 1: bytes-only predicts every held-out 2-bit configuration within ±15%**: not triggered. 0 of 7 2-bit configurations within ±15%; errors final2-adopt-0 -47.3%, final2-adopt-28k -44.6%, final2-w13-0 -44.9%, final2-w13-28k -42.0%, census-d28-f1-mgemmn0 -59.6%, qwen3-8b__Q2_K_XL__f16|longctx -22.1%, qwen3-8b__Q2_K_XL__q4_0|longctx -47.5%.
- **Kill 2: the priced graph misses 2 or more held-out configurations by more than 20%**: TRIGGERED. 4 of 11 miss by more than 20% (literal reading, every held-out configuration): census-d28-f1-mgemmn0 -24.9%, qwen38-27b__Q4_K_XL__f16|longctx +48.7%, qwen3-8b__Q2_K_XL__f16|longctx +22.6%, qwen3-8b__Q2_K_XL__q4_0|longctx +27.6%. Excluding cells with attention on the CPU: 3 of 9..
- **Priced graph within ±12%, median at most 8%, except cells with attention on the CPU**: refuted. max |error| 27.6%, median 9.2% over 9 configurations.
- **Bytes-only underpredicts 2080 Ti undrafted time by 37-45% (75-86 tok/s against 47.53 measured)**: holds. w13 at fill 0 (the arm named): bytes-only 86.3 tok/s against 47.53 measured, time underpredicted by 44.9%. The other undrafted held-out points: final2-adopt-0 47.3%, final2-adopt-28k 44.6%, final2-w13-28k 42.0%, census-d28-f1-mgemmn0 59.6%.
- **Bytes-only predicts llama.cpp IQ2_S 16-18% slower than IQ2_XXS (measured 7.7%)**: partly: outside the registered range. bytes-only 15.5% slower with the recurrent state and cache (16.2% on weights alone); measured 7.7% (llamabench_e1 tg128, depth 0). Bytes-only overstates the gap either way..
- **Bytes-only predicts +1.9 ms for the KV read at 32K (measured +8.75 ms)**: holds. bytes-only +1.908 ms; measured +8.75 ms (IQ2_XXS, llamabench_e1), +11.88 ms (IQ2_S).
- **Bytes-only gets the sign wrong for the laptop 8B's KV contrast at 16K (predicts q4_0 1.3-1.45x faster than f16; measured 0.88-0.98x)**: holds. bytes-only Q2_K_XL 1.447x, Q4_K_XL 1.313x; measured per step Q2_K_XL 0.976x, Q4_K_XL 0.884x (pooled per token, as P6 reports: Q2_K_XL 0.975x, Q4_K_XL 0.883x); priced graph Q2_K_XL 0.938x, Q4_K_XL 0.943x.
- **Bytes-only fits the EXL3 28K KV penalty (+1.5-1.9 ms against +1.73)**: holds. bytes-only +1.630 ms; measured final2 adopted +1.86 ms, final2 w13 +1.75 ms, w9 censuses +9.78 ms.
- **A host-RAM edge costs 0.021-0.026 ms per offloaded MiB, 4.5-5.5x a VRAM-resident MiB**: refuted. fitted 0.0294 ms per MiB; 6.99x the effective VRAM MiB, 9.44x the bandwidth floor (secondary fit: 0.0298 ms per MiB).
- **For capped arms, energy equals cap × time within ±3%**: refuted. 19 of 23 capped arms (cap flag in at least 95% of samples) within ±3%; ratio of time-weighted watts to the cap 0.960-0.988.

## Held-out configurations

| configuration | measured | priced graph | error | bytes-only | error | attention on CPU |
|---|---|---|---|---|---|---|
| final2-adopt-0 | 22.00 | 22.97 | +4.4% | 11.59 | -47.3% | no |
| final2-adopt-28k | 23.86 | 24.60 | +3.1% | 13.22 | -44.6% | no |
| final2-w13-0 | 21.04 | 22.97 | +9.2% | 11.59 | -44.9% | no |
| final2-w13-28k | 22.79 | 24.60 | +8.0% | 13.22 | -42.0% | no |
| census-d28-f1-mgemmn0 | 32.75 | 24.60 | -24.9% | 13.22 | -59.6% | no |
| qwen38-27b__Q4_K_XL__f16|longctx | 297.17 | 441.86 | +48.7% | 118.98 | -60.0% | yes |
| qwen38-27b__Q4_K_XL__q4_0|longctx | 237.91 | 246.36 | +3.5% | 109.55 | -54.0% | yes |
| qwen3-8b__Q2_K_XL__f16|longctx | 21.10 | 25.86 | +22.6% | 16.43 | -22.1% | no |
| qwen3-8b__Q2_K_XL__q4_0|longctx | 21.62 | 27.59 | +27.6% | 11.35 | -47.5% | no |
| qwen3-8b__Q4_K_XL__f16|longctx | 25.04 | 28.80 | +15.0% | 21.28 | -15.0% | no |
| qwen3-8b__Q4_K_XL__q4_0|longctx | 28.34 | 30.53 | +7.7% | 16.21 | -42.8% | no |

Times are ms per decoded token (2080 Ti) or per decode step (laptop). The laptop's secondary variant (fixed cost 0, declared before comparing) gives errors of qwen38-27b__Q4_K_XL__f16|longctx +48.9%, qwen38-27b__Q4_K_XL__q4_0|longctx +4.0%, qwen3-8b__Q2_K_XL__f16|longctx +15.9%, qwen3-8b__Q2_K_XL__q4_0|longctx +21.8%, qwen3-8b__Q4_K_XL__f16|longctx +6.4%, qwen3-8b__Q4_K_XL__q4_0|longctx +0.7%. The first structure (host cache cost independent of type), declared as an alternative: qwen38-27b__Q4_K_XL__f16|longctx +8.8%, qwen38-27b__Q4_K_XL__q4_0|longctx +25.6%, qwen3-8b__Q2_K_XL__f16|longctx +64.7%, qwen3-8b__Q2_K_XL__q4_0|longctx -2.7%, qwen3-8b__Q4_K_XL__f16|longctx +50.2%, qwen3-8b__Q4_K_XL__q4_0|longctx -15.6%.

## The 28K census, node by node

Predicted attention 2843 µs/token, measured 11351; predicted busy 23.439 ms/token, measured 31.573. Measured classes (µs/token): matrix products 18162.3, attention 11350.9, GDN recurrent state 612.0, small kernels 1444.7.

## Exploratory (not registered)

- **fp16 GDN state.** The registered bytes-only predictor prices the GDN state as fp32 (0.30 GB per token). exllamav3 1.6.2 and later keep it in fp16 by default, and the censuses and final2 ran at the default (`research/TURING_EXL3_RESEARCH_2026-10-30.md` line 138 and section 5.4), so the state is 0.15 GB. With fp16 state, bytes-only errors become final2-adopt-0 -48.5%, final2-adopt-28k -45.6%, final2-w13-0 -46.1%, final2-w13-28k -43.1%, census-d28-f1-mgemmn0 -60.4%. Kill 1 is unchanged. The 37-45% claim depends on it: w13 at fill 0 is underpredicted by 46.1% with fp16 state against 44.9% with fp32, so it holds as registered and fails with the state the engine actually stores. The 28K penalty and the laptop items are unaffected (llama.cpp's state is fp32).
- final2's SM-clock snapshots (median per arm): final2-adopt-0 1740 MHz, final2-adopt-28k 1755 MHz, final2-w13-0 1770 MHz, final2-w13-28k 1770 MHz. The census decode clock is not on record locally, so no clock adjustment is made.
- Laptop per-step against per-token pooling: P6's own speeds divide by all generated tokens, which counts the free first token; on 16K items with about 15 tokens each that understates the time per step by about 7%. The per-step figures above use the engine's definition.

