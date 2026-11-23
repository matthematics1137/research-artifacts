# Credits for the lab's Turing patches and kernels

Who each lab patch borrows from or sits beside, kept in one place. The patch files themselves stay as they were built,
because their hashes are recorded in the P8 note. The full record is `research/TURING_W7_INT8_GEMV_2026-10-30.md` and
`research/TURING_RELATED_WORK_2026-10-31.md`.

**Publication hold (Matt, 2026-10-30).** None of these patches is pushed to a public fork, proposed upstream as a pull
request or issue, or otherwise published until the paper that reports it is out. The lab's private repository holds
them until then.

## Borrowed

- **W13, split-K GEMV for verify rows.** Ported from rafatxf's exllamav3 `turing` branch at commit `15654d3` (MIT
  licence), with credit in the patch header. The archived tarball and the patch hash are recorded in the P8 note
  (R1 "Source" and the W13 amendment).

## Related work found after the fact (cited, not borrowed)

- **W11 / W11b, fd75d decode attention for exllamav3's 8-bit cache.** None of these was known when W11 was designed:
  - LMDeploy TurboMind (2024): sm_75 decode over int8/int4 KV with the same S^T orientation and `movmatrix`;
  - rafatxf's fdq4 (2026-09-27): sm_75 tensor cores, register-direct from exllamav3's 4-bit cache;
  - kyojin's qdec (2026-10-23): sm_80+, split-phase drop-in over exllamav3's caches;
  - cookjohn/vllm-turing `fdec.cu` (2026-10-23): sm_75, FP8 KV, Qwen3.8-27B.
- **W7, W10b, the int8 GEMV fit on 64 KB parts.** It was run earlier by vcruz305/exl3-turing (2026-09-02),
  weicj/exllamav3-turing (2026-09-06) and spiritbuun/buun-llama-cpp (2026-09-06 to 09-18). vcruz305 very likely ran
  the two-row int8 verify path first.

Licences the sources record are MIT for exllamav3, rafatxf's branch, weicj, ik_llama.cpp and llama.cpp. vcruz305 has
none. The licences of buun-llama-cpp, kyojin's qdec, NInfer, cookjohn's vllm-turing and FBGEMM are not recorded in
the lab's notes, so check them before borrowing anything from those projects.
