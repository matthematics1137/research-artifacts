// Paper 8 W12b limiter study: the int8 sq GEMV's per-weight loop alone, K = 2, one activation row, on sm_75.
//
// Each block runs (k-slice x 256-column) units like exl3_gemv_int8_sq_kernel, but the slice's splats are copied from
// global memory instead of Hadamard-transformed and quantized, and there is no epilogue: the per-slice int32 partials
// are the output. So the time is the per-weight work plus a small per-unit cost. Variants (template V):
//   0  upstream narrow unit (gemv_int8_unit_narrow<2>, the one K = 2 runs in the engine)
//   1  upstream smem-staged unit (gemv_int8_unit_smem<2>, cp.async 4 rows deep; upstream routes K = 3/5/7 there)
//   2  narrow, PRMT window extraction (below)
//   3  staged, PRMT window extraction
//   4  narrow, no global loads: the two words per tile come from a hash of (row, lane) - compute only
//   5  narrow, dp4a replaced by integer adds of the hashed windows - FMA-pipe relief bound
//   6  narrow, no mul1 hash multiply - FMA-pipe relief bound
//   7  loads only: each warp streams its block pair's rows with 16-byte loads - the memory floor for this layout
//   8  narrow, running pointers: lane word offsets hoisted, two pointers step one row (added before any timing)
//   9  running pointers + PRMT extraction
//   10, 11  running pointers + register prefetch 1 or 2 rows ahead (sqmb2)
//   12  prefetch 2 rows ahead + PRMT
//   13  one coalesced word per lane + 4 warp shuffles per row (W12e), running pointer
//   14, 15  W12e + register prefetch 1 or 2 rows ahead
//   16  upstream narrow unit with a stepped row pointer only (W12b-ptr scouting: the minimal engine form)
// 0-3 and 8-16 must give bit-identical partials. 4-7 compute other things and are timed only.
// Built by w12_sqmb.py with the patched exllamav3 source tree on the include path.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "quant/exl3_gemv_int8_kernel.cuh"

// Byte permute with zero fill: selector nibbles pick source bytes 0-3 of a, nibble 4 is a zero byte
template <uint32_t SEL>
__device__ __forceinline__ uint32_t prmt0(uint32_t a)
{
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(0u), "n"(SEL));
    return d;
}

enum { X_UP = 0, X_PRMT = 1, X_NOLOAD = 2, X_NODP4A = 3, X_NOHASH = 4 };

// The eight 16-bit trellis windows of lane t0's 8 weights in one 16x16 tile, K = 2 (as ext8w<2>). After the funnel
// shift the windows sit at bit offsets 14, 12, ..., 0 of b (w0 at 14, w7 at 0). The PRMT form takes b, b >> 2,
// b >> 4 and b >> 6 and cuts each at bytes 0-1 and 1-2: 3 shifts + 8 byte permutes, where upstream spends a mask
// and seven bit-field extracts (each a shift plus a zero-extend on sm_75).
// From the tile's two words a (word i0) and b (word i1)
template <int X>
__device__ __forceinline__ void mb_words(uint32_t a, uint32_t b, int t0, uint32_t (&w)[8])
{
    b = fshift(b, a, ((~t0) & 8) << 1);
    if constexpr (X == X_PRMT)
    {
        uint32_t b2 = b >> 2, b4 = b >> 4, b6 = b >> 6;
        w[7] = prmt0<0x4410>(b);  w[3] = prmt0<0x4421>(b);
        w[6] = prmt0<0x4410>(b2); w[2] = prmt0<0x4421>(b2);
        w[5] = prmt0<0x4410>(b4); w[1] = prmt0<0x4421>(b4);
        w[4] = prmt0<0x4410>(b6); w[0] = prmt0<0x4421>(b6);
    }
    else
    {
        w[7] = b & 0xffff;
        BFE16_IMM(w[6], b, 2);
        BFE16_IMM(w[5], b, 4);
        BFE16_IMM(w[4], b, 6);
        BFE16_IMM(w[3], b, 8);
        BFE16_IMM(w[2], b, 10);
        BFE16_IMM(w[1], b, 12);
        BFE16_IMM(w[0], b, 14);
    }
}

// Upstream's form: word indices from t0, loads through the tile pointer (or, X_NOLOAD, words from a register hash)
template <int X>
__device__ __forceinline__ void mb_ext(const uint32_t* ptr, int t0, uint32_t salt, uint32_t (&w)[8])
{
    uint32_t a, b;
    if constexpr (X == X_NOLOAD)
    {
        a = (salt * 2654435761u) ^ (uint32_t) t0;
        b = a * 747796405u + 2891336453u;
    }
    else
    {
        uint32_t i1 = t0 >> 4;
        uint32_t i0 = (i1 + 15) & 15;
        a = ptr[i0];
        b = ptr[i1];
    }
    mb_words<X>(a, b, t0, w);
}

// One tile's 8 weights (windows already cut) into the lane's two accumulators (n = g and g + 8)
template <int X>
__device__ __forceinline__ void mb_dot(uint32_t (&w)[8], uint2 as01, uint2 as89, int& acc0, int& acc1)
{
    if constexpr (X != X_NOHASH)
    {
        #pragma unroll
        for (int i = 0; i < 8; ++i) w[i] *= 0x83DCD12Du;
    }
    if constexpr (X == X_NODP4A)
    {
        acc0 += (int) (w[0] + w[1] + w[2] + w[3]);
        acc1 += (int) (w[4] + w[5] + w[6] + w[7]);
    }
    else
    {
        acc0 = dp4a_us(w[0], as01.x, acc0);
        acc0 = dp4a_us(w[1], as01.y, acc0);
        acc0 = dp4a_us(w[2], as89.x, acc0);
        acc0 = dp4a_us(w[3], as89.y, acc0);
        acc1 = dp4a_us(w[4], as01.x, acc1);
        acc1 = dp4a_us(w[5], as01.y, acc1);
        acc1 = dp4a_us(w[6], as89.x, acc1);
        acc1 = dp4a_us(w[7], as89.y, acc1);
    }
}

// One tile's 8 weights into the lane's two accumulators (n = g and g + 8), as gemv_int8_pair_row does per tile
template <int X>
__device__ __forceinline__ void mb_tile(const uint32_t* blk, int t0, uint32_t salt, uint2 as01, uint2 as89, int& acc0, int& acc1)
{
    uint32_t w[8];
    mb_ext<X>(blk, t0, salt, w);
    mb_dot<X>(w, as01, as89, acc0, acc1);
}

// Running pointers (V8, V9): the lane's two word offsets are constant, so two pointers step one row per iteration
// and tile B sits 16 words (64 bytes) further, an immediate offset. Upstream recomputes the 64-bit tile address
// from kb every row (~15 instructions per row in the V0 machine code, for 16 weights per lane).
template <int X>
__device__ __forceinline__ void mb_unit_ptr(const uint16_t* __restrict__ B, int* __restrict__ accs, const uint32_t* __restrict__ sh_as,
                                            int nb256, int kb0, int nrows, int size_n)
{
    constexpr int TWORDS = 16;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n / 16 * TWORDS;
    int t0 = lane << 3;
    uint32_t i1 = t0 >> 4;
    uint32_t i0 = (i1 + 15) & 15;
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * (2 * TWORDS);
    const uint32_t* p0 = bp + i0;
    const uint32_t* p1 = bp + i1;
    const uint32_t* as = sh_as + 2 * (lane & 3);
    int ia0[1] = {}, ia1[1] = {}, ib0[1] = {}, ib1[1] = {};
    for (int kb = 0; kb < nrows; ++kb)
    {
        uint32_t aA = p0[0], bA = p1[0], aB = p0[TWORDS], bB = p1[TWORDS];
        p0 += row_stride;
        p1 += row_stride;
        uint2 as01 = *(const uint2*) (as);
        uint2 as89 = *(const uint2*) (as + 8);
        as += 16;
        uint32_t w[8];
        mb_words<X>(aA, bA, t0, w);
        mb_dot<X>(w, as01, as89, ia0[0], ia1[0]);
        mb_words<X>(aB, bB, t0, w);
        mb_dot<X>(w, as01, as89, ib0[0], ib1[0]);
    }
    gemv_int8_pair_tail<1, false, false>(accs, size_n, nbp, lane, size_n, ia0, ia1, ib0, ib1, ia0, ia1, ib0, ib1);
}

// Register prefetch (V10-V12, sqmb2): running pointers plus a ring of PF rows' words already in flight, so each row's
// global loads are issued PF rows before its arithmetic. sqmb1 found loads alone and arithmetic alone each near the
// memory floor and the combination ~40% slower: the loop does not overlap them. Loads past the slice are guarded.
template <int X, int PF>
__device__ __forceinline__ void mb_unit_pf(const uint16_t* __restrict__ B, int* __restrict__ accs, const uint32_t* __restrict__ sh_as,
                                           int nb256, int kb0, int nrows, int size_n)
{
    constexpr int TWORDS = 16;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n / 16 * TWORDS;
    int t0 = lane << 3;
    uint32_t i1 = t0 >> 4;
    uint32_t i0 = (i1 + 15) & 15;
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * (2 * TWORDS);
    const uint32_t* p0 = bp + i0;
    const uint32_t* p1 = bp + i1;
    const uint32_t* as = sh_as + 2 * (lane & 3);
    uint32_t q[PF + 1][4];
    #pragma unroll
    for (int j = 0; j < PF; ++j)
    {
        if (j < nrows) { q[j][0] = p0[0]; q[j][1] = p1[0]; q[j][2] = p0[TWORDS]; q[j][3] = p1[TWORDS]; }
        p0 += row_stride;
        p1 += row_stride;
    }
    int ia0[1] = {}, ia1[1] = {}, ib0[1] = {}, ib1[1] = {};
    #pragma unroll 2
    for (int kb = 0; kb < nrows; ++kb)
    {
        if (kb + PF < nrows) { q[PF][0] = p0[0]; q[PF][1] = p1[0]; q[PF][2] = p0[TWORDS]; q[PF][3] = p1[TWORDS]; }
        p0 += row_stride;
        p1 += row_stride;
        uint2 as01 = *(const uint2*) (as);
        uint2 as89 = *(const uint2*) (as + 8);
        as += 16;
        uint32_t w[8];
        mb_words<X>(q[0][0], q[0][1], t0, w);
        mb_dot<X>(w, as01, as89, ia0[0], ia1[0]);
        mb_words<X>(q[0][2], q[0][3], t0, w);
        mb_dot<X>(w, as01, as89, ib0[0], ib1[0]);
        #pragma unroll
        for (int j = 0; j < PF; ++j)
        {
            q[j][0] = q[j + 1][0]; q[j][1] = q[j + 1][1]; q[j][2] = q[j + 1][2]; q[j][3] = q[j + 1][3];
        }
    }
    gemv_int8_pair_tail<1, false, false>(accs, size_n, nbp, lane, size_n, ia0, ia1, ib0, ib1, ia0, ia1, ib0, ib1);
}

// One coalesced word per lane (W12e, V13-V15). In the narrow unit lane L reads words i0 and i1 of both tiles (i1 =
// L >> 1, i0 = i1 - 1 mod 16): 4 loads per row per lane, 128 lane-loads for the pair's 32 words, each word read by
// 4 lanes. sqmb2 found register prefetch does not close the gap between loads alone and arithmetic alone, so the
// suspect is the load instructions' issue cost, not their latency. Here lane L loads only word L of the tile pair
// (tile A words 0-15 in lanes 0-15, tile B in lanes 16-31: one 128-byte transaction, 1 load per lane per row) and
// takes its four words from the owning lanes with __shfl_sync. The words are the same, so the partials must equal
// V0's bit for bit. PF > 0 issues each row's load PF rows ahead (one register per row of ring).
template <int X, int PF>
__device__ __forceinline__ void mb_unit_shfl(const uint16_t* __restrict__ B, int* __restrict__ accs, const uint32_t* __restrict__ sh_as,
                                             int nb256, int kb0, int nrows, int size_n)
{
    constexpr int TWORDS = 16;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n / 16 * TWORDS;
    int t0 = lane << 3;
    int i1 = t0 >> 4;
    int i0 = (i1 + 15) & 15;
    const uint32_t* p = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * (2 * TWORDS) + lane;
    const uint32_t* as = sh_as + 2 * (lane & 3);
    uint32_t q[PF + 1];
    #pragma unroll
    for (int j = 0; j < PF; ++j)
    {
        if (j < nrows) q[j] = p[0];
        p += row_stride;
    }
    int ia0[1] = {}, ia1[1] = {}, ib0[1] = {}, ib1[1] = {};
    #pragma unroll 2
    for (int kb = 0; kb < nrows; ++kb)
    {
        if (kb + PF < nrows) q[PF] = p[0];
        p += row_stride;
        uint32_t v = q[0];
        uint32_t aA = __shfl_sync(0xffffffffu, v, i0);
        uint32_t bA = __shfl_sync(0xffffffffu, v, i1);
        uint32_t aB = __shfl_sync(0xffffffffu, v, i0 + 16);
        uint32_t bB = __shfl_sync(0xffffffffu, v, i1 + 16);
        uint2 as01 = *(const uint2*) (as);
        uint2 as89 = *(const uint2*) (as + 8);
        as += 16;
        uint32_t w[8];
        mb_words<X>(aA, bA, t0, w);
        mb_dot<X>(w, as01, as89, ia0[0], ia1[0]);
        mb_words<X>(aB, bB, t0, w);
        mb_dot<X>(w, as01, as89, ib0[0], ib1[0]);
        #pragma unroll
        for (int j = 0; j < PF; ++j) q[j] = q[j + 1];
    }
    gemv_int8_pair_tail<1, false, false>(accs, size_n, nbp, lane, size_n, ia0, ia1, ib0, ib1, ia0, ia1, ib0, ib1);
}

// The minimal engine form of running pointers (V16, W12b-ptr scouting): upstream's narrow unit unchanged except that
// the row pointer steps by row_stride instead of being recomputed from kb. The words still come through upstream's
// own extraction (word indices from t0), so a one-line engine patch would cover every K. V8 also hoists the lane's
// two word offsets into the pointers; V16 against V8 says whether that half is needed.
__device__ __forceinline__ void mb_unit_step(const uint16_t* __restrict__ B, int* __restrict__ accs, const uint32_t* __restrict__ sh_as,
                                             int nb256, int kb0, int nrows, int size_n)
{
    constexpr int TWORDS = 16;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n / 16 * TWORDS;
    const uint32_t* blockA = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * (2 * TWORDS);
    int c2 = 2 * (lane & 3);
    int ia0[1] = {}, ia1[1] = {}, ib0[1] = {}, ib1[1] = {};
    for (int kb = 0; kb < nrows; ++kb)
    {
        gemv_int8_pair_row<2, 1, false, false>(blockA, blockA + TWORDS, sh_as + (kb << 4), 0, c2, lane << 3,
                                               ia0, ia1, ib0, ib1, ia0, ia1, ib0, ib1);
        blockA += row_stride;
    }
    gemv_int8_pair_tail<1, false, false>(accs, size_n, nbp, lane, size_n, ia0, ia1, ib0, ib1, ia0, ia1, ib0, ib1);
}

// The narrow and staged units of exl3_gemv_int8_kernel.cuh (K = 2, M = 1, no residual, plain stores), with the
// tile work swapped for mb_tile<X>
template <int X, bool STAGED>
__device__ __forceinline__ void mb_unit(const uint16_t* __restrict__ B, int* __restrict__ accs, const uint32_t* __restrict__ sh_as,
                                        uint32_t* __restrict__ sh_b, int nb256, int kb0, int nrows, int size_n)
{
    constexpr int TWORDS = 16;                 // 8 * bits
    constexpr int pairwords = 2 * TWORDS;
    constexpr int D = GEMV_STAGE_D;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n / 16 * TWORDS;
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * pairwords;
    uint32_t* sb = sh_b + warp * (D * pairwords);
    auto stage_row = [&] (int kb)
    {
        if (kb < nrows && lane < pairwords / 4)
            cp_async(sb + (kb % D) * pairwords + lane * 4, bp + (size_t) kb * row_stride + lane * 4);
        cp_async_fence();
    };
    if constexpr (STAGED)
    {
        #pragma unroll
        for (int r = 0; r < D - 1; ++r) stage_row(r);
    }
    int c2 = 2 * (lane & 3);
    int t0 = lane << 3;
    int ia0[1] = {}, ia1[1] = {}, ib0[1] = {}, ib1[1] = {};
    for (int kb = 0; kb < nrows; ++kb)
    {
        const uint32_t* blockA;
        if constexpr (STAGED)
        {
            cp_async_wait<D - 2>();
            __syncwarp();
            stage_row(kb + D - 1);
            blockA = sb + (kb % D) * pairwords;
        }
        else
            blockA = bp + (size_t) kb * row_stride;
        const uint32_t* as = sh_as + (kb << 4);
        uint2 as01 = *(const uint2*) (as + c2);
        uint2 as89 = *(const uint2*) (as + c2 + 8);
        mb_tile<X>(blockA, t0, (uint32_t) (kb0 + kb) * 2, as01, as89, ia0[0], ia1[0]);
        mb_tile<X>(blockA + TWORDS, t0, (uint32_t) (kb0 + kb) * 2 + 1, as01, as89, ib0[0], ib1[0]);
    }
    gemv_int8_pair_tail<1, false, false>(accs, size_n, nbp, lane, size_n, ia0, ia1, ib0, ib1, ia0, ia1, ib0, ib1);
}

// Loads only: the same rows, 16 bytes per lane for lanes 0-7, folded into one word that is stored
__device__ __forceinline__ void mb_stream(const uint16_t* __restrict__ B, int* __restrict__ accs, int nb256, int kb0, int nrows, int size_n)
{
    constexpr int pairwords = 32;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n / 16 * 16;
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * pairwords;
    uint32_t x = 0;
    if (lane < 8)
        for (int kb = 0; kb < nrows; ++kb)
        {
            uint4 v = *(const uint4*) (bp + (size_t) kb * row_stride + lane * 4);
            x += v.x ^ v.y ^ v.z ^ v.w;
        }
    if (lane < 8) accs[nbp * 32 + lane] = (int) x;
}

template <int V>
__global__ __launch_bounds__(NUM_THREADS)
void sqmb_kernel(const uint16_t* __restrict__ B, const uint32_t* __restrict__ splats, int* __restrict__ partials,
                 int size_k, int size_n, int rows_per)
{
    extern __shared__ uint32_t shmem[];
    int rows_total = size_k >> 4;
    int nb256_total = size_n / 256;
    int ksplit = CEIL_DIVIDE(rows_total, rows_per);
    int units = nb256_total * ksplit;
    uint32_t* sh_as = shmem;
    uint32_t* sh_b = shmem + rows_per * 16;
    int prev_slice = -1;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x)
    {
        int slice = unit / nb256_total;
        int nb256 = unit % nb256_total;
        int kb0 = slice * rows_per;
        int nrows = MIN(rows_per, rows_total - kb0);
        if (slice != prev_slice && V != 7)
        {
            __syncthreads();
            for (int i = threadIdx.x; i < nrows * 16; i += NUM_THREADS) sh_as[i] = splats[kb0 * 16 + i];
            __syncthreads();
            prev_slice = slice;
        }
        int* pacc = partials + (size_t) slice * size_n;
        if constexpr (V == 0) gemv_int8_unit_narrow<2, 1, false, false>(B, pacc, size_n, sh_as, rows_per * 16, nb256, kb0, nrows, size_n);
        else if constexpr (V == 1) gemv_int8_unit_smem<2, 1, false, false>(B, pacc, size_n, sh_as, rows_per * 16, sh_b, nb256, kb0, nrows, size_n);
        else if constexpr (V == 2) mb_unit<X_PRMT, false>(B, pacc, sh_as, sh_b, nb256, kb0, nrows, size_n);
        else if constexpr (V == 3) mb_unit<X_PRMT, true>(B, pacc, sh_as, sh_b, nb256, kb0, nrows, size_n);
        else if constexpr (V == 4) mb_unit<X_NOLOAD, false>(B, pacc, sh_as, sh_b, nb256, kb0, nrows, size_n);
        else if constexpr (V == 5) mb_unit<X_NODP4A, false>(B, pacc, sh_as, sh_b, nb256, kb0, nrows, size_n);
        else if constexpr (V == 6) mb_unit<X_NOHASH, false>(B, pacc, sh_as, sh_b, nb256, kb0, nrows, size_n);
        else if constexpr (V == 7) mb_stream(B, pacc, nb256, kb0, nrows, size_n);
        else if constexpr (V == 8) mb_unit_ptr<X_UP>(B, pacc, sh_as, nb256, kb0, nrows, size_n);
        else if constexpr (V == 9) mb_unit_ptr<X_PRMT>(B, pacc, sh_as, nb256, kb0, nrows, size_n);
        else if constexpr (V == 10) mb_unit_pf<X_UP, 1>(B, pacc, sh_as, nb256, kb0, nrows, size_n);
        else if constexpr (V == 11) mb_unit_pf<X_UP, 2>(B, pacc, sh_as, nb256, kb0, nrows, size_n);
        else if constexpr (V == 12) mb_unit_pf<X_PRMT, 2>(B, pacc, sh_as, nb256, kb0, nrows, size_n);
        else if constexpr (V == 13) mb_unit_shfl<X_UP, 0>(B, pacc, sh_as, nb256, kb0, nrows, size_n);
        else if constexpr (V == 14) mb_unit_shfl<X_UP, 1>(B, pacc, sh_as, nb256, kb0, nrows, size_n);
        else if constexpr (V == 15) mb_unit_shfl<X_UP, 2>(B, pacc, sh_as, nb256, kb0, nrows, size_n);
        else mb_unit_step(B, pacc, sh_as, nb256, kb0, nrows, size_n);
    }
}

static const void* kernel_for(int v)
{
    switch (v)
    {
        case 0: return (const void*) sqmb_kernel<0>;
        case 1: return (const void*) sqmb_kernel<1>;
        case 2: return (const void*) sqmb_kernel<2>;
        case 3: return (const void*) sqmb_kernel<3>;
        case 4: return (const void*) sqmb_kernel<4>;
        case 5: return (const void*) sqmb_kernel<5>;
        case 6: return (const void*) sqmb_kernel<6>;
        case 7: return (const void*) sqmb_kernel<7>;
        case 8: return (const void*) sqmb_kernel<8>;
        case 9: return (const void*) sqmb_kernel<9>;
        case 10: return (const void*) sqmb_kernel<10>;
        case 11: return (const void*) sqmb_kernel<11>;
        case 12: return (const void*) sqmb_kernel<12>;
        case 13: return (const void*) sqmb_kernel<13>;
        case 14: return (const void*) sqmb_kernel<14>;
        case 15: return (const void*) sqmb_kernel<15>;
        case 16: return (const void*) sqmb_kernel<16>;
    }
    TORCH_CHECK(false, "variant 0-16");
    return nullptr;
}

static size_t smem_bytes(int v, int rows_per)
{
    bool staged = v == 1 || v == 3;
    return (size_t) rows_per * 16 * 4 + (staged ? (size_t) 8 * GEMV_STAGE_D * 32 * 4 : 0);
}

// (registers per thread, resident blocks per SM at this slice height)
std::vector<int64_t> info(int v, int rows_per)
{
    const void* fn = kernel_for(v);
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, fn);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, NUM_THREADS, smem_bytes(v, rows_per));
    return {attr.numRegs, blocks};
}

void launch(int v, at::Tensor B, at::Tensor splats, at::Tensor partials, int64_t size_k, int64_t size_n, int64_t rows_per, int64_t grid)
{
    TORCH_CHECK(size_k % 16 == 0 && size_n % 256 == 0, "k % 16, n % 256");
    const void* fn = kernel_for(v);
    const uint16_t* b = (const uint16_t*) B.data_ptr();
    const uint32_t* s = (const uint32_t*) splats.data_ptr();
    int* p = (int*) partials.data_ptr();
    int k = (int) size_k, n = (int) size_n, r = (int) rows_per;
    void* args[] = {(void*) &b, (void*) &s, (void*) &p, (void*) &k, (void*) &n, (void*) &r};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cuda_check(cudaLaunchKernel(fn, dim3((unsigned) grid), dim3(NUM_THREADS), args, smem_bytes(v, r), stream));
}

// ---------------------------------------------------------------------------------------------------------------------
// Several activation rows (MTP verify), K = 2: upstream's dp4a narrow unit at M rows against an integer tensor-core
// form. mma.m8n8k16.s8.u8 takes A (8 x 16, s8) and B (16 x 8, u8): lane (g, c) supplies A[g][4c..4c+3] and
// B[4c..4c+3][g]. The narrow unit's lane (g, c) already holds the hash words of weights (k = 2c, 2c+1, 2c+8, 2c+9;
// n = g) as w0..w3 and the same k at n = g + 8 as w4..w7, and each hash word's four bytes are four k entries whose
// sum is the codebook value. So w_i is the lane's B fragment as it stands, and the A fragment is row g's splat of
// the matching k (zero for g >= M). One instruction covers 4 weights x 8 columns x 8 rows. Its exact int32 sums
// equal the dp4a path's in any order, so the partials must match upstream bit for bit.

__device__ __forceinline__ void imma8816(int (&d)[2], uint32_t a, uint32_t b)
{
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(d[0]), "+r"(d[1]) : "r"(a), "r"(b));
}

template <int M>
__device__ __forceinline__ void mb_unit_imma(const uint16_t* __restrict__ B, int* __restrict__ accs, size_t acc_stride,
                                             const uint32_t* __restrict__ sh_as, int slice_stride,
                                             int nb256, int kb0, int nrows, int size_n)
{
    constexpr int TWORDS = 16;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int g = lane >> 2, c = lane & 3;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n / 16 * TWORDS;
    int t0 = lane << 3;
    uint32_t i1 = t0 >> 4;
    uint32_t i0 = (i1 + 15) & 15;
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * (2 * TWORDS);
    const uint32_t* p0 = bp + i0;
    const uint32_t* p1 = bp + i1;
    const bool live = g < M;
    const uint32_t* as = sh_as + (live ? g : 0) * slice_stride + 2 * c;
    int d[4][2] = {};
    for (int kb = 0; kb < nrows; ++kb)
    {
        uint32_t aA = p0[0], bA = p1[0], aB = p0[TWORDS], bB = p1[TWORDS];
        p0 += row_stride;
        p1 += row_stride;
        uint2 s01 = *(const uint2*) (as);
        uint2 s89 = *(const uint2*) (as + 8);
        as += 16;
        if (!live) s01 = s89 = make_uint2(0, 0);
        uint32_t w[8];
        mb_words<X_UP>(aA, bA, t0, w);
        #pragma unroll
        for (int i = 0; i < 8; ++i) w[i] *= 0x83DCD12Du;
        imma8816(d[0], s01.x, w[0]); imma8816(d[0], s01.y, w[1]); imma8816(d[0], s89.x, w[2]); imma8816(d[0], s89.y, w[3]);
        imma8816(d[1], s01.x, w[4]); imma8816(d[1], s01.y, w[5]); imma8816(d[1], s89.x, w[6]); imma8816(d[1], s89.y, w[7]);
        mb_words<X_UP>(aB, bB, t0, w);
        #pragma unroll
        for (int i = 0; i < 8; ++i) w[i] *= 0x83DCD12Du;
        imma8816(d[2], s01.x, w[0]); imma8816(d[2], s01.y, w[1]); imma8816(d[2], s89.x, w[2]); imma8816(d[2], s89.y, w[3]);
        imma8816(d[3], s01.x, w[4]); imma8816(d[3], s01.y, w[5]); imma8816(d[3], s89.x, w[6]); imma8816(d[3], s89.y, w[7]);
    }
    // D fragment: lane (g, c) holds row g, columns 2c and 2c + 1 of each 8-column group
    if (live)
    {
        int* acc = accs + g * acc_stride + nbp * 32 + 2 * c;
        acc[0] = d[0][0];  acc[1] = d[0][1];
        acc[8] = d[1][0];  acc[9] = d[1][1];
        acc[16] = d[2][0]; acc[17] = d[2][1];
        acc[24] = d[3][0]; acc[25] = d[3][1];
    }
}

// V 0: upstream narrow unit at M rows (dp4a, M per weight); V 1: the IMMA form. Partials are (ksplit, M, n).
template <int V, int M>
__global__ __launch_bounds__(NUM_THREADS)
void sqmb_rows_kernel(const uint16_t* __restrict__ B, const uint32_t* __restrict__ splats, int* __restrict__ partials,
                      int size_k, int size_n, int rows_per)
{
    extern __shared__ uint32_t shmem[];
    int rows_total = size_k >> 4;
    int nb256_total = size_n / 256;
    int ksplit = CEIL_DIVIDE(rows_total, rows_per);
    int units = nb256_total * ksplit;
    int slice_stride = rows_per * 16;
    uint32_t* sh_as = shmem;
    int prev_slice = -1;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x)
    {
        int slice = unit / nb256_total;
        int nb256 = unit % nb256_total;
        int kb0 = slice * rows_per;
        int nrows = MIN(rows_per, rows_total - kb0);
        if (slice != prev_slice)
        {
            __syncthreads();
            for (int r = 0; r < M; ++r)
                for (int i = threadIdx.x; i < nrows * 16; i += NUM_THREADS)
                    sh_as[r * slice_stride + i] = splats[(size_t) r * size_k + kb0 * 16 + i];
            __syncthreads();
            prev_slice = slice;
        }
        int* pacc = partials + (size_t) slice * M * size_n;
        if constexpr (V == 0) gemv_int8_unit_narrow<2, M, false, false>(B, pacc, size_n, sh_as, slice_stride, nb256, kb0, nrows, size_n);
        else mb_unit_imma<M>(B, pacc, size_n, sh_as, slice_stride, nb256, kb0, nrows, size_n);
    }
}

template <int M>
static const void* rows_kernel(int v) { return v ? (const void*) sqmb_rows_kernel<1, M> : (const void*) sqmb_rows_kernel<0, M>; }

static const void* rows_kernel_for(int v, int m)
{
    switch (m)
    {
        case 1: return rows_kernel<1>(v);
        case 2: return rows_kernel<2>(v);
        case 3: return rows_kernel<3>(v);
        case 4: return rows_kernel<4>(v);
        case 6: return rows_kernel<6>(v);
        case 8: return rows_kernel<8>(v);
    }
    TORCH_CHECK(false, "rows 1, 2, 3, 4, 6 or 8");
    return nullptr;
}

std::vector<int64_t> rows_info(int v, int m, int rows_per)
{
    const void* fn = rows_kernel_for(v, m);
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, fn);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, NUM_THREADS, (size_t) m * rows_per * 16 * 4);
    return {attr.numRegs, blocks};
}

void launch_rows(int v, int64_t m, at::Tensor B, at::Tensor splats, at::Tensor partials, int64_t size_k, int64_t size_n,
                 int64_t rows_per, int64_t grid)
{
    TORCH_CHECK(size_k % 16 == 0 && size_n % 256 == 0, "k % 16, n % 256");
    const void* fn = rows_kernel_for(v, (int) m);
    const uint16_t* b = (const uint16_t*) B.data_ptr();
    const uint32_t* s = (const uint32_t*) splats.data_ptr();
    int* p = (int*) partials.data_ptr();
    int k = (int) size_k, n = (int) size_n, r = (int) rows_per;
    size_t smem = (size_t) m * r * 16 * 4;
    if (smem > 48 * 1024) cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    void* args[] = {(void*) &b, (void*) &s, (void*) &p, (void*) &k, (void*) &n, (void*) &r};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cuda_check(cudaLaunchKernel(fn, dim3((unsigned) grid), dim3(NUM_THREADS), args, smem, stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("info", &info, "registers per thread and resident blocks per SM");
    m.def("launch", &launch, "one launch of variant v");
    m.def("rows_info", &rows_info, "registers and blocks per SM for the M-row kernels");
    m.def("launch_rows", &launch_rows, "one launch of the M-row kernel: v 0 upstream dp4a, 1 IMMA");
}
