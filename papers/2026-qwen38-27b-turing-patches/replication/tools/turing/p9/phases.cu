// P9 phase 1, M2 (research/prereg/2026-11-05-p9-phase1.md): the one-row int8 sq GEMV (exl3_gemv_int8_sq_kernel at
// m = 1, fp16 output, no residual) with switches for its parts, compiled against a build's own patched source tree, so
// the time each part adds to a call's critical path can be read from intercept differences. Parts (template P, flags):
//   1  stage     each slice's input Hadamard transform, maximum and int8 quantization (gemv_int8_stage_slice)
//   2  loop      the per-weight main loop over a slice's rows for one 256-column group (the unit)
//   4  epilogue  fixed-order combine over slices, output Hadamard and scale (gemv_int8_epilogue_group_sq)
// The completion counter that elects each column group's epilogue block runs in every variant. P = 7 is the engine's
// kernel and must reproduce ext.exl3_gemm's output bit for bit. Without the stage the loop reads whatever splats
// shared memory holds, and without the loop the epilogue combines stale partials: the work, and so the time, is the
// same. P = 16 returns at once, with the same grid and shared memory (the null kernel).
// Built by p9_phases.py with the patched exllamav3 source tree's exllamav3_ext directory on the include path.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "quant/exl3_gemv_int8_kernel.cuh"

// The full kernel (P = 7) keeps the engine's launch bounds exactly, so its registers and code are the engine's. Every
// other variant is held to the full kernel's blocks per SM (PH_MINB, which p9_phases.py reads from a first build), so
// that removing a part cannot lower residency, and so add a wave, at the engine's grid
#ifndef PH_MINB
#define PH_MINB 1
#endif

#define PH_ARGS const half* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C, const int size_k, \
                const int size_n, int* __restrict__ locks, const half* __restrict__ suh, const half* __restrict__ svh, \
                const int rows_per

template <int bits, int P>
__device__ __forceinline__ void phases_body(PH_ARGS)
{
    if constexpr (P & 16) return;
    constexpr int M = 1;
    extern __shared__ uint32_t shmem[];

    // As exl3_gemv_int8_sq_kernel, with the slice height always given by the host
    int rows_total = size_k >> 4;
    int nb256_total = size_n / 256;
    int ksplit = CEIL_DIVIDE(rows_total, rows_per);
    int units = nb256_total * ksplit;
    int slice_stride = rows_per * 16;
    int pstride = size_n;

    int* counters = locks;
    float* qsums = (float*) (locks + SQ_COUNTERS_CAP);
    int* partials = locks + SQ_WS_RESERVED;

    half* sh_ah = (half*) shmem;
    uint32_t* sh_as = shmem + rows_per * 8;
    uint32_t* sh_b = sh_as + slice_stride * M;
    float* sh_tmp = (float*) (sh_b + gemv_int8_stage_bytes(bits, false) / 4);
    __shared__ float sh_red[33];
    __shared__ int sh_last;

    int t = threadIdx.x;
    int prev_slice = -1;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x)
    {
        int slice = unit / nb256_total;
        int nb256 = unit % nb256_total;
        int kb0 = slice * rows_per;
        int nrows = MIN(rows_per, rows_total - kb0);
        if constexpr (P & 1)
        {
            if (slice != prev_slice)
            {
                gemv_int8_stage_slice<M, false>(A, 1, size_k, suh, qsums + 4 * slice * M,
                                                sh_ah, sh_as, slice_stride, sh_red, kb0, nrows);
                prev_slice = slice;
            }
        }
        int* pacc = partials + (size_t) slice * M * pstride;
        if constexpr (P & 2)
        {
            if constexpr (bits == 4)
                gemv_int8_unit_wide<M, false, false>(B, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);
            else if constexpr (gemv_int8_stage_smem(bits, false))
                gemv_int8_unit_smem<bits, M, false, false, false>(B, pacc, pstride, sh_as, slice_stride, sh_b, nb256, kb0, nrows, size_n);
            else
                gemv_int8_unit_narrow<bits, M, false, false, false>(B, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);
        }

        __threadfence();
        __syncthreads();
        if (t == 0) sh_last = (atomicAdd(&counters[nb256], 1) == ksplit - 1) ? 1 : 0;
        __syncthreads();
        if (sh_last)
        {
            if constexpr (P & 4)
                gemv_int8_epilogue_group_sq<M, false, false>(partials, qsums, pstride, ksplit, 1,
                                                             C, svh, sh_tmp, nb256, size_n);
            if (t == 0) counters[nb256] = 0;
        }
    }
}

template <int bits>
__global__ __launch_bounds__(NUM_THREADS)
void phases_full(PH_ARGS)
{
    phases_body<bits, 7>(A, B, C, size_k, size_n, locks, suh, svh, rows_per);
}

template <int bits, int P>
__global__ __launch_bounds__(NUM_THREADS, PH_MINB)
void phases_kernel(PH_ARGS)
{
    phases_body<bits, P>(A, B, C, size_k, size_n, locks, suh, svh, rows_per);
}

#define PH_VARIANTS(b) \
    case 0: return (const void*) phases_kernel<b, 0>;  case 1: return (const void*) phases_kernel<b, 1>; \
    case 2: return (const void*) phases_kernel<b, 2>;  case 3: return (const void*) phases_kernel<b, 3>; \
    case 4: return (const void*) phases_kernel<b, 4>;  case 5: return (const void*) phases_kernel<b, 5>; \
    case 6: return (const void*) phases_kernel<b, 6>;  case 7: return (const void*) phases_full<b>; \
    case 16: return (const void*) phases_kernel<b, 16>;

static const void* kernel_for(int bits, int v)
{
    if (bits == 1) switch (v) { PH_VARIANTS(1) }
    if (bits == 2) switch (v) { PH_VARIANTS(2) }
    TORCH_CHECK(false, "bits 1-2, variants 0-7 and 16");
    return nullptr;
}

// The shared memory of a launch, as exl3_gemv_int8_sq's smem_for at m = 1 without residual
static size_t smem_for(int bits, int rows_per)
{
    return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 + (size_t) gemv_int8_stage_bytes(bits, false)
           + (size_t) 2 * 128 * 4;
}

// Registers, static shared memory, resident blocks per SM at a slice height, dynamic and opt-in shared memory, local
// (spilled) bytes per thread
std::vector<int64_t> info(int bits, int v, int rows_per)
{
    const void* fn = kernel_for(bits, v);
    int optin = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes attr;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, fn));
    C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int) attr.sharedSizeBytes));
    C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, NUM_THREADS, smem_for(bits, rows_per)));
    return {attr.numRegs, (int64_t) attr.sharedSizeBytes, blocks, (int64_t) smem_for(bits, rows_per), optin,
            (int64_t) attr.localSizeBytes};
}

void launch(int bits, int v, const at::Tensor& A, const at::Tensor& B, at::Tensor& C, at::Tensor& locks,
            const at::Tensor& suh, const at::Tensor& svh, int size_k, int size_n, int rows_per, int grid)
{
    const void* fn = kernel_for(bits, v);
    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = C.data_ptr();
    int* locks_ptr = (int*) locks.data_ptr();
    const half* suh_ptr = (const half*) suh.data_ptr();
    const half* svh_ptr = (const half*) svh.data_ptr();
    void* args[] = { (void*) &A_ptr, (void*) &B_ptr, (void*) &C_ptr, (void*) &size_k, (void*) &size_n,
                     (void*) &locks_ptr, (void*) &suh_ptr, (void*) &svh_ptr, (void*) &rows_per };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaLaunchKernel(fn, dim3(grid), dim3(NUM_THREADS), args, smem_for(bits, rows_per), stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("info", &info, "registers, static smem, blocks per SM, dynamic smem, opt-in limit, local bytes");
    m.def("launch", &launch, "one launch of variant v");
    m.def("ws_reserved", [] () { return (int64_t) SQ_WS_RESERVED; });
    m.def("rows_max", [] () { return (int64_t) gemv_int8_sq_rows_max(1, false); });
}
