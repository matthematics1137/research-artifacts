#pragma once

// fd75d: flash-decoding split kernel for Turing (sm_75) over the 8-bit quantized paged cache, on HMMA tensor
// cores. Drop-in for phase 1 of triton_paged._paged_attn_decode_split_kernel (QCK = QCV = 8, head_dim 256,
// no window / softcap / sinks): same grid (programs, splits_cap), same dense (query, head) row packing, same
// partial_o / partial_ml layout, so the Triton combine kernel finishes the reduction unchanged.
// Local_llms Paper 8, W11.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Kernel parameter indices that change between graph replays (Graph::record_param)
#define FD75D_P_BLOCK_TABLE 3
#define FD75D_P_SEQLENS 4
#define FD75D_P_SPLIT_LEN 9
#define FD75D_P_NUM_PAGES 10
#define FD75D_P_NUM_SPLITS 11

// The kernel's handle, for Graph::record_param
void* fd75d_split_fn();

// Launch phase 1. q (bsz, q_len, n_q_heads, 256) fp16; qk / qv packed 8-bit cache words (pages, page_size,
// n_kv_heads * 64); sk / sv fp16 scales (pages, page_size, n_kv_heads * 8). Returns the launch status.
cudaError_t fd75d_split_launch
(
    const half* q, const uint32_t* qk, const uint32_t* qv, const int* block_table, const int* cache_seqlens,
    float* partial_o, float* partial_ml, const half* sk, const half* sv,
    int split_len, int num_pages_per_seq, int num_splits,
    int bsz, int q_len, int n_q_heads, int n_kv_heads, int page_size, float scale, bool causal,
    int kv_append_len, int splits_cap, bool live, cudaStream_t stream
);

// The Triton decode row layout (decode_row_layout): rows per program and row blocks per kv head
void fd75d_row_layout(int q_len, int group_size, int* block_rows, int* h_blocks);
