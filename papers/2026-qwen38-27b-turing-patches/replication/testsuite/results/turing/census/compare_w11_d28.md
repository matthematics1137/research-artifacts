# Census arms vs d28-f1-mgemmn0 (nodraft)

| arm | code | math | prose | mean ms/token | change |
|---|---|---|---|---|---|
| d28-f1-mgemmn0 | 32.78 | 32.73 | 32.74 | 32.75 | +0.0% |
| d28-w11-mgemmn0 | 24.82 | 26.16 | 25.06 | 25.35 | -22.6% |

Launch shapes, us per decoded token (launches per token):

| family | grid | block | d28-f1-mgemmn0 | d28-w11-mgemmn0 |
|---|---|---|---|---|
| `exl3_gemv_int8_sq_kernel` | 272x1x1 | 256x1x1 | 18162 (399.43) | 19533 (399.22) |
| `_paged_attn_decode_split_kernel` | 4x34x1 | 128x1x1 | 11140 (15.94) | - |
| `cuda_recurrent_gated_delta_rule_kernel_128` | 1x48x4 | 128x4x1 | 612 (47.81) | 654 (47.79) |
| `rms_norm_kernel` | 1x1x1 | 1024x1x1 | 542 (128.5) | 585 (128.43) |
| `_paged_attn_decode_combine_kernel` | 4x8x1 | 128x1x1 | 211 (15.94) | 222 (15.93) |
| `gdn_ba_gemv_rows_kernel` | 96x1x1 | 256x1x1 | 195 (47.81) | 213 (47.79) |
| `conv1d_update_kernel` | 40x1x1 | 256x1x1 | 179 (47.81) | 195 (47.79) |
| `gated_rms_norm_kernel` | 48x1x1 | 32x1x1 | 122 (47.81) | 133 (47.79) |
| `act_mul_kernel_h` | 34x1x1 | 256x1x1 | 118 (63.75) | 127 (63.72) |
| `gated_delta_net_fused_op_3_kernel` | 41x1x1 | 256x1x1 | 98 (47.81) | 102 (47.79) |
| `fd75d_split_kernel` | 4x34x1 | 128x1x1 | - | 2149 (15.93) |
