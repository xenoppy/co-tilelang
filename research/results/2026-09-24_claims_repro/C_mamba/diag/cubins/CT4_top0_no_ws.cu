#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void chunk_state_fwd_kernel(const half_t* __restrict__ B, half_t* __restrict__ Output, const half_t* __restrict__ dA_cumsum, const half_t* __restrict__ dt, const half_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) chunk_state_fwd_kernel(const half_t* __restrict__ B, half_t* __restrict__ Output, const half_t* __restrict__ dA_cumsum, const half_t* __restrict__ dt, const half_t* __restrict__ x) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* acc_o_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* dA_cumsum_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 512));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 1024));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 33792));
  float dA_cs_last[1];
  float acc_o[64];
  float dA_cumsum_local[16];
  float dt_local[16];
  float scale[16];
  half_t x_local[64];
  half_t xt_local[64];
  dA_cs_last[0] = ((float)dA_cumsum[(((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + 255)]);
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dA_cumsum_shared)[((int)threadIdx.x)] = dA_cumsum[(((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + ((int)threadIdx.x))];
  }
  tl::cp_async_commit();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dt_shared)[((int)threadIdx.x)] = dt[(((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + ((int)threadIdx.x))];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)x_shared)[(((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(x[(((((((((int)blockIdx.z) & 63) * 8388608) + ((((int)blockIdx.z) >> 6) * 1048576)) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((((int)blockIdx.z) & 63) * 262144) + ((((int)blockIdx.z) >> 6) * 32768)) + (i_2 * 1024)) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dA_cumsum_shared)[(((int)threadIdx.x) + 64)] = dA_cumsum[((((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + ((int)threadIdx.x)) + 64)];
  }
  tl::cp_async_commit();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dt_shared)[(((int)threadIdx.x) + 64)] = dt[((((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + ((int)threadIdx.x)) + 64)];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)x_shared)[((((((i_3 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(x[((((((((((int)blockIdx.z) & 63) * 8388608) + ((((int)blockIdx.z) >> 6) * 1048576)) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B[((((((((int)blockIdx.z) & 63) * 262144) + ((((int)blockIdx.z) >> 6) * 32768)) + (i_4 * 1024)) + (((int)threadIdx.x) * 8)) + 8192)])));
  }
  tl::cp_async_commit();
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dA_cumsum_shared)[(((int)threadIdx.x) + 128)] = dA_cumsum[((((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + ((int)threadIdx.x)) + 128)];
  }
  tl::cp_async_commit();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dt_shared)[(((int)threadIdx.x) + 128)] = dt[((((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + ((int)threadIdx.x)) + 128)];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    tl::cp_async_gs<16>((&(((half_t*)x_shared)[((((((i_5 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(x[((((((((((int)blockIdx.z) & 63) * 8388608) + ((((int)blockIdx.z) >> 6) * 1048576)) + (i_5 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 524288)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_6 = 0; i_6 < 8; ++i_6) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(B[((((((((int)blockIdx.z) & 63) * 262144) + ((((int)blockIdx.z) >> 6) * 32768)) + (i_6 * 1024)) + (((int)threadIdx.x) * 8)) + 16384)])));
  }
  tl::cp_async_commit();
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dA_cumsum_shared)[(((int)threadIdx.x) + 192)] = dA_cumsum[((((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + ((int)threadIdx.x)) + 192)];
  }
  tl::cp_async_commit();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dt_shared)[(((int)threadIdx.x) + 192)] = dt[((((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + ((int)threadIdx.x)) + 192)];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 4; ++i_7) {
    tl::cp_async_gs<16>((&(((half_t*)x_shared)[((((((i_7 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)])), (&(x[((((((((((int)blockIdx.z) & 63) * 8388608) + ((((int)blockIdx.z) >> 6) * 1048576)) + (i_7 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 786432)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_8 = 0; i_8 < 8; ++i_8) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_8 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)])), (&(B[((((((((int)blockIdx.z) & 63) * 262144) + ((((int)blockIdx.z) >> 6) * 32768)) + (i_8 * 1024)) + (((int)threadIdx.x) * 8)) + 24576)])));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<15>();
  __syncthreads();
  #pragma unroll
  for (int i_9 = 0; i_9 < 8; ++i_9) {
    half_t dA_cumsum_shared_local_cast[2];
    *(uint1*)(dA_cumsum_shared_local_cast + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + ((i_9 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __1;
    uint1 v_ = *(uint1*)(dA_cumsum_shared_local_cast + 0);
    ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
    *(float2*)(dA_cumsum_local + (i_9 * 2)) = __1;
  }
  tl::cp_async_wait<14>();
  __syncthreads();
  #pragma unroll
  for (int i_10 = 0; i_10 < 8; ++i_10) {
    half_t dt_shared_local_cast_1[2];
    *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_10 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __2;
    uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
    ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
    *(float2*)(dt_local + (i_10 * 2)) = __2;
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 4; ++i_11) {
    float broadcast_var_1 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    float4 __3;
      float4 __4;
      float4 __5;
        float4 v__2 = make_float4((dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
        float4 __6;
          float4 v__3 = *(float4*)(dA_cumsum_local + (i_11 * 4));
          float4 v__4 = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
          *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(v__3.x)), *(float2*)(&(v__4.x)));
          *(float2*)(&(__6.z)) = tl::mul2(*(float2*)(&(v__3.z)), *(float2*)(&(v__4.z)));
        *(float2*)(&(__5.x)) = tl::sub2(*(float2*)(&(v__2.x)), *(float2*)(&(__6.x)));
        *(float2*)(&(__5.z)) = tl::sub2(*(float2*)(&(v__2.z)), *(float2*)(&(__6.z)));
      __4.x = exp2f(__5.x);
      __4.y = exp2f(__5.y);
      __4.z = exp2f(__5.z);
      __4.w = exp2f(__5.w);
      float4 v__5 = *(float4*)(dt_local + (i_11 * 4));
      *(float2*)(&(__3.x)) = tl::mul2(*(float2*)(&(__4.x)), *(float2*)(&(v__5.x)));
      *(float2*)(&(__3.z)) = tl::mul2(*(float2*)(&(__4.z)), *(float2*)(&(v__5.z)));
    *(float4*)(scale + (i_11 * 4)) = __3;
  }
  tl::cp_async_wait<13>();
  __syncthreads();
  #pragma unroll
  for (int i_12 = 0; i_12 < 64; ++i_12) {
    x_local[i_12] = ((half_t*)x_shared)[((((((((i_12 >> 3) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_12 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_12 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_12 & 7) >> 2) + (i_12 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 64; ++i_13) {
    xt_local[i_13] = ((half_t)(((float)x_local[((((((i_13 >> 4) * 16) + (((i_13 & 7) >> 2) * 8)) + ((i_13 & 1) * 4)) + (((i_13 & 15) >> 3) * 2)) + ((i_13 & 3) >> 1))]) * scale[((((i_13 >> 4) * 4) + (((i_13 & 7) >> 2) * 2)) + (i_13 & 1))]));
  }
  tl::cp_async_wait<12>();
  __syncthreads();
  {
    half_t B_local[32];
    for (int ki = 0; ki < 4; ++ki) {
      #pragma unroll
      for (int i_14 = 0; i_14 < 4; ++i_14) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 4096) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_14 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_14 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_14 * 8)])));
      }
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_15 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki * 16) + (i_15 * 8))), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_15 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki * 16) + (i_15 * 8))), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<11>();
  __syncthreads();
  #pragma unroll
  for (int i_16 = 0; i_16 < 8; ++i_16) {
    half_t dA_cumsum_shared_local_cast_2[2];
    *(uint1*)(dA_cumsum_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + (((i_16 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
    float2 __7;
    uint1 v__6 = *(uint1*)(dA_cumsum_shared_local_cast_2 + 0);
    ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__6))[0]);
    *(float2*)(dA_cumsum_local + (i_16 * 2)) = __7;
  }
  tl::cp_async_wait<10>();
  __syncthreads();
  #pragma unroll
  for (int i_17 = 0; i_17 < 8; ++i_17) {
    half_t dt_shared_local_cast_3[2];
    *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + (((i_17 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
    float2 __8;
    uint1 v__7 = *(uint1*)(dt_shared_local_cast_3 + 0);
    ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__7))[0]);
    *(float2*)(dt_local + (i_17 * 2)) = __8;
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 4; ++i_18) {
    float broadcast_var_2 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    float4 __9;
      float4 __10;
      float4 __11;
        float4 v__8 = make_float4((dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
        float4 __12;
          float4 v__9 = *(float4*)(dA_cumsum_local + (i_18 * 4));
          float4 v__10 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
          *(float2*)(&(__12.x)) = tl::mul2(*(float2*)(&(v__9.x)), *(float2*)(&(v__10.x)));
          *(float2*)(&(__12.z)) = tl::mul2(*(float2*)(&(v__9.z)), *(float2*)(&(v__10.z)));
        *(float2*)(&(__11.x)) = tl::sub2(*(float2*)(&(v__8.x)), *(float2*)(&(__12.x)));
        *(float2*)(&(__11.z)) = tl::sub2(*(float2*)(&(v__8.z)), *(float2*)(&(__12.z)));
      __10.x = exp2f(__11.x);
      __10.y = exp2f(__11.y);
      __10.z = exp2f(__11.z);
      __10.w = exp2f(__11.w);
      float4 v__11 = *(float4*)(dt_local + (i_18 * 4));
      *(float2*)(&(__9.x)) = tl::mul2(*(float2*)(&(__10.x)), *(float2*)(&(v__11.x)));
      *(float2*)(&(__9.z)) = tl::mul2(*(float2*)(&(__10.z)), *(float2*)(&(v__11.z)));
    *(float4*)(scale + (i_18 * 4)) = __9;
  }
  tl::cp_async_wait<9>();
  __syncthreads();
  #pragma unroll
  for (int i_19 = 0; i_19 < 64; ++i_19) {
    x_local[i_19] = ((half_t*)x_shared)[(((((((((i_19 >> 3) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_19 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_19 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_19 & 7) >> 2) + (i_19 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 4096)];
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 64; ++i_20) {
    xt_local[i_20] = ((half_t)(((float)x_local[((((((i_20 >> 4) * 16) + (((i_20 & 7) >> 2) * 8)) + ((i_20 & 1) * 4)) + (((i_20 & 15) >> 3) * 2)) + ((i_20 & 3) >> 1))]) * scale[((((i_20 >> 4) * 4) + (((i_20 & 7) >> 2) * 2)) + (i_20 & 1))]));
  }
  tl::cp_async_wait<8>();
  __syncthreads();
  {
    half_t B_local_1[32];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      #pragma unroll
      for (int i_21 = 0; i_21 < 4; ++i_21) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 4096) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_21 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_1[(i_21 * 8)])));
      }
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_22 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki_1 * 16) + (i_22 * 8))), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_22 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki_1 * 16) + (i_22 * 8))), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<7>();
  __syncthreads();
  #pragma unroll
  for (int i_23 = 0; i_23 < 8; ++i_23) {
    half_t dA_cumsum_shared_local_cast_4[2];
    *(uint1*)(dA_cumsum_shared_local_cast_4 + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + (((i_23 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 128));
    float2 __13;
    uint1 v__12 = *(uint1*)(dA_cumsum_shared_local_cast_4 + 0);
    ((float2*)(&__13))[0] = __half22float2(((half2*)(&v__12))[0]);
    *(float2*)(dA_cumsum_local + (i_23 * 2)) = __13;
  }
  tl::cp_async_wait<6>();
  __syncthreads();
  #pragma unroll
  for (int i_24 = 0; i_24 < 8; ++i_24) {
    half_t dt_shared_local_cast_5[2];
    *(uint1*)(dt_shared_local_cast_5 + 0) = *(uint1*)(((half_t*)dt_shared) + (((i_24 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 128));
    float2 __14;
    uint1 v__13 = *(uint1*)(dt_shared_local_cast_5 + 0);
    ((float2*)(&__14))[0] = __half22float2(((half2*)(&v__13))[0]);
    *(float2*)(dt_local + (i_24 * 2)) = __14;
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 4; ++i_25) {
    float broadcast_var_3 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    float4 __15;
      float4 __16;
      float4 __17;
        float4 v__14 = make_float4((dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
        float4 __18;
          float4 v__15 = *(float4*)(dA_cumsum_local + (i_25 * 4));
          float4 v__16 = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
          *(float2*)(&(__18.x)) = tl::mul2(*(float2*)(&(v__15.x)), *(float2*)(&(v__16.x)));
          *(float2*)(&(__18.z)) = tl::mul2(*(float2*)(&(v__15.z)), *(float2*)(&(v__16.z)));
        *(float2*)(&(__17.x)) = tl::sub2(*(float2*)(&(v__14.x)), *(float2*)(&(__18.x)));
        *(float2*)(&(__17.z)) = tl::sub2(*(float2*)(&(v__14.z)), *(float2*)(&(__18.z)));
      __16.x = exp2f(__17.x);
      __16.y = exp2f(__17.y);
      __16.z = exp2f(__17.z);
      __16.w = exp2f(__17.w);
      float4 v__17 = *(float4*)(dt_local + (i_25 * 4));
      *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(__16.x)), *(float2*)(&(v__17.x)));
      *(float2*)(&(__15.z)) = tl::mul2(*(float2*)(&(__16.z)), *(float2*)(&(v__17.z)));
    *(float4*)(scale + (i_25 * 4)) = __15;
  }
  tl::cp_async_wait<5>();
  __syncthreads();
  #pragma unroll
  for (int i_26 = 0; i_26 < 64; ++i_26) {
    x_local[i_26] = ((half_t*)x_shared)[(((((((((i_26 >> 3) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_26 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_26 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_26 & 7) >> 2) + (i_26 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 8192)];
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 64; ++i_27) {
    xt_local[i_27] = ((half_t)(((float)x_local[((((((i_27 >> 4) * 16) + (((i_27 & 7) >> 2) * 8)) + ((i_27 & 1) * 4)) + (((i_27 & 15) >> 3) * 2)) + ((i_27 & 3) >> 1))]) * scale[((((i_27 >> 4) * 4) + (((i_27 & 7) >> 2) * 2)) + (i_27 & 1))]));
  }
  tl::cp_async_wait<4>();
  __syncthreads();
  {
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
      #pragma unroll
      for (int i_28 = 0; i_28 < 4; ++i_28) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 4096) + (ki_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_28 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(B_local_2[(i_28 * 8)])));
      }
      for (int i_29 = 0; i_29 < 2; ++i_29) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_29 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki_2 * 16) + (i_29 * 8))), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_29 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki_2 * 16) + (i_29 * 8))), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  #pragma unroll
  for (int i_30 = 0; i_30 < 8; ++i_30) {
    half_t dA_cumsum_shared_local_cast_6[2];
    *(uint1*)(dA_cumsum_shared_local_cast_6 + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + (((i_30 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 192));
    float2 __19;
    uint1 v__18 = *(uint1*)(dA_cumsum_shared_local_cast_6 + 0);
    ((float2*)(&__19))[0] = __half22float2(((half2*)(&v__18))[0]);
    *(float2*)(dA_cumsum_local + (i_30 * 2)) = __19;
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  #pragma unroll
  for (int i_31 = 0; i_31 < 8; ++i_31) {
    half_t dt_shared_local_cast_7[2];
    *(uint1*)(dt_shared_local_cast_7 + 0) = *(uint1*)(((half_t*)dt_shared) + (((i_31 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 192));
    float2 __20;
    uint1 v__19 = *(uint1*)(dt_shared_local_cast_7 + 0);
    ((float2*)(&__20))[0] = __half22float2(((half2*)(&v__19))[0]);
    *(float2*)(dt_local + (i_31 * 2)) = __20;
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 4; ++i_32) {
    float broadcast_var_4 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    float4 __21;
      float4 __22;
      float4 __23;
        float4 v__20 = make_float4((dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
        float4 __24;
          float4 v__21 = *(float4*)(dA_cumsum_local + (i_32 * 4));
          float4 v__22 = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
          *(float2*)(&(__24.x)) = tl::mul2(*(float2*)(&(v__21.x)), *(float2*)(&(v__22.x)));
          *(float2*)(&(__24.z)) = tl::mul2(*(float2*)(&(v__21.z)), *(float2*)(&(v__22.z)));
        *(float2*)(&(__23.x)) = tl::sub2(*(float2*)(&(v__20.x)), *(float2*)(&(__24.x)));
        *(float2*)(&(__23.z)) = tl::sub2(*(float2*)(&(v__20.z)), *(float2*)(&(__24.z)));
      __22.x = exp2f(__23.x);
      __22.y = exp2f(__23.y);
      __22.z = exp2f(__23.z);
      __22.w = exp2f(__23.w);
      float4 v__23 = *(float4*)(dt_local + (i_32 * 4));
      *(float2*)(&(__21.x)) = tl::mul2(*(float2*)(&(__22.x)), *(float2*)(&(v__23.x)));
      *(float2*)(&(__21.z)) = tl::mul2(*(float2*)(&(__22.z)), *(float2*)(&(v__23.z)));
    *(float4*)(scale + (i_32 * 4)) = __21;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_33 = 0; i_33 < 64; ++i_33) {
    x_local[i_33] = ((half_t*)x_shared)[(((((((((i_33 >> 3) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_33 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_33 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_33 & 7) >> 2) + (i_33 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 12288)];
  }
  #pragma unroll
  for (int i_34 = 0; i_34 < 64; ++i_34) {
    xt_local[i_34] = ((half_t)(((float)x_local[((((((i_34 >> 4) * 16) + (((i_34 & 7) >> 2) * 8)) + ((i_34 & 1) * 4)) + (((i_34 & 15) >> 3) * 2)) + ((i_34 & 3) >> 1))]) * scale[((((i_34 >> 4) * 4) + (((i_34 & 7) >> 2) * 2)) + (i_34 & 1))]));
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t B_local_3[32];
    for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
      #pragma unroll
      for (int i_35 = 0; i_35 < 4; ++i_35) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 4096) + (ki_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_35 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_35 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)])), (&(B_local_3[(i_35 * 8)])));
      }
      for (int i_36 = 0; i_36 < 2; ++i_36) {
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_36 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki_3 * 16) + (i_36 * 8))), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_36 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki_3 * 16) + (i_36 * 8))), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_37 = 0; i_37 < 8; ++i_37) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_o_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_37 >> 2) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + ((i_37 & 3) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_37 * 8)]), ((half_t)acc_o[((i_37 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_37 * 8) + 2)]), ((half_t)acc_o[((i_37 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_37 * 8) + 4)]), ((half_t)acc_o[((i_37 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_37 * 8) + 6)]), ((half_t)acc_o[((i_37 * 8) + 7)])));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store((&(Output[((((((int)blockIdx.z) & 63) * 4194304) + ((((int)blockIdx.z) >> 6) * 524288)) + (((int)blockIdx.x) * 8192))])), (&(((half_t*)acc_o_shared)[0])), 16384);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

