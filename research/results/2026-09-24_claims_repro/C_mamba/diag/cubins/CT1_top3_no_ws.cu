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
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 256));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 512));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 16896));
  float dA_cs_last[1];
  float acc_o[64];
  float dA_cumsum_local[8];
  float dt_local[8];
  float scale[8];
  half_t x_local[32];
  half_t xt_local[32];
  dA_cs_last[0] = ((float)dA_cumsum[(((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + 255)]);
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (((int)threadIdx.x) < 32) {
    ((half_t*)dA_cumsum_shared)[((int)threadIdx.x)] = dA_cumsum[(((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + ((int)threadIdx.x))];
  }
  tl::cp_async_commit();
  if (((int)threadIdx.x) < 32) {
    ((half_t*)dt_shared)[((int)threadIdx.x)] = dt[(((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + ((int)threadIdx.x))];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)x_shared)[(((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(x[(((((((int)blockIdx.z) * 1048576) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((int)blockIdx.z) * 32768) + (i_2 * 1024)) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((half_t*)dA_cumsum_shared)[(((int)threadIdx.x) + 32)] = dA_cumsum[((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + ((int)threadIdx.x)) + 32)];
  }
  tl::cp_async_commit();
  if (((int)threadIdx.x) < 32) {
    ((half_t*)dt_shared)[(((int)threadIdx.x) + 32)] = dt[((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + ((int)threadIdx.x)) + 32)];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)x_shared)[((((((i_3 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(x[((((((((int)blockIdx.z) * 1048576) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 131072)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B[((((((int)blockIdx.z) * 32768) + (i_4 * 1024)) + (((int)threadIdx.x) * 8)) + 4096)])));
  }
  tl::cp_async_commit();
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((half_t*)dA_cumsum_shared)[(((int)threadIdx.x) + 64)] = dA_cumsum[((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + ((int)threadIdx.x)) + 64)];
  }
  tl::cp_async_commit();
  if (((int)threadIdx.x) < 32) {
    ((half_t*)dt_shared)[(((int)threadIdx.x) + 64)] = dt[((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + ((int)threadIdx.x)) + 64)];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 2; ++i_5) {
    tl::cp_async_gs<16>((&(((half_t*)x_shared)[((((((i_5 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(x[((((((((int)blockIdx.z) * 1048576) + (i_5 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_6 = 0; i_6 < 4; ++i_6) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B[((((((int)blockIdx.z) * 32768) + (i_6 * 1024)) + (((int)threadIdx.x) * 8)) + 8192)])));
  }
  tl::cp_async_commit();
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((half_t*)dA_cumsum_shared)[(((int)threadIdx.x) + 96)] = dA_cumsum[((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + ((int)threadIdx.x)) + 96)];
  }
  tl::cp_async_commit();
  if (((int)threadIdx.x) < 32) {
    ((half_t*)dt_shared)[(((int)threadIdx.x) + 96)] = dt[((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + ((int)threadIdx.x)) + 96)];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 2; ++i_7) {
    tl::cp_async_gs<16>((&(((half_t*)x_shared)[((((((i_7 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6144)])), (&(x[((((((((int)blockIdx.z) * 1048576) + (i_7 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 393216)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_8 = 0; i_8 < 4; ++i_8) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_8 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)])), (&(B[((((((int)blockIdx.z) * 32768) + (i_8 * 1024)) + (((int)threadIdx.x) * 8)) + 12288)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 4; ++k) {
    tl::cp_async_wait<15>();
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      half_t dA_cumsum_shared_local_cast[2];
      *(uint1*)(dA_cumsum_shared_local_cast + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + (((k * 32) + (i_9 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __1;
      uint1 v_ = *(uint1*)(dA_cumsum_shared_local_cast + 0);
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
      *(float2*)(dA_cumsum_local + (i_9 * 2)) = __1;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      ((half_t*)dA_cumsum_shared)[((k * 32) + ((int)threadIdx.x))] = dA_cumsum[(((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + (k * 32)) + ((int)threadIdx.x)) + 128)];
    }
    tl::cp_async_commit();
    tl::cp_async_wait<15>();
    __syncthreads();
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      half_t dt_shared_local_cast_1[2];
      *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + (((k * 32) + (i_10 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __2;
      uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
      ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
      *(float2*)(dt_local + (i_10 * 2)) = __2;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      ((half_t*)dt_shared)[((k * 32) + ((int)threadIdx.x))] = dt[(((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + (k * 32)) + ((int)threadIdx.x)) + 128)];
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
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
    tl::cp_async_wait<15>();
    __syncthreads();
    #pragma unroll
    for (int i_12 = 0; i_12 < 32; ++i_12) {
      x_local[i_12] = ((half_t*)x_shared)[((((((((k * 2048) + ((i_12 >> 3) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((i_12 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_12 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_12 & 7) >> 2) + (i_12 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    __syncthreads();
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      tl::cp_async_gs<16>((&(((half_t*)x_shared)[((((((k * 2048) + (i_13 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(x[(((((((((int)blockIdx.z) * 1048576) + (k * 131072)) + (i_13 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 524288)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_14 = 0; i_14 < 32; ++i_14) {
      xt_local[i_14] = ((half_t)(((float)x_local[((((((i_14 >> 4) * 16) + (((i_14 & 7) >> 2) * 8)) + ((i_14 & 1) * 4)) + (((i_14 & 15) >> 3) * 2)) + ((i_14 & 3) >> 1))]) * scale[((((i_14 >> 4) * 4) + (((i_14 & 7) >> 2) * 2)) + (i_14 & 1))]));
    }
    tl::cp_async_wait<15>();
    __syncthreads();
    {
      half_t B_local[32];
      for (int ki = 0; ki < 2; ++ki) {
        #pragma unroll
        for (int i_15 = 0; i_15 < 4; ++i_15) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[(((((((k * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_15 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_15 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_15 * 8)])));
        }
        for (int i_16 = 0; i_16 < 2; ++i_16) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_16 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki * 16) + (i_16 * 8))), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_16 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki * 16) + (i_16 * 8))), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_17 = 0; i_17 < 4; ++i_17) {
      tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((k * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_17 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((((int)blockIdx.z) * 32768) + (k * 4096)) + (i_17 * 1024)) + (((int)threadIdx.x) * 8)) + 16384)])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<15>();
  __syncthreads();
  #pragma unroll
  for (int i_18 = 0; i_18 < 4; ++i_18) {
    half_t dA_cumsum_shared_local_cast_2[2];
    *(uint1*)(dA_cumsum_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + ((i_18 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __7;
    uint1 v__6 = *(uint1*)(dA_cumsum_shared_local_cast_2 + 0);
    ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__6))[0]);
    *(float2*)(dA_cumsum_local + (i_18 * 2)) = __7;
  }
  tl::cp_async_wait<14>();
  __syncthreads();
  #pragma unroll
  for (int i_19 = 0; i_19 < 4; ++i_19) {
    half_t dt_shared_local_cast_3[2];
    *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_19 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __8;
    uint1 v__7 = *(uint1*)(dt_shared_local_cast_3 + 0);
    ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__7))[0]);
    *(float2*)(dt_local + (i_19 * 2)) = __8;
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 2; ++i_20) {
    float broadcast_var_2 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    float4 __9;
      float4 __10;
      float4 __11;
        float4 v__8 = make_float4((dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
        float4 __12;
          float4 v__9 = *(float4*)(dA_cumsum_local + (i_20 * 4));
          float4 v__10 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
          *(float2*)(&(__12.x)) = tl::mul2(*(float2*)(&(v__9.x)), *(float2*)(&(v__10.x)));
          *(float2*)(&(__12.z)) = tl::mul2(*(float2*)(&(v__9.z)), *(float2*)(&(v__10.z)));
        *(float2*)(&(__11.x)) = tl::sub2(*(float2*)(&(v__8.x)), *(float2*)(&(__12.x)));
        *(float2*)(&(__11.z)) = tl::sub2(*(float2*)(&(v__8.z)), *(float2*)(&(__12.z)));
      __10.x = exp2f(__11.x);
      __10.y = exp2f(__11.y);
      __10.z = exp2f(__11.z);
      __10.w = exp2f(__11.w);
      float4 v__11 = *(float4*)(dt_local + (i_20 * 4));
      *(float2*)(&(__9.x)) = tl::mul2(*(float2*)(&(__10.x)), *(float2*)(&(v__11.x)));
      *(float2*)(&(__9.z)) = tl::mul2(*(float2*)(&(__10.z)), *(float2*)(&(v__11.z)));
    *(float4*)(scale + (i_20 * 4)) = __9;
  }
  tl::cp_async_wait<13>();
  __syncthreads();
  #pragma unroll
  for (int i_21 = 0; i_21 < 32; ++i_21) {
    x_local[i_21] = ((half_t*)x_shared)[((((((((i_21 >> 3) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_21 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_21 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_21 & 7) >> 2) + (i_21 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 32; ++i_22) {
    xt_local[i_22] = ((half_t)(((float)x_local[((((((i_22 >> 4) * 16) + (((i_22 & 7) >> 2) * 8)) + ((i_22 & 1) * 4)) + (((i_22 & 15) >> 3) * 2)) + ((i_22 & 3) >> 1))]) * scale[((((i_22 >> 4) * 4) + (((i_22 & 7) >> 2) * 2)) + (i_22 & 1))]));
  }
  tl::cp_async_wait<12>();
  __syncthreads();
  {
    half_t B_local_1[32];
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      #pragma unroll
      for (int i_23 = 0; i_23 < 4; ++i_23) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 2048) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_23 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_23 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_23 * 8)])));
      }
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_24 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki_1 * 16) + (i_24 * 8))), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_24 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki_1 * 16) + (i_24 * 8))), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<11>();
  __syncthreads();
  #pragma unroll
  for (int i_25 = 0; i_25 < 4; ++i_25) {
    half_t dA_cumsum_shared_local_cast_4[2];
    *(uint1*)(dA_cumsum_shared_local_cast_4 + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + (((i_25 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 32));
    float2 __13;
    uint1 v__12 = *(uint1*)(dA_cumsum_shared_local_cast_4 + 0);
    ((float2*)(&__13))[0] = __half22float2(((half2*)(&v__12))[0]);
    *(float2*)(dA_cumsum_local + (i_25 * 2)) = __13;
  }
  tl::cp_async_wait<10>();
  __syncthreads();
  #pragma unroll
  for (int i_26 = 0; i_26 < 4; ++i_26) {
    half_t dt_shared_local_cast_5[2];
    *(uint1*)(dt_shared_local_cast_5 + 0) = *(uint1*)(((half_t*)dt_shared) + (((i_26 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 32));
    float2 __14;
    uint1 v__13 = *(uint1*)(dt_shared_local_cast_5 + 0);
    ((float2*)(&__14))[0] = __half22float2(((half2*)(&v__13))[0]);
    *(float2*)(dt_local + (i_26 * 2)) = __14;
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 2; ++i_27) {
    float broadcast_var_3 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    float4 __15;
      float4 __16;
      float4 __17;
        float4 v__14 = make_float4((dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
        float4 __18;
          float4 v__15 = *(float4*)(dA_cumsum_local + (i_27 * 4));
          float4 v__16 = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
          *(float2*)(&(__18.x)) = tl::mul2(*(float2*)(&(v__15.x)), *(float2*)(&(v__16.x)));
          *(float2*)(&(__18.z)) = tl::mul2(*(float2*)(&(v__15.z)), *(float2*)(&(v__16.z)));
        *(float2*)(&(__17.x)) = tl::sub2(*(float2*)(&(v__14.x)), *(float2*)(&(__18.x)));
        *(float2*)(&(__17.z)) = tl::sub2(*(float2*)(&(v__14.z)), *(float2*)(&(__18.z)));
      __16.x = exp2f(__17.x);
      __16.y = exp2f(__17.y);
      __16.z = exp2f(__17.z);
      __16.w = exp2f(__17.w);
      float4 v__17 = *(float4*)(dt_local + (i_27 * 4));
      *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(__16.x)), *(float2*)(&(v__17.x)));
      *(float2*)(&(__15.z)) = tl::mul2(*(float2*)(&(__16.z)), *(float2*)(&(v__17.z)));
    *(float4*)(scale + (i_27 * 4)) = __15;
  }
  tl::cp_async_wait<9>();
  __syncthreads();
  #pragma unroll
  for (int i_28 = 0; i_28 < 32; ++i_28) {
    x_local[i_28] = ((half_t*)x_shared)[(((((((((i_28 >> 3) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_28 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_28 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_28 & 7) >> 2) + (i_28 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 2048)];
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 32; ++i_29) {
    xt_local[i_29] = ((half_t)(((float)x_local[((((((i_29 >> 4) * 16) + (((i_29 & 7) >> 2) * 8)) + ((i_29 & 1) * 4)) + (((i_29 & 15) >> 3) * 2)) + ((i_29 & 3) >> 1))]) * scale[((((i_29 >> 4) * 4) + (((i_29 & 7) >> 2) * 2)) + (i_29 & 1))]));
  }
  tl::cp_async_wait<8>();
  __syncthreads();
  {
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
      #pragma unroll
      for (int i_30 = 0; i_30 < 4; ++i_30) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 2048) + (ki_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_30 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_30 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B_local_2[(i_30 * 8)])));
      }
      for (int i_31 = 0; i_31 < 2; ++i_31) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_31 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki_2 * 16) + (i_31 * 8))), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_31 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki_2 * 16) + (i_31 * 8))), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<7>();
  __syncthreads();
  #pragma unroll
  for (int i_32 = 0; i_32 < 4; ++i_32) {
    half_t dA_cumsum_shared_local_cast_6[2];
    *(uint1*)(dA_cumsum_shared_local_cast_6 + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + (((i_32 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
    float2 __19;
    uint1 v__18 = *(uint1*)(dA_cumsum_shared_local_cast_6 + 0);
    ((float2*)(&__19))[0] = __half22float2(((half2*)(&v__18))[0]);
    *(float2*)(dA_cumsum_local + (i_32 * 2)) = __19;
  }
  tl::cp_async_wait<6>();
  __syncthreads();
  #pragma unroll
  for (int i_33 = 0; i_33 < 4; ++i_33) {
    half_t dt_shared_local_cast_7[2];
    *(uint1*)(dt_shared_local_cast_7 + 0) = *(uint1*)(((half_t*)dt_shared) + (((i_33 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
    float2 __20;
    uint1 v__19 = *(uint1*)(dt_shared_local_cast_7 + 0);
    ((float2*)(&__20))[0] = __half22float2(((half2*)(&v__19))[0]);
    *(float2*)(dt_local + (i_33 * 2)) = __20;
  }
  #pragma unroll
  for (int i_34 = 0; i_34 < 2; ++i_34) {
    float broadcast_var_4 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    float4 __21;
      float4 __22;
      float4 __23;
        float4 v__20 = make_float4((dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
        float4 __24;
          float4 v__21 = *(float4*)(dA_cumsum_local + (i_34 * 4));
          float4 v__22 = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
          *(float2*)(&(__24.x)) = tl::mul2(*(float2*)(&(v__21.x)), *(float2*)(&(v__22.x)));
          *(float2*)(&(__24.z)) = tl::mul2(*(float2*)(&(v__21.z)), *(float2*)(&(v__22.z)));
        *(float2*)(&(__23.x)) = tl::sub2(*(float2*)(&(v__20.x)), *(float2*)(&(__24.x)));
        *(float2*)(&(__23.z)) = tl::sub2(*(float2*)(&(v__20.z)), *(float2*)(&(__24.z)));
      __22.x = exp2f(__23.x);
      __22.y = exp2f(__23.y);
      __22.z = exp2f(__23.z);
      __22.w = exp2f(__23.w);
      float4 v__23 = *(float4*)(dt_local + (i_34 * 4));
      *(float2*)(&(__21.x)) = tl::mul2(*(float2*)(&(__22.x)), *(float2*)(&(v__23.x)));
      *(float2*)(&(__21.z)) = tl::mul2(*(float2*)(&(__22.z)), *(float2*)(&(v__23.z)));
    *(float4*)(scale + (i_34 * 4)) = __21;
  }
  tl::cp_async_wait<5>();
  __syncthreads();
  #pragma unroll
  for (int i_35 = 0; i_35 < 32; ++i_35) {
    x_local[i_35] = ((half_t*)x_shared)[(((((((((i_35 >> 3) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_35 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_35 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_35 & 7) >> 2) + (i_35 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 4096)];
  }
  #pragma unroll
  for (int i_36 = 0; i_36 < 32; ++i_36) {
    xt_local[i_36] = ((half_t)(((float)x_local[((((((i_36 >> 4) * 16) + (((i_36 & 7) >> 2) * 8)) + ((i_36 & 1) * 4)) + (((i_36 & 15) >> 3) * 2)) + ((i_36 & 3) >> 1))]) * scale[((((i_36 >> 4) * 4) + (((i_36 & 7) >> 2) * 2)) + (i_36 & 1))]));
  }
  tl::cp_async_wait<4>();
  __syncthreads();
  {
    half_t B_local_3[32];
    for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
      #pragma unroll
      for (int i_37 = 0; i_37 < 4; ++i_37) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 2048) + (ki_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_37 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_37 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_3[(i_37 * 8)])));
      }
      for (int i_38 = 0; i_38 < 2; ++i_38) {
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_38 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki_3 * 16) + (i_38 * 8))), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_38 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki_3 * 16) + (i_38 * 8))), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  #pragma unroll
  for (int i_39 = 0; i_39 < 4; ++i_39) {
    half_t dA_cumsum_shared_local_cast_8[2];
    *(uint1*)(dA_cumsum_shared_local_cast_8 + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + (((i_39 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 96));
    float2 __25;
    uint1 v__24 = *(uint1*)(dA_cumsum_shared_local_cast_8 + 0);
    ((float2*)(&__25))[0] = __half22float2(((half2*)(&v__24))[0]);
    *(float2*)(dA_cumsum_local + (i_39 * 2)) = __25;
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  #pragma unroll
  for (int i_40 = 0; i_40 < 4; ++i_40) {
    half_t dt_shared_local_cast_9[2];
    *(uint1*)(dt_shared_local_cast_9 + 0) = *(uint1*)(((half_t*)dt_shared) + (((i_40 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 96));
    float2 __26;
    uint1 v__25 = *(uint1*)(dt_shared_local_cast_9 + 0);
    ((float2*)(&__26))[0] = __half22float2(((half2*)(&v__25))[0]);
    *(float2*)(dt_local + (i_40 * 2)) = __26;
  }
  #pragma unroll
  for (int i_41 = 0; i_41 < 2; ++i_41) {
    float broadcast_var_5 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    float4 __27;
      float4 __28;
      float4 __29;
        float4 v__26 = make_float4((dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
        float4 __30;
          float4 v__27 = *(float4*)(dA_cumsum_local + (i_41 * 4));
          float4 v__28 = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
          *(float2*)(&(__30.x)) = tl::mul2(*(float2*)(&(v__27.x)), *(float2*)(&(v__28.x)));
          *(float2*)(&(__30.z)) = tl::mul2(*(float2*)(&(v__27.z)), *(float2*)(&(v__28.z)));
        *(float2*)(&(__29.x)) = tl::sub2(*(float2*)(&(v__26.x)), *(float2*)(&(__30.x)));
        *(float2*)(&(__29.z)) = tl::sub2(*(float2*)(&(v__26.z)), *(float2*)(&(__30.z)));
      __28.x = exp2f(__29.x);
      __28.y = exp2f(__29.y);
      __28.z = exp2f(__29.z);
      __28.w = exp2f(__29.w);
      float4 v__29 = *(float4*)(dt_local + (i_41 * 4));
      *(float2*)(&(__27.x)) = tl::mul2(*(float2*)(&(__28.x)), *(float2*)(&(v__29.x)));
      *(float2*)(&(__27.z)) = tl::mul2(*(float2*)(&(__28.z)), *(float2*)(&(v__29.z)));
    *(float4*)(scale + (i_41 * 4)) = __27;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_42 = 0; i_42 < 32; ++i_42) {
    x_local[i_42] = ((half_t*)x_shared)[(((((((((i_42 >> 3) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_42 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_42 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_42 & 7) >> 2) + (i_42 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 6144)];
  }
  #pragma unroll
  for (int i_43 = 0; i_43 < 32; ++i_43) {
    xt_local[i_43] = ((half_t)(((float)x_local[((((((i_43 >> 4) * 16) + (((i_43 & 7) >> 2) * 8)) + ((i_43 & 1) * 4)) + (((i_43 & 15) >> 3) * 2)) + ((i_43 & 3) >> 1))]) * scale[((((i_43 >> 4) * 4) + (((i_43 & 7) >> 2) * 2)) + (i_43 & 1))]));
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t B_local_4[32];
    for (int ki_4 = 0; ki_4 < 2; ++ki_4) {
      #pragma unroll
      for (int i_44 = 0; i_44 < 4; ++i_44) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 2048) + (ki_4 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_44 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_44 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)])), (&(B_local_4[(i_44 * 8)])));
      }
      for (int i_45 = 0; i_45 < 2; ++i_45) {
        for (int j_4 = 0; j_4 < 4; ++j_4) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_45 * 32) + (j_4 * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki_4 * 16) + (i_45 * 8))), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_45 * 32) + (j_4 * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki_4 * 16) + (i_45 * 8))), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_46 = 0; i_46 < 8; ++i_46) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_o_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_46 >> 2) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + ((i_46 & 3) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_46 * 8)]), ((half_t)acc_o[((i_46 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_46 * 8) + 2)]), ((half_t)acc_o[((i_46 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_46 * 8) + 4)]), ((half_t)acc_o[((i_46 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_46 * 8) + 6)]), ((half_t)acc_o[((i_46 * 8) + 7)])));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store((&(Output[((((int)blockIdx.z) * 524288) + (((int)blockIdx.x) * 8192))])), (&(((half_t*)acc_o_shared)[0])), 16384);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

