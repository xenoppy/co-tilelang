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
  float dA_cumsum_local[16];
  float dt_local[16];
  float scale[16];
  half_t x_local[64];
  half_t xt_local[64];
  dA_cs_last[0] = ((float)dA_cumsum[(((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + 255)]);
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dA_cumsum_shared)[((int)threadIdx.x)] = dA_cumsum[(((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + ((int)threadIdx.x))];
  }
  tl::cp_async_commit();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dt_shared)[((int)threadIdx.x)] = dt[(((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + ((int)threadIdx.x))];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)x_shared)[(((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(x[(((((((int)blockIdx.z) * 1048576) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((int)blockIdx.z) * 32768) + (i_2 * 1024)) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dA_cumsum_shared)[(((int)threadIdx.x) + 64)] = dA_cumsum[((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + ((int)threadIdx.x)) + 64)];
  }
  tl::cp_async_commit();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dt_shared)[(((int)threadIdx.x) + 64)] = dt[((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + ((int)threadIdx.x)) + 64)];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)x_shared)[((((((i_3 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(x[((((((((int)blockIdx.z) * 1048576) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B[((((((int)blockIdx.z) * 32768) + (i_4 * 1024)) + (((int)threadIdx.x) * 8)) + 8192)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 2; ++k) {
    tl::cp_async_wait<7>();
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      half_t dA_cumsum_shared_local_cast[2];
      *(uint1*)(dA_cumsum_shared_local_cast + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + (((k * 64) + (i_5 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __1;
      uint1 v_ = *(uint1*)(dA_cumsum_shared_local_cast + 0);
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
      *(float2*)(dA_cumsum_local + (i_5 * 2)) = __1;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      ((half_t*)dA_cumsum_shared)[((k * 64) + ((int)threadIdx.x))] = dA_cumsum[(((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + (k * 64)) + ((int)threadIdx.x)) + 128)];
    }
    tl::cp_async_commit();
    tl::cp_async_wait<7>();
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      half_t dt_shared_local_cast_1[2];
      *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + (((k * 64) + (i_6 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __2;
      uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
      ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
      *(float2*)(dt_local + (i_6 * 2)) = __2;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      ((half_t*)dt_shared)[((k * 64) + ((int)threadIdx.x))] = dt[(((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + (k * 64)) + ((int)threadIdx.x)) + 128)];
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      float broadcast_var_1 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
      float4 __3;
        float4 __4;
        float4 __5;
          float4 v__2 = make_float4((dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
          float4 __6;
            float4 v__3 = *(float4*)(dA_cumsum_local + (i_7 * 4));
            float4 v__4 = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
            *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(v__3.x)), *(float2*)(&(v__4.x)));
            *(float2*)(&(__6.z)) = tl::mul2(*(float2*)(&(v__3.z)), *(float2*)(&(v__4.z)));
          *(float2*)(&(__5.x)) = tl::sub2(*(float2*)(&(v__2.x)), *(float2*)(&(__6.x)));
          *(float2*)(&(__5.z)) = tl::sub2(*(float2*)(&(v__2.z)), *(float2*)(&(__6.z)));
        __4.x = exp2f(__5.x);
        __4.y = exp2f(__5.y);
        __4.z = exp2f(__5.z);
        __4.w = exp2f(__5.w);
        float4 v__5 = *(float4*)(dt_local + (i_7 * 4));
        *(float2*)(&(__3.x)) = tl::mul2(*(float2*)(&(__4.x)), *(float2*)(&(v__5.x)));
        *(float2*)(&(__3.z)) = tl::mul2(*(float2*)(&(__4.z)), *(float2*)(&(v__5.z)));
      *(float4*)(scale + (i_7 * 4)) = __3;
    }
    tl::cp_async_wait<7>();
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 64; ++i_8) {
      x_local[i_8] = ((half_t*)x_shared)[((((((((k * 4096) + ((i_8 >> 3) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((i_8 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_8 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_8 & 7) >> 2) + (i_8 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      tl::cp_async_gs<16>((&(((half_t*)x_shared)[((((((k * 4096) + (i_9 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(x[(((((((((int)blockIdx.z) * 1048576) + (k * 262144)) + (i_9 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 524288)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      xt_local[i_10] = ((half_t)(((float)x_local[((((((i_10 >> 4) * 16) + (((i_10 & 7) >> 2) * 8)) + ((i_10 & 1) * 4)) + (((i_10 & 15) >> 3) * 2)) + ((i_10 & 3) >> 1))]) * scale[((((i_10 >> 4) * 4) + (((i_10 & 7) >> 2) * 2)) + (i_10 & 1))]));
    }
    tl::cp_async_wait<7>();
    __syncthreads();
    {
      half_t B_local[32];
      for (int ki = 0; ki < 4; ++ki) {
        #pragma unroll
        for (int i_11 = 0; i_11 < 4; ++i_11) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[(((((((k * 8192) + ((((int)threadIdx.x) >> 6) * 4096)) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_11 * 8)])));
        }
        for (int i_12 = 0; i_12 < 2; ++i_12) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_12 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki * 16) + (i_12 * 8))), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_12 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki * 16) + (i_12 * 8))), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((k * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_13 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((((int)blockIdx.z) * 32768) + (k * 8192)) + (i_13 * 1024)) + (((int)threadIdx.x) * 8)) + 16384)])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<7>();
  __syncthreads();
  #pragma unroll
  for (int i_14 = 0; i_14 < 8; ++i_14) {
    half_t dA_cumsum_shared_local_cast_2[2];
    *(uint1*)(dA_cumsum_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + ((i_14 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __7;
    uint1 v__6 = *(uint1*)(dA_cumsum_shared_local_cast_2 + 0);
    ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__6))[0]);
    *(float2*)(dA_cumsum_local + (i_14 * 2)) = __7;
  }
  tl::cp_async_wait<6>();
  __syncthreads();
  #pragma unroll
  for (int i_15 = 0; i_15 < 8; ++i_15) {
    half_t dt_shared_local_cast_3[2];
    *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + ((i_15 * 8) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __8;
    uint1 v__7 = *(uint1*)(dt_shared_local_cast_3 + 0);
    ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__7))[0]);
    *(float2*)(dt_local + (i_15 * 2)) = __8;
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 4; ++i_16) {
    float broadcast_var_2 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    float4 __9;
      float4 __10;
      float4 __11;
        float4 v__8 = make_float4((dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
        float4 __12;
          float4 v__9 = *(float4*)(dA_cumsum_local + (i_16 * 4));
          float4 v__10 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
          *(float2*)(&(__12.x)) = tl::mul2(*(float2*)(&(v__9.x)), *(float2*)(&(v__10.x)));
          *(float2*)(&(__12.z)) = tl::mul2(*(float2*)(&(v__9.z)), *(float2*)(&(v__10.z)));
        *(float2*)(&(__11.x)) = tl::sub2(*(float2*)(&(v__8.x)), *(float2*)(&(__12.x)));
        *(float2*)(&(__11.z)) = tl::sub2(*(float2*)(&(v__8.z)), *(float2*)(&(__12.z)));
      __10.x = exp2f(__11.x);
      __10.y = exp2f(__11.y);
      __10.z = exp2f(__11.z);
      __10.w = exp2f(__11.w);
      float4 v__11 = *(float4*)(dt_local + (i_16 * 4));
      *(float2*)(&(__9.x)) = tl::mul2(*(float2*)(&(__10.x)), *(float2*)(&(v__11.x)));
      *(float2*)(&(__9.z)) = tl::mul2(*(float2*)(&(__10.z)), *(float2*)(&(v__11.z)));
    *(float4*)(scale + (i_16 * 4)) = __9;
  }
  tl::cp_async_wait<5>();
  __syncthreads();
  #pragma unroll
  for (int i_17 = 0; i_17 < 64; ++i_17) {
    x_local[i_17] = ((half_t*)x_shared)[((((((((i_17 >> 3) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_17 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_17 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_17 & 7) >> 2) + (i_17 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 64; ++i_18) {
    xt_local[i_18] = ((half_t)(((float)x_local[((((((i_18 >> 4) * 16) + (((i_18 & 7) >> 2) * 8)) + ((i_18 & 1) * 4)) + (((i_18 & 15) >> 3) * 2)) + ((i_18 & 3) >> 1))]) * scale[((((i_18 >> 4) * 4) + (((i_18 & 7) >> 2) * 2)) + (i_18 & 1))]));
  }
  tl::cp_async_wait<4>();
  __syncthreads();
  {
    half_t B_local_1[32];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      #pragma unroll
      for (int i_19 = 0; i_19 < 4; ++i_19) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 4096) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_19 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_19 * 8)])));
      }
      for (int i_20 = 0; i_20 < 2; ++i_20) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_20 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki_1 * 16) + (i_20 * 8))), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_20 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki_1 * 16) + (i_20 * 8))), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  #pragma unroll
  for (int i_21 = 0; i_21 < 8; ++i_21) {
    half_t dA_cumsum_shared_local_cast_4[2];
    *(uint1*)(dA_cumsum_shared_local_cast_4 + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + (((i_21 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
    float2 __13;
    uint1 v__12 = *(uint1*)(dA_cumsum_shared_local_cast_4 + 0);
    ((float2*)(&__13))[0] = __half22float2(((half2*)(&v__12))[0]);
    *(float2*)(dA_cumsum_local + (i_21 * 2)) = __13;
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  #pragma unroll
  for (int i_22 = 0; i_22 < 8; ++i_22) {
    half_t dt_shared_local_cast_5[2];
    *(uint1*)(dt_shared_local_cast_5 + 0) = *(uint1*)(((half_t*)dt_shared) + (((i_22 * 8) + ((((int)threadIdx.x) & 3) * 2)) + 64));
    float2 __14;
    uint1 v__13 = *(uint1*)(dt_shared_local_cast_5 + 0);
    ((float2*)(&__14))[0] = __half22float2(((half2*)(&v__13))[0]);
    *(float2*)(dt_local + (i_22 * 2)) = __14;
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 4; ++i_23) {
    float broadcast_var_3 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    float4 __15;
      float4 __16;
      float4 __17;
        float4 v__14 = make_float4((dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
        float4 __18;
          float4 v__15 = *(float4*)(dA_cumsum_local + (i_23 * 4));
          float4 v__16 = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
          *(float2*)(&(__18.x)) = tl::mul2(*(float2*)(&(v__15.x)), *(float2*)(&(v__16.x)));
          *(float2*)(&(__18.z)) = tl::mul2(*(float2*)(&(v__15.z)), *(float2*)(&(v__16.z)));
        *(float2*)(&(__17.x)) = tl::sub2(*(float2*)(&(v__14.x)), *(float2*)(&(__18.x)));
        *(float2*)(&(__17.z)) = tl::sub2(*(float2*)(&(v__14.z)), *(float2*)(&(__18.z)));
      __16.x = exp2f(__17.x);
      __16.y = exp2f(__17.y);
      __16.z = exp2f(__17.z);
      __16.w = exp2f(__17.w);
      float4 v__17 = *(float4*)(dt_local + (i_23 * 4));
      *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(__16.x)), *(float2*)(&(v__17.x)));
      *(float2*)(&(__15.z)) = tl::mul2(*(float2*)(&(__16.z)), *(float2*)(&(v__17.z)));
    *(float4*)(scale + (i_23 * 4)) = __15;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_24 = 0; i_24 < 64; ++i_24) {
    x_local[i_24] = ((half_t*)x_shared)[(((((((((i_24 >> 3) * 512) + ((((int)threadIdx.x) & 3) * 128)) + (((i_24 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_24 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_24 & 7) >> 2) + (i_24 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 4096)];
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 64; ++i_25) {
    xt_local[i_25] = ((half_t)(((float)x_local[((((((i_25 >> 4) * 16) + (((i_25 & 7) >> 2) * 8)) + ((i_25 & 1) * 4)) + (((i_25 & 15) >> 3) * 2)) + ((i_25 & 3) >> 1))]) * scale[((((i_25 >> 4) * 4) + (((i_25 & 7) >> 2) * 2)) + (i_25 & 1))]));
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
      #pragma unroll
      for (int i_26 = 0; i_26 < 4; ++i_26) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 4096) + (ki_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_26 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_26 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_2[(i_26 * 8)])));
      }
      for (int i_27 = 0; i_27 < 2; ++i_27) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_27 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki_2 * 16) + (i_27 * 8))), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_27 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki_2 * 16) + (i_27 * 8))), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_28 = 0; i_28 < 8; ++i_28) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_o_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_28 >> 2) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + ((i_28 & 3) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_28 * 8)]), ((half_t)acc_o[((i_28 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_28 * 8) + 2)]), ((half_t)acc_o[((i_28 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_28 * 8) + 4)]), ((half_t)acc_o[((i_28 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_28 * 8) + 6)]), ((half_t)acc_o[((i_28 * 8) + 7)])));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store((&(Output[((((int)blockIdx.z) * 524288) + (((int)blockIdx.x) * 8192))])), (&(((half_t*)acc_o_shared)[0])), 16384);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

