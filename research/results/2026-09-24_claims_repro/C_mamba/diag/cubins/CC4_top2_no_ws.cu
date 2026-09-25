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

extern "C" __global__ void chunk_scan_fwd_kernel(const half_t* __restrict__ C, const half_t* __restrict__ D, __grid_constant__ const CUtensorMap Output_desc, const half_t* __restrict__ cb, const half_t* __restrict__ dA_cumsum, const half_t* __restrict__ dt, const half_t* __restrict__ prev_states, const half_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) chunk_scan_fwd_kernel(const half_t* __restrict__ C, const half_t* __restrict__ D, __grid_constant__ const CUtensorMap Output_desc, const half_t* __restrict__ cb, const half_t* __restrict__ dA_cumsum, const half_t* __restrict__ dt, const half_t* __restrict__ prev_states, const half_t* __restrict__ x) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* C_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* acc_o_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* cb_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* dA_cs_m_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* x_residual_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* dA_cs_k_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* prev_state_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 16640));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 16896));
  float dA_cs_m_local[4];
  float acc_o[32];
  float scale_m_local[4];
  half_t cb_local[64];
  float dA_cs_k_local[16];
  float dt_local[16];
  float D_local[1];
  float x_residual_local[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dA_cs_m_shared)[((int)threadIdx.x)] = dA_cumsum[((((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))];
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    dA_cs_m_local[i] = ((float)((half_t*)dA_cs_m_shared)[(((((((int)threadIdx.x) & 63) >> 5) * 32) + (i * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
  float4 __1;
  float4 __2;
    float4 v_ = *(float4*)(dA_cs_m_local + 0);
    float4 v__1 = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
    *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
  __1.x = exp2f(__2.x);
  __1.y = exp2f(__2.y);
  __1.z = exp2f(__2.z);
  __1.w = exp2f(__2.w);
  *(float4*)(scale_m_local + 0) = __1;
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    *(uint4*)(((half_t*)C_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(C + ((((((((int)blockIdx.z) & 63) * 262144) + ((((int)blockIdx.z) >> 6) * 32768)) + (((int)blockIdx.y) * 8192)) + (i_2 * 1024)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    *(uint4*)(((half_t*)prev_state_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(prev_states + ((((((((int)blockIdx.z) & 63) * 4194304) + ((((int)blockIdx.z) >> 6) * 524288)) + (((int)blockIdx.x) * 8192)) + (i_3 * 1024)) + (((int)threadIdx.x) * 8)));
  }
  {
    half_t A_local[16];
    half_t B_local[16];
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        tl::ptx_ldmatrix_x4((&(((half_t*)C_shared)[((((((((ki >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_4 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_4 * 8)])));
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        tl::ptx_ldmatrix_x4((&(((half_t*)prev_state_shared)[(((((((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (i_5 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_5 * 8)])));
      }
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_6 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_6 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_6 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_6 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 16; ++i_7) {
    float2 __3;
      float2 v__2 = *(float2*)(acc_o + (i_7 * 2));
      float2 v__3 = make_float2(scale_m_local[(((i_7 >> 3) * 2) + (i_7 & 1))], scale_m_local[(((i_7 >> 3) * 2) + (i_7 & 1))]);
      *(float2*)(&(__3.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
    *(float2*)(acc_o + (i_7 * 2)) = __3;
  }
  __syncthreads();
  #pragma unroll
  for (int i_8 = 0; i_8 < 4; ++i_8) {
    tl::cp_async_gs<16>((&(((half_t*)cb_shared)[(((((i_8 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(cb[(((((((((int)blockIdx.z) & 63) * 524288) + ((((int)blockIdx.z) >> 6) * 65536)) + (((int)blockIdx.y) * 16384)) + (i_8 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dA_cs_k_shared)[((int)threadIdx.x)] = dA_cumsum[(((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + ((int)threadIdx.x))];
  }
  tl::cp_async_commit();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dt_shared)[((int)threadIdx.x)] = dt[(((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + ((int)threadIdx.x))];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_9 = 0; i_9 < 4; ++i_9) {
    tl::cp_async_gs<16>((&(((half_t*)x_shared)[(((((i_9 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(x[(((((((((int)blockIdx.z) & 63) * 8388608) + ((((int)blockIdx.z) >> 6) * 1048576)) + (i_9 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  if (0 < ((int)blockIdx.y)) {
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      tl::cp_async_gs<16>((&(((half_t*)cb_shared)[((((((i_10 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(cb[((((((((((int)blockIdx.z) & 63) * 524288) + ((((int)blockIdx.z) >> 6) * 65536)) + (((int)blockIdx.y) * 16384)) + (i_10 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
    }
    tl::cp_async_commit();
  }
  __syncthreads();
  if (0 < ((int)blockIdx.y)) {
    if (((int)threadIdx.x) < 64) {
      ((half_t*)dA_cs_k_shared)[(((int)threadIdx.x) + 64)] = dA_cumsum[((((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + ((int)threadIdx.x)) + 64)];
    }
    tl::cp_async_commit();
    if (((int)threadIdx.x) < 64) {
      ((half_t*)dt_shared)[(((int)threadIdx.x) + 64)] = dt[((((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + ((int)threadIdx.x)) + 64)];
    }
    tl::cp_async_commit();
  }
  __syncthreads();
  if (0 < ((int)blockIdx.y)) {
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      tl::cp_async_gs<16>((&(((half_t*)x_shared)[((((((i_11 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(x[((((((((((int)blockIdx.z) & 63) * 8388608) + ((((int)blockIdx.z) >> 6) * 1048576)) + (i_11 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144)])));
    }
    tl::cp_async_commit();
  }
  for (int k = 0; k < (((int)blockIdx.y) - 1); ++k) {
    tl::cp_async_wait<7>();
    __syncthreads();
    #pragma unroll
    for (int i_12 = 0; i_12 < 32; ++i_12) {
      *(uint1*)(cb_local + (i_12 * 2)) = *(uint1*)(((half_t*)cb_shared) + (((((((((k * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (((i_12 & 7) >> 2) * 1024)) + ((i_12 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((i_12 >> 3) * 16) + (((i_12 & 3) >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_12 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_12 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      tl::cp_async_gs<16>((&(((half_t*)cb_shared)[((((((k * 4096) + (i_13 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(cb[(((((((((((int)blockIdx.z) & 63) * 524288) + ((((int)blockIdx.z) >> 6) * 65536)) + (((int)blockIdx.y) * 16384)) + (i_13 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<7>();
    __syncthreads();
    #pragma unroll
    for (int i_14 = 0; i_14 < 8; ++i_14) {
      half_t dA_cs_k_shared_local_cast[2];
      *(uint1*)(dA_cs_k_shared_local_cast + 0) = *(uint1*)(((half_t*)dA_cs_k_shared) + (((k * 64) + (i_14 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __4;
      uint1 v__4 = *(uint1*)(dA_cs_k_shared_local_cast + 0);
      ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__4))[0]);
      *(float2*)(dA_cs_k_local + (i_14 * 2)) = __4;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      ((half_t*)dA_cs_k_shared)[((k * 64) + ((int)threadIdx.x))] = dA_cumsum[(((((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + (k * 64)) + ((int)threadIdx.x)) + 128)];
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_15 = 0; i_15 < 32; ++i_15) {
      float broadcast_var_2 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
      uint1 __5;
      float2 __6;
        float2 __7;
        uint1 v__5 = *(uint1*)(cb_local + (i_15 * 2));
        ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__5))[0]);
        float2 __8;
        float2 __9;
          float2 v__6 = make_float2((dA_cs_m_local[((((i_15 & 7) >> 2) * 2) + (i_15 & 1))] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_m_local[((((i_15 & 7) >> 2) * 2) + (i_15 & 1))] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
          float2 __10;
            float2 v__7 = *(float2*)(dA_cs_k_local + (((i_15 >> 3) * 4) + (((i_15 & 3) >> 1) * 2)));
            float2 v__8 = make_float2(broadcast_var_2, broadcast_var_2);
            *(float2*)(&(__10.x)) = tl::mul2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)));
          *(float2*)(&(__9.x)) = tl::sub2(*(float2*)(&(v__6.x)), *(float2*)(&(__10.x)));
        __8.x = exp2f(__9.x);
        __8.y = exp2f(__9.y);
        *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(__7.x)), *(float2*)(&(__8.x)));
      ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&__6))[0]);
      *(uint1*)(cb_local + (i_15 * 2)) = __5;
    }
    tl::cp_async_wait<7>();
    __syncthreads();
    #pragma unroll
    for (int i_16 = 0; i_16 < 8; ++i_16) {
      half_t dt_shared_local_cast_1[2];
      *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + (((k * 64) + (i_16 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __11;
      uint1 v__9 = *(uint1*)(dt_shared_local_cast_1 + 0);
      ((float2*)(&__11))[0] = __half22float2(((half2*)(&v__9))[0]);
      *(float2*)(dt_local + (i_16 * 2)) = __11;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      ((half_t*)dt_shared)[((k * 64) + ((int)threadIdx.x))] = dt[(((((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + (k * 64)) + ((int)threadIdx.x)) + 128)];
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_17 = 0; i_17 < 32; ++i_17) {
      uint1 __12;
      float2 __13;
        float2 __14;
        uint1 v__10 = *(uint1*)(cb_local + (i_17 * 2));
        ((float2*)(&__14))[0] = __half22float2(((half2*)(&v__10))[0]);
        float2 v__11 = *(float2*)(dt_local + (((i_17 >> 3) * 4) + (((i_17 & 3) >> 1) * 2)));
        *(float2*)(&(__13.x)) = tl::mul2(*(float2*)(&(__14.x)), *(float2*)(&(v__11.x)));
      ((half2*)(&__12))[0] = __float22half2_rn(((float2*)(&__13))[0]);
      *(uint1*)(cb_local + (i_17 * 2)) = __12;
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 8; ++i_18) {
      for (int vec_s = 0; vec_s < 8; ++vec_s) {
        half_t condval;
        if (((((((k * 64) + ((i_18 >> 1) * 16)) + ((vec_s >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (vec_s & 1)) <= (((((((int)blockIdx.y) * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_18 & 1) * 16)) + (((vec_s & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval = cb_local[((i_18 * 8) + vec_s)];
        } else {
          condval = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local[((i_18 * 8) + vec_s)] = condval;
      }
    }
    tl::cp_async_wait<7>();
    __syncthreads();
    {
      half_t B_local_1[16];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        #pragma unroll
        for (int i_19 = 0; i_19 < 2; ++i_19) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)x_shared)[((((((k * 4096) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_19) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_19 * 8)])));
        }
        for (int i_20 = 0; i_20 < 2; ++i_20) {
          for (int j_1 = 0; j_1 < 2; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_20 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(cb_local + ((ki_1 * 16) + (i_20 * 8))), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_20 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(cb_local + ((ki_1 * 16) + (i_20 * 8))), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_21 = 0; i_21 < 4; ++i_21) {
      tl::cp_async_gs<16>((&(((half_t*)x_shared)[((((((k * 4096) + (i_21 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(x[(((((((((((int)blockIdx.z) & 63) * 8388608) + ((((int)blockIdx.z) >> 6) * 1048576)) + (k * 262144)) + (i_21 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 524288)])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<7>();
  __syncthreads();
  if (1 <= ((int)blockIdx.y)) {
    #pragma unroll
    for (int i_22 = 0; i_22 < 32; ++i_22) {
      *(uint1*)(cb_local + (i_22 * 2)) = *(uint1*)(((half_t*)cb_shared) + (((((((((((((int)blockIdx.y) + 1) & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (((i_22 & 7) >> 2) * 1024)) + ((i_22 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((i_22 >> 3) * 16) + (((i_22 & 3) >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_22 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_22 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
  }
  tl::cp_async_wait<6>();
  __syncthreads();
  if (1 <= ((int)blockIdx.y)) {
    #pragma unroll
    for (int i_23 = 0; i_23 < 8; ++i_23) {
      half_t dA_cs_k_shared_local_cast_2[2];
      *(uint1*)(dA_cs_k_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)dA_cs_k_shared) + (((((((int)blockIdx.y) + 1) & 1) * 64) + (i_23 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __15;
      uint1 v__12 = *(uint1*)(dA_cs_k_shared_local_cast_2 + 0);
      ((float2*)(&__15))[0] = __half22float2(((half2*)(&v__12))[0]);
      *(float2*)(dA_cs_k_local + (i_23 * 2)) = __15;
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 32; ++i_24) {
      float broadcast_var_3 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
      uint1 __16;
      float2 __17;
        float2 __18;
        uint1 v__13 = *(uint1*)(cb_local + (i_24 * 2));
        ((float2*)(&__18))[0] = __half22float2(((half2*)(&v__13))[0]);
        float2 __19;
        float2 __20;
          float2 v__14 = make_float2((dA_cs_m_local[((((i_24 & 7) >> 2) * 2) + (i_24 & 1))] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_m_local[((((i_24 & 7) >> 2) * 2) + (i_24 & 1))] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
          float2 __21;
            float2 v__15 = *(float2*)(dA_cs_k_local + (((i_24 >> 3) * 4) + (((i_24 & 3) >> 1) * 2)));
            float2 v__16 = make_float2(broadcast_var_3, broadcast_var_3);
            *(float2*)(&(__21.x)) = tl::mul2(*(float2*)(&(v__15.x)), *(float2*)(&(v__16.x)));
          *(float2*)(&(__20.x)) = tl::sub2(*(float2*)(&(v__14.x)), *(float2*)(&(__21.x)));
        __19.x = exp2f(__20.x);
        __19.y = exp2f(__20.y);
        *(float2*)(&(__17.x)) = tl::mul2(*(float2*)(&(__18.x)), *(float2*)(&(__19.x)));
      ((half2*)(&__16))[0] = __float22half2_rn(((float2*)(&__17))[0]);
      *(uint1*)(cb_local + (i_24 * 2)) = __16;
    }
  }
  tl::cp_async_wait<5>();
  __syncthreads();
  if (1 <= ((int)blockIdx.y)) {
    #pragma unroll
    for (int i_25 = 0; i_25 < 8; ++i_25) {
      half_t dt_shared_local_cast_3[2];
      *(uint1*)(dt_shared_local_cast_3 + 0) = *(uint1*)(((half_t*)dt_shared) + (((((((int)blockIdx.y) + 1) & 1) * 64) + (i_25 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __22;
      uint1 v__17 = *(uint1*)(dt_shared_local_cast_3 + 0);
      ((float2*)(&__22))[0] = __half22float2(((half2*)(&v__17))[0]);
      *(float2*)(dt_local + (i_25 * 2)) = __22;
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 32; ++i_26) {
      uint1 __23;
      float2 __24;
        float2 __25;
        uint1 v__18 = *(uint1*)(cb_local + (i_26 * 2));
        ((float2*)(&__25))[0] = __half22float2(((half2*)(&v__18))[0]);
        float2 v__19 = *(float2*)(dt_local + (((i_26 >> 3) * 4) + (((i_26 & 3) >> 1) * 2)));
        *(float2*)(&(__24.x)) = tl::mul2(*(float2*)(&(__25.x)), *(float2*)(&(v__19.x)));
      ((half2*)(&__23))[0] = __float22half2_rn(((float2*)(&__24))[0]);
      *(uint1*)(cb_local + (i_26 * 2)) = __23;
    }
  }
  tl::cp_async_wait<4>();
  __syncthreads();
  if (1 <= ((int)blockIdx.y)) {
    {
      half_t B_local_2[16];
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        #pragma unroll
        for (int i_27 = 0; i_27 < 2; ++i_27) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)x_shared)[((((((((((int)blockIdx.y) + 1) & 1) * 4096) + (ki_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_27) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[(i_27 * 8)])));
        }
        for (int i_28 = 0; i_28 < 2; ++i_28) {
          for (int j_2 = 0; j_2 < 2; ++j_2) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_28 * 16) + (j_2 * 8))), reinterpret_cast<const unsigned*>(cb_local + ((ki_2 * 16) + (i_28 * 8))), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_28 * 16) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(cb_local + ((ki_2 * 16) + (i_28 * 8))), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
          }
        }
      }
    }
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  #pragma unroll
  for (int i_29 = 0; i_29 < 32; ++i_29) {
    *(uint1*)(cb_local + (i_29 * 2)) = *(uint1*)(((half_t*)cb_shared) + ((((((((((((int)blockIdx.y) & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (((i_29 & 7) >> 2) * 1024)) + ((i_29 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((i_29 >> 3) * 16) + (((i_29 & 3) >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_29 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_29 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  #pragma unroll
  for (int i_30 = 0; i_30 < 8; ++i_30) {
    half_t dA_cs_k_shared_local_cast_4[2];
    *(uint1*)(dA_cs_k_shared_local_cast_4 + 0) = *(uint1*)(((half_t*)dA_cs_k_shared) + ((((((int)blockIdx.y) & 1) * 64) + (i_30 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __26;
    uint1 v__20 = *(uint1*)(dA_cs_k_shared_local_cast_4 + 0);
    ((float2*)(&__26))[0] = __half22float2(((half2*)(&v__20))[0]);
    *(float2*)(dA_cs_k_local + (i_30 * 2)) = __26;
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 32; ++i_31) {
    float broadcast_var_4 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    uint1 __27;
    float2 __28;
      float2 __29;
      uint1 v__21 = *(uint1*)(cb_local + (i_31 * 2));
      ((float2*)(&__29))[0] = __half22float2(((half2*)(&v__21))[0]);
      float2 __30;
      float2 __31;
        float2 v__22 = make_float2((dA_cs_m_local[((((i_31 & 7) >> 2) * 2) + (i_31 & 1))] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_m_local[((((i_31 & 7) >> 2) * 2) + (i_31 & 1))] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
        float2 __32;
          float2 v__23 = *(float2*)(dA_cs_k_local + (((i_31 >> 3) * 4) + (((i_31 & 3) >> 1) * 2)));
          float2 v__24 = make_float2(broadcast_var_4, broadcast_var_4);
          *(float2*)(&(__32.x)) = tl::mul2(*(float2*)(&(v__23.x)), *(float2*)(&(v__24.x)));
        *(float2*)(&(__31.x)) = tl::sub2(*(float2*)(&(v__22.x)), *(float2*)(&(__32.x)));
      __30.x = exp2f(__31.x);
      __30.y = exp2f(__31.y);
      *(float2*)(&(__28.x)) = tl::mul2(*(float2*)(&(__29.x)), *(float2*)(&(__30.x)));
    ((half2*)(&__27))[0] = __float22half2_rn(((float2*)(&__28))[0]);
    *(uint1*)(cb_local + (i_31 * 2)) = __27;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_32 = 0; i_32 < 8; ++i_32) {
    half_t dt_shared_local_cast_5[2];
    *(uint1*)(dt_shared_local_cast_5 + 0) = *(uint1*)(((half_t*)dt_shared) + ((((((int)blockIdx.y) & 1) * 64) + (i_32 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __33;
    uint1 v__25 = *(uint1*)(dt_shared_local_cast_5 + 0);
    ((float2*)(&__33))[0] = __half22float2(((half2*)(&v__25))[0]);
    *(float2*)(dt_local + (i_32 * 2)) = __33;
  }
  #pragma unroll
  for (int i_33 = 0; i_33 < 32; ++i_33) {
    uint1 __34;
    float2 __35;
      float2 __36;
      uint1 v__26 = *(uint1*)(cb_local + (i_33 * 2));
      ((float2*)(&__36))[0] = __half22float2(((half2*)(&v__26))[0]);
      float2 v__27 = *(float2*)(dt_local + (((i_33 >> 3) * 4) + (((i_33 & 3) >> 1) * 2)));
      *(float2*)(&(__35.x)) = tl::mul2(*(float2*)(&(__36.x)), *(float2*)(&(v__27.x)));
    ((half2*)(&__34))[0] = __float22half2_rn(((float2*)(&__35))[0]);
    *(uint1*)(cb_local + (i_33 * 2)) = __34;
  }
  #pragma unroll
  for (int i_34 = 0; i_34 < 64; ++i_34) {
    half_t condval_1;
    if (((((((i_34 >> 4) * 16) + (((i_34 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_34 & 1)) <= ((((((((int)threadIdx.x) & 63) >> 5) * 32) + (((i_34 & 15) >> 3) * 16)) + (((i_34 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
      condval_1 = cb_local[i_34];
    } else {
      condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    cb_local[i_34] = condval_1;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t B_local_3[16];
    for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
      #pragma unroll
      for (int i_35 = 0; i_35 < 2; ++i_35) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)x_shared)[(((((((((int)blockIdx.y) & 1) * 4096) + (ki_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_35) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[(i_35 * 8)])));
      }
      for (int i_36 = 0; i_36 < 2; ++i_36) {
        for (int j_3 = 0; j_3 < 2; ++j_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_36 * 16) + (j_3 * 8))), reinterpret_cast<const unsigned*>(cb_local + ((ki_3 * 16) + (i_36 * 8))), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_36 * 16) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(cb_local + ((ki_3 * 16) + (i_36 * 8))), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
        }
      }
    }
  }
  D_local[0] = ((float)D[((int)blockIdx.x)]);
  __syncthreads();
  #pragma unroll
  for (int i_37 = 0; i_37 < 4; ++i_37) {
    *(uint4*)(((half_t*)x_residual_shared) + (((((i_37 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(x + ((((((((((int)blockIdx.z) & 63) * 8388608) + ((((int)blockIdx.z) >> 6) * 1048576)) + (((int)blockIdx.y) * 262144)) + (i_37 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_38 = 0; i_38 < 16; ++i_38) {
    half_t x_residual_shared_local_cast_6[2];
    *(uint1*)(x_residual_shared_local_cast_6 + 0) = *(uint1*)(((half_t*)x_residual_shared) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_38 >> 3) * 1024)) + ((i_38 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_38 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_38 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __37;
    uint1 v__28 = *(uint1*)(x_residual_shared_local_cast_6 + 0);
    ((float2*)(&__37))[0] = __half22float2(((half2*)(&v__28))[0]);
    *(float2*)(x_residual_local + (i_38 * 2)) = __37;
  }
  #pragma unroll
  for (int i_39 = 0; i_39 < 8; ++i_39) {
    float4 __38;
      float4 v__29 = *(float4*)(x_residual_local + (i_39 * 4));
      float4 v__30 = make_float4(D_local[0], D_local[0], D_local[0], D_local[0]);
      float4 v__31 = *(float4*)(acc_o + (i_39 * 4));
      *(float2*)(&(__38.x)) = tl::fma2(*(float2*)(&(v__29.x)), *(float2*)(&(v__30.x)), *(float2*)(&(v__31.x)));
      *(float2*)(&(__38.z)) = tl::fma2(*(float2*)(&(v__29.z)), *(float2*)(&(v__30.z)), *(float2*)(&(v__31.z)));
    *(float4*)(acc_o + (i_39 * 4)) = __38;
  }
  __syncthreads();
  #pragma unroll
  for (int i_40 = 0; i_40 < 4; ++i_40) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_o_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_40 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_40 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((half_t)acc_o[(i_40 * 8)]), ((half_t)acc_o[((i_40 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_40 * 8) + 2)]), ((half_t)acc_o[((i_40 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_40 * 8) + 4)]), ((half_t)acc_o[((i_40 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_40 * 8) + 6)]), ((half_t)acc_o[((i_40 * 8) + 7)])));
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store(Output_desc, (&(((half_t*)acc_o_shared)[0])), 0, (((((int)blockIdx.z) >> 6) * 256) + (((int)blockIdx.y) * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 63));
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
}

