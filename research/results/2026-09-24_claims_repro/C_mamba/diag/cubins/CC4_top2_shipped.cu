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
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void chunk_scan_fwd_kernel(const half_t* __restrict__ C, const half_t* __restrict__ D, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap cb_desc, const half_t* __restrict__ dA_cumsum, const half_t* __restrict__ dt, const half_t* __restrict__ prev_states, const half_t* __restrict__ x, __grid_constant__ const CUtensorMap x_desc);
extern "C" __global__ void __launch_bounds__(256, 1) chunk_scan_fwd_kernel(const half_t* __restrict__ C, const half_t* __restrict__ D, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap cb_desc, const half_t* __restrict__ dA_cumsum, const half_t* __restrict__ dt, const half_t* __restrict__ prev_states, const half_t* __restrict__ x, __grid_constant__ const CUtensorMap x_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* acc_o_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* dA_cs_m_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* x_residual_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* C_shared = ((void*)((char*)buf_dyn_shmem + 128));
  void* prev_state_shared = ((void*)((char*)buf_dyn_shmem + 16512));
  void* cb_shared = ((void*)((char*)buf_dyn_shmem + 33792));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 50176));
  void* dA_cs_k_shared = ((void*)((char*)buf_dyn_shmem + 66560));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 67584));
  __shared__ __align__(16) uint64_t mbarrier_mem[16];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float dA_cs_m_local[4];
  float acc_o[32];
  float scale_m_local[4];
  float D_local[1];
  float x_residual_local[32];
  half_t cb_local[64];
  float dA_cs_k_local[16];
  float dt_local[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(cb_desc);
    tl::prefetch_tma_descriptor(x_desc);
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(1);
    mbarrier[4].init(1);
    mbarrier[5].init(1);
    mbarrier[6].init(1);
    mbarrier[7].init(1);
    mbarrier[8].init(128);
    mbarrier[9].init(128);
    mbarrier[10].init(128);
    mbarrier[11].init(128);
    mbarrier[12].init(128);
    mbarrier[13].init(128);
    mbarrier[14].init(128);
    mbarrier[15].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((half_t*)dA_cs_m_shared)[((int)threadIdx.x)] = dA_cumsum[((((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))];
  }
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint4*)(((half_t*)C_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(C + ((((((((int)blockIdx.z) & 63) * 262144) + ((((int)blockIdx.z) >> 6) * 32768)) + (((int)blockIdx.y) * 8192)) + (i * 2048)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(uint4*)(((half_t*)prev_state_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(prev_states + ((((((((int)blockIdx.z) & 63) * 4194304) + ((((int)blockIdx.z) >> 6) * 524288)) + (((int)blockIdx.x) * 8192)) + (i_1 * 2048)) + (((int)threadIdx.x) * 8)));
  }
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::fence_proxy_async();
    for (int k = 0; k < (((int)blockIdx.y) + 1); ++k) {
      mbarrier[((k & 1) + 8)].wait(((k >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k & 1)].arrive_and_expect_tx(8192);
        tl::tma_load(cb_desc, mbarrier[(k & 1)], (&(((half_t*)cb_shared)[((k & 1) * 4096)])), (k * 64), (((int)blockIdx.y) * 64), 0, (((int)blockIdx.z) >> 6), (((int)blockIdx.z) & 63));
      }
      mbarrier[((k & 1) + 10)].wait(((k >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((k & 1) + 2)].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dA_cs_k_shared)[((k & 1) * 64)])), (&(dA_cumsum[(((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + (k * 64))])), mbarrier[((k & 1) + 2)], 128);
      }
      mbarrier[((k & 1) + 12)].wait(((k >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((k & 1) + 4)].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dt_shared)[((k & 1) * 64)])), (&(dt[(((((((int)blockIdx.z) & 63) * 131072) + (((int)blockIdx.x) * 2048)) + ((((int)blockIdx.z) >> 6) * 256)) + (k * 64))])), mbarrier[((k & 1) + 4)], 128);
      }
      mbarrier[((k & 1) + 14)].wait(((k >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((k & 1) + 6)].arrive_and_expect_tx(8192);
        tl::tma_load(x_desc, mbarrier[((k & 1) + 6)], (&(((half_t*)x_shared)[((k & 1) * 4096)])), 0, (((((int)blockIdx.z) >> 6) * 256) + (k * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 63));
      }
    }
  } else {
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      dA_cs_m_local[i_2] = ((float)((half_t*)dA_cs_m_shared)[(((((((int)threadIdx.x) & 63) >> 5) * 32) + (i_2 * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_3 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
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
    {
      half_t A_local[16];
      half_t B_local[16];
      for (int ki = 0; ki < 8; ++ki) {
        #pragma unroll
        for (int i_4 = 0; i_4 < 2; ++i_4) {
          tl::ptx_ldmatrix_x4((&(((half_t*)C_shared)[((((((((ki >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_4 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_4 * 8)])));
        }
        #pragma unroll
        for (int i_5 = 0; i_5 < 2; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((half_t*)prev_state_shared)[(((((((((ki >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (i_5 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_5 * 8)])));
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
    for (int k_1 = 0; k_1 < (((int)blockIdx.y) + 1); ++k_1) {
      mbarrier[(k_1 & 1)].wait((k_1 >> 1));
      #pragma unroll
      for (int i_8 = 0; i_8 < 32; ++i_8) {
        *(uint1*)(cb_local + (i_8 * 2)) = *(uint1*)(((half_t*)cb_shared) + ((((((((((k_1 & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (((i_8 & 7) >> 2) * 1024)) + ((i_8 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((i_8 >> 3) * 16) + (((i_8 & 3) >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_8 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_8 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      mbarrier[((k_1 & 1) + 8)].arrive();
      mbarrier[((k_1 & 1) + 2)].wait((k_1 >> 1));
      #pragma unroll
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        half_t dA_cs_k_shared_local_cast[2];
        *(uint1*)(dA_cs_k_shared_local_cast + 0) = *(uint1*)(((half_t*)dA_cs_k_shared) + ((((k_1 & 1) * 64) + (i_9 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __4;
        uint1 v__4 = *(uint1*)(dA_cs_k_shared_local_cast + 0);
        ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__4))[0]);
        *(float2*)(dA_cs_k_local + (i_9 * 2)) = __4;
      }
      mbarrier[((k_1 & 1) + 10)].arrive();
      #pragma unroll
      for (int i_10 = 0; i_10 < 32; ++i_10) {
        float broadcast_var_2 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
        uint1 __5;
        float2 __6;
          float2 __7;
          uint1 v__5 = *(uint1*)(cb_local + (i_10 * 2));
          ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__5))[0]);
          float2 __8;
          float2 __9;
            float2 v__6 = make_float2((dA_cs_m_local[((((i_10 & 7) >> 2) * 2) + (i_10 & 1))] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_m_local[((((i_10 & 7) >> 2) * 2) + (i_10 & 1))] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
            float2 __10;
              float2 v__7 = *(float2*)(dA_cs_k_local + (((i_10 >> 3) * 4) + (((i_10 & 3) >> 1) * 2)));
              float2 v__8 = make_float2(broadcast_var_2, broadcast_var_2);
              *(float2*)(&(__10.x)) = tl::mul2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)));
            *(float2*)(&(__9.x)) = tl::sub2(*(float2*)(&(v__6.x)), *(float2*)(&(__10.x)));
          __8.x = exp2f(__9.x);
          __8.y = exp2f(__9.y);
          *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(__7.x)), *(float2*)(&(__8.x)));
        ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&__6))[0]);
        *(uint1*)(cb_local + (i_10 * 2)) = __5;
      }
      mbarrier[((k_1 & 1) + 4)].wait((k_1 >> 1));
      #pragma unroll
      for (int i_11 = 0; i_11 < 8; ++i_11) {
        half_t dt_shared_local_cast_1[2];
        *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + ((((k_1 & 1) * 64) + (i_11 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __11;
        uint1 v__9 = *(uint1*)(dt_shared_local_cast_1 + 0);
        ((float2*)(&__11))[0] = __half22float2(((half2*)(&v__9))[0]);
        *(float2*)(dt_local + (i_11 * 2)) = __11;
      }
      mbarrier[((k_1 & 1) + 12)].arrive();
      #pragma unroll
      for (int i_12 = 0; i_12 < 32; ++i_12) {
        uint1 __12;
        float2 __13;
          float2 __14;
          uint1 v__10 = *(uint1*)(cb_local + (i_12 * 2));
          ((float2*)(&__14))[0] = __half22float2(((half2*)(&v__10))[0]);
          float2 v__11 = *(float2*)(dt_local + (((i_12 >> 3) * 4) + (((i_12 & 3) >> 1) * 2)));
          *(float2*)(&(__13.x)) = tl::mul2(*(float2*)(&(__14.x)), *(float2*)(&(v__11.x)));
        ((half2*)(&__12))[0] = __float22half2_rn(((float2*)(&__13))[0]);
        *(uint1*)(cb_local + (i_12 * 2)) = __12;
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 64; ++i_13) {
        half_t condval;
        if (((((((k_1 * 64) + ((i_13 >> 4) * 16)) + (((i_13 & 7) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_13 & 1)) <= (((((((int)blockIdx.y) * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + (((i_13 & 15) >> 3) * 16)) + (((i_13 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval = cb_local[i_13];
        } else {
          condval = half_t(0x0p+0f/*0.000000e+00*/);
        }
        cb_local[i_13] = condval;
      }
      mbarrier[((k_1 & 1) + 6)].wait((k_1 >> 1));
      {
        half_t B_local_1[16];
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          #pragma unroll
          for (int i_14 = 0; i_14 < 2; ++i_14) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)x_shared)[(((((((k_1 & 1) * 4096) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_14 * 8)])));
          }
          for (int i_15 = 0; i_15 < 2; ++i_15) {
            for (int j_1 = 0; j_1 < 2; ++j_1) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_15 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(cb_local + ((ki_1 * 16) + (i_15 * 8))), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_15 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(cb_local + ((ki_1 * 16) + (i_15 * 8))), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
            }
          }
        }
      }
      mbarrier[((k_1 & 1) + 14)].arrive();
    }
    D_local[0] = ((float)D[((int)blockIdx.x)]);
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_16 = 0; i_16 < 4; ++i_16) {
      *(uint4*)(((half_t*)x_residual_shared) + ((((((i_16 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1024)) = *(uint4*)(x + (((((((((((int)blockIdx.z) & 63) * 8388608) + ((((int)blockIdx.z) >> 6) * 1048576)) + (((int)blockIdx.y) * 262144)) + (i_16 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) - 65536));
    }
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_17 = 0; i_17 < 16; ++i_17) {
      half_t x_residual_shared_local_cast_2[2];
      *(uint1*)(x_residual_shared_local_cast_2 + 0) = *(uint1*)(((half_t*)x_residual_shared) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_17 >> 3) * 1024)) + ((i_17 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_17 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_17 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __15;
      uint1 v__12 = *(uint1*)(x_residual_shared_local_cast_2 + 0);
      ((float2*)(&__15))[0] = __half22float2(((half2*)(&v__12))[0]);
      *(float2*)(x_residual_local + (i_17 * 2)) = __15;
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 8; ++i_18) {
      float4 __16;
        float4 v__13 = *(float4*)(x_residual_local + (i_18 * 4));
        float4 v__14 = make_float4(D_local[0], D_local[0], D_local[0], D_local[0]);
        float4 v__15 = *(float4*)(acc_o + (i_18 * 4));
        *(float2*)(&(__16.x)) = tl::fma2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)), *(float2*)(&(v__15.x)));
        *(float2*)(&(__16.z)) = tl::fma2(*(float2*)(&(v__13.z)), *(float2*)(&(v__14.z)), *(float2*)(&(v__15.z)));
      *(float4*)(acc_o + (i_18 * 4)) = __16;
    }
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_19 = 0; i_19 < 4; ++i_19) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_o_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_19 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((half_t)acc_o[(i_19 * 8)]), ((half_t)acc_o[((i_19 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_19 * 8) + 2)]), ((half_t)acc_o[((i_19 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_19 * 8) + 4)]), ((half_t)acc_o[((i_19 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_19 * 8) + 6)]), ((half_t)acc_o[((i_19 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::fence_proxy_async();
      tl::tma_store(Output_desc, (&(((half_t*)acc_o_shared)[0])), 0, (((((int)blockIdx.z) >> 6) * 256) + (((int)blockIdx.y) * 64)), ((int)blockIdx.x), (((int)blockIdx.z) & 63));
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

