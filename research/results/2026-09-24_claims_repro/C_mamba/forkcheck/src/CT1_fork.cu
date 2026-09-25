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

extern "C" __global__ void chunk_state_fwd_kernel(__grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ Output, const half_t* __restrict__ dA_cumsum, const half_t* __restrict__ dt, __grid_constant__ const CUtensorMap x_desc);
extern "C" __global__ void __launch_bounds__(256, 1) chunk_state_fwd_kernel(__grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ Output, const half_t* __restrict__ dA_cumsum, const half_t* __restrict__ dt, __grid_constant__ const CUtensorMap x_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* acc_o_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* dA_cumsum_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  void* dt_shared = ((void*)((char*)buf_dyn_shmem + 99328));
  __shared__ __align__(16) uint64_t mbarrier_mem[32];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float dA_cs_last[1];
  float acc_o[64];
  float dA_cumsum_local[16];
  float dt_local[16];
  float scale[16];
  half_t x_local[64];
  half_t xt_local[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(x_desc);
    tl::prefetch_tma_descriptor(B_desc);
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
    mbarrier[8].init(1);
    mbarrier[9].init(1);
    mbarrier[10].init(1);
    mbarrier[11].init(1);
    mbarrier[12].init(1);
    mbarrier[13].init(1);
    mbarrier[14].init(1);
    mbarrier[15].init(1);
    mbarrier[16].init(128);
    mbarrier[17].init(128);
    mbarrier[18].init(128);
    mbarrier[19].init(128);
    mbarrier[20].init(128);
    mbarrier[21].init(128);
    mbarrier[22].init(128);
    mbarrier[23].init(128);
    mbarrier[24].init(128);
    mbarrier[25].init(128);
    mbarrier[26].init(128);
    mbarrier[27].init(128);
    mbarrier[28].init(128);
    mbarrier[29].init(128);
    mbarrier[30].init(128);
    mbarrier[31].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 4; ++k) {
      mbarrier[(k + 16)].wait(1);
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[k].arrive_and_expect_tx(8192);
        tl::tma_load(x_desc, mbarrier[k], (&(((half_t*)x_shared)[(k * 4096)])), 0, ((((int)blockIdx.z) * 256) + (k * 64)), ((int)blockIdx.x), 0);
      }
      mbarrier[(k + 20)].wait(1);
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k + 4)].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dA_cumsum_shared)[(k * 64)])), (&(dA_cumsum[(((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + (k * 64))])), mbarrier[(k + 4)], 128);
      }
      mbarrier[(k + 24)].wait(1);
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k + 8)].arrive_and_expect_tx(128);
        tl::tma_load((&(((half_t*)dt_shared)[(k * 64)])), (&(dt[(((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + (k * 64))])), mbarrier[(k + 8)], 128);
      }
      mbarrier[(k + 28)].wait(1);
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k + 12)].arrive_and_expect_tx(16384);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
          tl::tma_load(B_desc, mbarrier[(k + 12)], (&(((half_t*)B_shared)[((k * 8192) + (i * 4096))])), (i * 64), ((((int)blockIdx.z) * 256) + (k * 64)), 0, 0);
        }
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    dA_cs_last[0] = ((float)dA_cumsum[(((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 256)) + 255)]);
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 4; ++k_1) {
      mbarrier[(k_1 + 4)].wait(0);
      #pragma unroll
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        half_t dA_cumsum_shared_local_cast[2];
        *(uint1*)(dA_cumsum_shared_local_cast + 0) = *(uint1*)(((half_t*)dA_cumsum_shared) + (((k_1 * 64) + (i_2 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __1;
        uint1 v_ = *(uint1*)(dA_cumsum_shared_local_cast + 0);
        ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
        *(float2*)(dA_cumsum_local + (i_2 * 2)) = __1;
      }
      mbarrier[(k_1 + 20)].arrive();
      mbarrier[(k_1 + 8)].wait(0);
      #pragma unroll
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        half_t dt_shared_local_cast_1[2];
        *(uint1*)(dt_shared_local_cast_1 + 0) = *(uint1*)(((half_t*)dt_shared) + (((k_1 * 64) + (i_3 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        float2 __2;
        uint1 v__1 = *(uint1*)(dt_shared_local_cast_1 + 0);
        ((float2*)(&__2))[0] = __half22float2(((half2*)(&v__1))[0]);
        *(float2*)(dt_local + (i_3 * 2)) = __2;
      }
      mbarrier[(k_1 + 24)].arrive();
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        float broadcast_var_1 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
        float4 __3;
          float4 __4;
          float4 __5;
            float4 v__2 = make_float4((dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (dA_cs_last[0] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
            float4 __6;
              float4 v__3 = *(float4*)(dA_cumsum_local + (i_4 * 4));
              float4 v__4 = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
              *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(v__3.x)), *(float2*)(&(v__4.x)));
              *(float2*)(&(__6.z)) = tl::mul2(*(float2*)(&(v__3.z)), *(float2*)(&(v__4.z)));
            *(float2*)(&(__5.x)) = tl::sub2(*(float2*)(&(v__2.x)), *(float2*)(&(__6.x)));
            *(float2*)(&(__5.z)) = tl::sub2(*(float2*)(&(v__2.z)), *(float2*)(&(__6.z)));
          __4.x = exp2f(__5.x);
          __4.y = exp2f(__5.y);
          __4.z = exp2f(__5.z);
          __4.w = exp2f(__5.w);
          float4 v__5 = *(float4*)(dt_local + (i_4 * 4));
          *(float2*)(&(__3.x)) = tl::mul2(*(float2*)(&(__4.x)), *(float2*)(&(v__5.x)));
          *(float2*)(&(__3.z)) = tl::mul2(*(float2*)(&(__4.z)), *(float2*)(&(v__5.z)));
        *(float4*)(scale + (i_4 * 4)) = __3;
      }
      mbarrier[k_1].wait(0);
      #pragma unroll
      for (int i_5 = 0; i_5 < 64; ++i_5) {
        x_local[i_5] = ((half_t*)x_shared)[((((((((k_1 * 4096) + ((i_5 >> 3) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((i_5 & 7) >> 2) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_5 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_5 & 7) >> 2) + (i_5 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      mbarrier[(k_1 + 16)].arrive();
      #pragma unroll
      for (int i_6 = 0; i_6 < 64; ++i_6) {
        xt_local[i_6] = ((half_t)(((float)x_local[((((((i_6 >> 4) * 16) + (((i_6 & 7) >> 2) * 8)) + ((i_6 & 1) * 4)) + (((i_6 & 15) >> 3) * 2)) + ((i_6 & 3) >> 1))]) * scale[((((i_6 >> 4) * 4) + (((i_6 & 7) >> 2) * 2)) + (i_6 & 1))]));
      }
      mbarrier[(k_1 + 12)].wait(0);
      {
        half_t B_local[32];
        for (int ki = 0; ki < 4; ++ki) {
          #pragma unroll
          for (int i_7 = 0; i_7 < 4; ++i_7) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[(((((((k_1 * 8192) + (((((int)threadIdx.x) & 127) >> 6) * 4096)) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_7 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_7 * 8)])));
          }
          for (int i_8 = 0; i_8 < 2; ++i_8) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_8 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(xt_local + ((ki * 16) + (i_8 * 8))), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_8 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(xt_local + ((ki * 16) + (i_8 * 8))), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      mbarrier[(k_1 + 28)].arrive();
    }
    tl::__sync_thread_partial(3, 128);
    #pragma unroll
    for (int i_9 = 0; i_9 < 8; ++i_9) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)acc_o_shared)[(((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_9 >> 2) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + ((i_9 & 3) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) - 128)])), __pack_half2(((half_t)acc_o[(i_9 * 8)]), ((half_t)acc_o[((i_9 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_9 * 8) + 2)]), ((half_t)acc_o[((i_9 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_9 * 8) + 4)]), ((half_t)acc_o[((i_9 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_9 * 8) + 6)]), ((half_t)acc_o[((i_9 * 8) + 7)])));
    }
    tl::__sync_thread_partial(3, 128);
    if (tl::tl_shuffle_elect<128>()) {
      tl::fence_proxy_async();
      tl::tma_store((&(Output[((((int)blockIdx.z) * 524288) + (((int)blockIdx.x) * 8192))])), (&(((half_t*)acc_o_shared)[0])), 16384);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

