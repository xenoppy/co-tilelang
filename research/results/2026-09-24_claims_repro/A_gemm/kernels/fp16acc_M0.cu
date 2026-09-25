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

extern "C" __global__ void gemm_autotune_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc);
extern "C" __global__ void __launch_bounds__(384, 1) gemm_autotune_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* C_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  half_t C_local[128];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(C_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(256);
    mbarrier[3].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 128; ++k) {
      mbarrier[((k & 1) + 2)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k & 1)].expect_transaction(16384);
        tl::tma_load(A_desc, mbarrier[(k & 1)], (&(((half_t*)A_shared)[((k & 1) * 8192)])), (k * 64), (((int)blockIdx.y) * 128));
        mbarrier[(k & 1)].arrive_and_expect_tx(32768);
        tl::tma_load(B_desc, mbarrier[(k & 1)], (&(((half_t*)B_shared)[((k & 1) * 16384)])), (k * 64), (((int)blockIdx.x) * 256));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
      *(uint2*)(C_local + (i * 4)) = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
    }
    for (int k_1 = 0; k_1 < 128; ++k_1) {
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      {
        half_t A_local[32];
        half_t B_local[32];
        for (int ki = 0; ki < 4; ++ki) {
          #pragma unroll
          for (int i_1 = 0; i_1 < 4; ++i_1) {
            tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((k_1 & 1) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (i_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_1 * 8)])));
          }
          #pragma unroll
          for (int i_2 = 0; i_2 < 4; ++i_2) {
            tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[(((((((((k_1 & 1) * 16384) + ((((((int)threadIdx.x) >> 6) + 2) & 3) * 4096)) + (i_2 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_2 * 8)])));
          }
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(C_local + ((i_3 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(C_local + (((i_3 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      mbarrier[((k_1 & 1) + 2)].arrive();
    }
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)C_shared)[((((((((((int)threadIdx.x) >> 5) * 4096) + ((i_4 >> 2) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 16384)])), __pack_half2(C_local[(i_4 * 8)], C_local[((i_4 * 8) + 1)]), __pack_half2(C_local[((i_4 * 8) + 2)], C_local[((i_4 * 8) + 3)]), __pack_half2(C_local[((i_4 * 8) + 4)], C_local[((i_4 * 8) + 5)]), __pack_half2(C_local[((i_4 * 8) + 6)], C_local[((i_4 * 8) + 7)]));
    }
    tl::__sync_thread_partial(3, 256);
    if (tl::tl_shuffle_elect<256>()) {
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        tl::tma_store(C_desc, (&(((half_t*)C_shared)[(i_5 * 8192)])), ((((int)blockIdx.x) * 256) + (i_5 * 64)), (((int)blockIdx.y) * 128));
      }
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

