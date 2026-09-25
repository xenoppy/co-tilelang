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
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* C_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float C_local[128];
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
    for (int k = 0; k < 64; ++k) {
      mbarrier[((k & 1) + 2)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k & 1)].expect_transaction(16384);
        tl::tma_load(A_desc, mbarrier[(k & 1)], (&(((fp8_e4_t*)A_shared)[((k & 1) * 16384)])), (k * 128), (((int)blockIdx.y) * 128));
        mbarrier[(k & 1)].arrive_and_expect_tx(32768);
        tl::tma_load(B_desc, mbarrier[(k & 1)], (&(((fp8_e4_t*)B_shared)[((k & 1) * 32768)])), (k * 128), (((int)blockIdx.x) * 256));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 64; ++k_1) {
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      {
        fp8_e4_t A_local[64];
        fp8_e4_t B_local[64];
        for (int ki = 0; ki < 4; ++ki) {
          #pragma unroll
          for (int i_1 = 0; i_1 < 4; ++i_1) {
            tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_shared)[((((((((k_1 & 1) * 16384) + (((((int)threadIdx.x) & 63) >> 5) * 8192)) + (i_1 * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A_local[(i_1 * 16)])));
          }
          #pragma unroll
          for (int i_2 = 0; i_2 < 4; ++i_2) {
            tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)B_shared)[(((((((((k_1 & 1) * 32768) + ((((((int)threadIdx.x) >> 6) + 2) & 3) * 8192)) + (i_2 * 2048)) + (((((int)threadIdx.x) & 31) >> 4) * 1024)) + ((((int)threadIdx.x) & 7) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(B_local[(i_2 * 16)])));
          }
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_local + ((i_3 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_3 * 16)), reinterpret_cast<const unsigned*>(B_local + (j * 16)));
              tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_local + (((i_3 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 16)), reinterpret_cast<const unsigned*>(B_local + ((j * 16) + 8)));
            }
          }
        }
      }
      mbarrier[((k_1 & 1) + 2)].arrive();
    }
    tl::__sync_thread_partial(3, 256);
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      fp8_e4_t C_shared_local_cast[2];
      fp8_e4_2_t __1;
      float2 v_ = *(float2*)(C_local + (i_4 * 2));
      (reinterpret_cast<__nv_fp8x2_storage_t*>(&__1))[0] = __nv_cvt_float2_to_fp8x2(((float2*)(&v_))[0], __NV_SATFINITE, __NV_E4M3);
      *(fp8_e4_2_t*)(C_shared_local_cast + 0) = __1;
      *(fp8_e4_2_t*)(((fp8_e4_t*)C_shared) + (((((((((((((((((int)threadIdx.x) >> 6) * 64) + (((i_4 & 15) >> 1) * 8)) >> 7) * 16384) + (((((int)threadIdx.x) & 63) >> 5) * 8192)) + ((i_4 >> 4) * 2048)) + ((i_4 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 64)) + (((((i_4 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 32)) + (((((i_4 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 16384)) = *(fp8_e4_2_t*)(C_shared_local_cast + 0);
    }
    tl::__sync_thread_partial(3, 256);
    if (tl::tl_shuffle_elect<256>()) {
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        tl::tma_store(C_desc, (&(((fp8_e4_t*)C_shared)[(i_5 * 16384)])), ((((int)blockIdx.x) * 256) + (i_5 * 128)), (((int)blockIdx.y) * 128));
      }
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
  }
}

