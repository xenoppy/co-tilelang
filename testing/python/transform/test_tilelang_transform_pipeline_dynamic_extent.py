"""Software pipelining of a cp.async loop whose trip count is only known at run time.

The trip count here depends on blockIdx (like the K/V loop of causal attention). The
pipeline epilogue covers [n, n + max_stage); its extent (n + max_stage) - n must be
simplified to the constant max_stage so InjectSoftwarePipeline expands the epilogue into
straight-line code. Before the fix it emitted a loop whose cp.async wait counts depended
on the loop variable (`ptx_wait_group(3 - k * 2)`), and CUDA codegen failed with
"Downcast from tirx.Sub to ir.IntImm failed". The GPU check covers trip counts smaller
than the pipeline depth (fully predicated prologue) as well.
"""

import tilelang
import tilelang.language as T
import tilelang.testing
import torch

BM, BN, BK = 64, 64, 32


def _kernel(num_blocks: int, num_stages: int):
    # C[b] = A[b, :, :k_len(b)] @ B[:, :k_len(b)]^T with k_len(b) = (b + 1) * BK
    K = num_blocks * BK

    @T.prim_func
    def main(A: T.Tensor((num_blocks * BM, K), "float16"), B: T.Tensor((BN, K), "float16"),
             C: T.Tensor((num_blocks * BM, BN), "float32")):
        with T.Kernel(num_blocks, threads=128) as bx:
            A_s = T.alloc_shared((BM, BK), "float16")
            B_s = T.alloc_shared((BN, BK), "float16")
            acc = T.alloc_fragment((BM, BN), "float32")
            T.clear(acc)
            for k in T.Pipelined(bx + 1, num_stages=num_stages):
                T.copy(A[bx * BM : (bx + 1) * BM, k * BK : (k + 1) * BK], A_s)
                T.copy(B[:, k * BK : (k + 1) * BK], B_s)
                T.gemm(A_s, B_s, acc, transpose_B=True)
            T.copy(acc, C[bx * BM : (bx + 1) * BM, :])

    return main


def _compile(num_blocks: int, num_stages: int):
    return tilelang.compile(_kernel(num_blocks, num_stages), out_idx=[2],
                            pass_configs={"tl.disable_warp_specialized": True})


def test_dynamic_trip_count_compiles():
    for stages in (2, 3):
        k = _compile(6, stages)
        src = k.get_kernel_source()
        assert "cp_async_wait" in src or "cp.async.wait_group" in src, src[:2000]


@tilelang.testing.requires_cuda
def test_dynamic_trip_count_correct():
    nb = 6
    for stages in (1, 2, 3, 4):
        k = _compile(nb, stages)
        A = torch.randn(nb * BM, nb * BK, device="cuda", dtype=torch.float16)
        B = torch.randn(BN, nb * BK, device="cuda", dtype=torch.float16)
        C = k(A, B)
        ref = torch.cat([A[b * BM : (b + 1) * BM, : (b + 1) * BK].float() @ B[:, : (b + 1) * BK].float().T
                         for b in range(nb)])
        torch.testing.assert_close(C, ref, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    test_dynamic_trip_count_compiles()
    if torch.cuda.is_available():
        test_dynamic_trip_count_correct()
    print("ok")
