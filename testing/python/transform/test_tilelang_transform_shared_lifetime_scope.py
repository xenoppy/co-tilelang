"""MergeSharedMemoryAllocations honours `tl.shared_lifetime_scope` AttrStmts.

Two shared buffers used in mutually exclusive branches of a persistent loop are both
live for the whole loop under TileLang's default plan (their allocations sit at kernel
scope), so the merged dynamic smem is the sum. Wrapping each use in a
`tl.shared_lifetime_scope` AttrStmt (no shared value crosses the scope boundary) lets
the planner overlay them: merged size = max. Correctness of the overlaid kernel is
checked on the GPU when one is available.
"""

import re

import tilelang
import tilelang.language as T
import tilelang.testing
import torch

N = 1024  # elements per buffer
ITERS = 8


def _kernel(scoped: bool):
    @T.prim_func
    def main(X: T.Tensor((ITERS, N), "float32"), Y: T.Tensor((ITERS, N), "float32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((2 * N,), "float32")
            tx = T.get_thread_binding()
            for it in T.serial(ITERS):
                T.sync_threads()
                if it % 2 == 0:
                    if scoped:
                        with T.attr(0, "tl.shared_lifetime_scope", 0):
                            for i in T.serial(N // 128):
                                a[i * 128 + tx] = X[it, i * 128 + tx] * 2.0
                            T.sync_threads()
                            for i in T.serial(N // 128):
                                Y[it, i * 128 + tx] = a[N - 1 - (i * 128 + tx)]
                    else:
                        for i in T.serial(N // 128):
                            a[i * 128 + tx] = X[it, i * 128 + tx] * 2.0
                        T.sync_threads()
                        for i in T.serial(N // 128):
                            Y[it, i * 128 + tx] = a[N - 1 - (i * 128 + tx)]
                else:
                    if scoped:
                        with T.attr(0, "tl.shared_lifetime_scope", 1):
                            for i in T.serial(2 * N // 128):
                                b[i * 128 + tx] = X[it, (i * 128 + tx) % N] + 1.0
                            T.sync_threads()
                            for i in T.serial(N // 128):
                                Y[it, i * 128 + tx] = b[2 * N - 1 - (i * 128 + tx)]
                    else:
                        for i in T.serial(2 * N // 128):
                            b[i * 128 + tx] = X[it, (i * 128 + tx) % N] + 1.0
                        T.sync_threads()
                        for i in T.serial(N // 128):
                            Y[it, i * 128 + tx] = b[2 * N - 1 - (i * 128 + tx)]

    return main


def _view_offsets(src: str) -> list[int]:
    # byte offsets of the buffers' views into the merged dynamic-smem arena
    offs = sorted({int(x) for x in re.findall(r"buf_dyn_shmem \+ (\d+)\)", src)})
    return offs


def _compile(scoped: bool):
    return tilelang.compile(_kernel(scoped), out_idx=[1], pass_configs={"tl.disable_warp_specialized": True})


@tilelang.testing.requires_cuda
def test_shared_lifetime_scope_overlays_branches():
    k_sum = _compile(False)
    k_max = _compile(True)
    off_sum = _view_offsets(k_sum.get_kernel_source())
    off_max = _view_offsets(k_max.get_kernel_source())
    # default plan: a and b get disjoint regions (a 4 KB + b 8 KB)
    assert len(off_sum) == 2 and off_sum[1] > 0, off_sum
    # lifetime scopes: both start at offset 0 (overlaid)
    assert off_max == [0], off_max

    X = torch.randn(ITERS, N, device="cuda")
    ref = torch.empty_like(X)
    for it in range(ITERS):
        if it % 2 == 0:
            ref[it] = (X[it] * 2.0).flip(0)
        else:
            ref[it] = (torch.cat([X[it], X[it]]) + 1.0).flip(0)[:N]
    for k in (k_sum, k_max):
        torch.testing.assert_close(k(X), ref, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
