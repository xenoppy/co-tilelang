"""%smid / %globaltimer probes and the SM-id remapping table.

Kernels (inline PTX, NVRTC):
  cta_probe      per CTA: blockIdx, %smid, %nsmid, dispatch ticket (atomic order),
                 %globaltimer at start/end, %warpid of warp 0, %nwarpid. Spins spin_ns
                 so CTAs of one wave are co-resident.
  gt_resolution  one thread per CTA polls %globaltimer and clock64, logging every change.
  gt_ping        1 CTA/SM; round r: CTA (r % G) stamps %globaltimer and raises a flag,
                 every other CTA stamps when it observes the flag -> pairwise
                 (latency + clock offset); both directions give the offset.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch

from .cudrv import CudaKernel, device_index

_SRC = r"""
__device__ __forceinline__ unsigned smid_()   { unsigned r; asm volatile("mov.u32 %0, %%smid;"   : "=r"(r)); return r; }
__device__ __forceinline__ unsigned nsmid_()  { unsigned r; asm volatile("mov.u32 %0, %%nsmid;"  : "=r"(r)); return r; }
__device__ __forceinline__ unsigned warpid_() { unsigned r; asm volatile("mov.u32 %0, %%warpid;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned nwarpid_(){ unsigned r; asm volatile("mov.u32 %0, %%nwarpid;": "=r"(r)); return r; }
__device__ __forceinline__ unsigned dsmem_()  { unsigned r; asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned long long gtimer_() {
  unsigned long long r; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); return r; }

extern "C" __global__ void cta_probe(unsigned long long* out, unsigned long long* ticket,
                                     unsigned long long spin_ns) {
  extern __shared__ volatile char smem_[];
  if (threadIdx.x == 0) {
    unsigned long long t0 = gtimer_();
    unsigned long long tk = atomicAdd(ticket, 1ull);
    unsigned long long* o = out + (unsigned long long)blockIdx.x * 8;
    o[0] = blockIdx.x; o[1] = smid_(); o[2] = nsmid_(); o[3] = tk; o[4] = t0;
    o[6] = warpid_(); o[7] = nwarpid_();
    if (dsmem_() > 0) smem_[0] = 1;   // keep the dynamic smem allocation live
    unsigned long long t;
    do { t = gtimer_(); } while (t - t0 < spin_ns);
    o[5] = t;
  }
  __syncthreads();
}

extern "C" __global__ void gt_resolution(unsigned long long* out, int max_changes,
                                         long long max_ns, unsigned long long* meta) {
  if (threadIdx.x != 0) return;
  unsigned long long* o = out + (unsigned long long)blockIdx.x * max_changes * 2;
  unsigned long long t0 = gtimer_(), prev = t0, reads = 0, decreases = 0;
  long long c0 = clock64();
  int n = 0;
  while (n < max_changes) {
    unsigned long long t = gtimer_();
    long long c = clock64();
    ++reads;
    if (t != prev) {
      if (t < prev) ++decreases;
      o[2 * n] = t; o[2 * n + 1] = (unsigned long long)c; ++n; prev = t;
    }
    if ((long long)(t - t0) > max_ns) break;
  }
  long long c1 = clock64();
  unsigned long long* m = meta + blockIdx.x * 6;
  m[0] = smid_(); m[1] = n; m[2] = reads; m[3] = decreases; m[4] = (unsigned long long)(c1 - c0);
  m[5] = prev - t0;
}

extern "C" __global__ void gt_ping(volatile unsigned* flags, unsigned long long* send_t,
                                   unsigned long long* recv_t, unsigned* smids, int rounds,
                                   unsigned* arrive, unsigned long long* err,
                                   unsigned long long delay_ns) {
  extern __shared__ volatile char smem_[];
  if (threadIdx.x != 0) return;
  if (dsmem_() > 0) smem_[0] = 1;
  const int G = gridDim.x, g = blockIdx.x;
  smids[g] = smid_();
  atomicAdd(arrive, 1u);
  unsigned long long t0 = gtimer_();
  while (*(volatile unsigned*)arrive < (unsigned)G) {
    if (gtimer_() - t0 > 2000000000ull) { atomicAdd(err, 1ull); return; }
  }
  for (int r = 0; r < rounds; ++r) {
    const int sender = r % G;
    if (g == sender) {
      unsigned long long ts = gtimer_();
      while (gtimer_() - ts < delay_ns) { }
      unsigned long long t = gtimer_();
      send_t[r] = t;
      __threadfence();
      flags[r] = 1u;
    } else {
      unsigned long long ts = gtimer_();
      while (flags[r] == 0u) {
        if (gtimer_() - ts > 1000000000ull) { atomicAdd(err, 1ull); return; }
      }
      recv_t[(unsigned long long)r * G + g] = gtimer_();
    }
  }
}
"""


@functools.lru_cache(maxsize=None)
def _k(device: int, name: str) -> CudaKernel:
    sig = {"cta_probe": "ppQ", "gt_resolution": "piqp", "gt_ping": "ppppippQ"}[name]
    return CudaKernel(_SRC, name, sig, device=device)


def _props(dev):
    return torch.cuda.get_device_properties(dev)


def one_cta_per_sm_smem(device=None) -> int:
    """Dynamic smem (bytes) that forces at most 1 resident CTA per SM."""
    p = _props(device_index(device))
    per_sm = p.shared_memory_per_multiprocessor
    optin = p.shared_memory_per_block_optin
    s = per_sm // 2 + 4096
    if s > optin:
        raise RuntimeError("cannot force 1 CTA/SM via shared memory on this device")
    return s


# ---------------------------------------------------------------------------
# CTA -> SM placement
# ---------------------------------------------------------------------------
FIELDS = ("block", "smid", "nsmid", "ticket", "t_start", "t_end", "warpid", "nwarpid")


def probe_ctas(grid: int, block: int = 128, *, smem: int = 0, spin_ns: int = 50_000,
               stream=None, device=None) -> dict:
    """Launch cta_probe; returns dict of numpy arrays (FIELDS) + 'occupancy' (CTAs/SM
    the occupancy calculator predicts for this block/smem)."""
    dev = device_index(device)
    k = _k(dev, "cta_probe")
    out = torch.zeros(grid * 8, dtype=torch.int64, device=f"cuda:{dev}")
    ticket = torch.zeros(1, dtype=torch.int64, device=f"cuda:{dev}")
    s = stream if stream is not None else torch.cuda.current_stream(dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    k(grid, block, out, ticket, spin_ns, smem=smem, stream=s)
    s.synchronize()
    a = out.view(grid, 8).cpu().numpy()
    res = {f: a[:, i].copy() for i, f in enumerate(FIELDS)}
    res["occupancy"] = k.occupancy(block, smem)
    return res


@dataclass
class SmRemap:
    """smid -> dense index (0..n-1, ascending smid order); -1 for unused ids."""
    smids: np.ndarray        # sorted observed smids
    nsmid: int               # %nsmid
    table: np.ndarray        # int32, len = max(nsmid, max smid + 1)

    @property
    def n(self) -> int:
        return int(self.smids.size)

    @property
    def holes(self) -> list[int]:
        return [int(i) for i in np.nonzero(self.table < 0)[0]]

    def dense(self, smid: int) -> int:
        return int(self.table[smid])

    def to_tensor(self, device=None) -> torch.Tensor:
        return torch.as_tensor(self.table, dtype=torch.int32, device=f"cuda:{device_index(device)}")

    @classmethod
    def from_smids(cls, smids, nsmid: int) -> "SmRemap":
        smids = np.unique(np.asarray(smids, dtype=np.int64))
        size = int(max(nsmid, int(smids.max()) + 1))
        table = np.full(size, -1, dtype=np.int32)
        table[smids] = np.arange(smids.size, dtype=np.int32)
        return cls(smids=smids, nsmid=int(nsmid), table=table)


def build_sm_remap(stream=None, *, expected: int | None = None, device=None,
                   ctas_per_sm: int = 4, spin_ns: int = 20_000, attempts: int = 3) -> SmRemap:
    """Discover the SMs a stream can use (all SMs for a normal stream, the partition for a
    green-context stream) and build the smid -> dense-index table.

    Launches ctas_per_sm * expected CTAs with smem forcing 1 CTA/SM so every SM is hit;
    raises if fewer than ``expected`` distinct smids are observed after ``attempts``.
    ``expected`` defaults to the device SM count for a normal stream; pass the partition
    size (e.g. SmPartition.n_sms) for a green stream.
    """
    dev = device_index(device)
    nsm_dev = _props(dev).multi_processor_count
    exp = expected if expected is not None else nsm_dev
    smem = one_cta_per_sm_smem(dev)
    seen: set[int] = set()
    nsmid = 0
    for _ in range(attempts):
        r = probe_ctas(exp * ctas_per_sm, 32, smem=smem, spin_ns=spin_ns, stream=stream, device=dev)
        seen.update(int(x) for x in r["smid"])
        nsmid = int(r["nsmid"].max())
        if len(seen) >= exp:
            break
        spin_ns *= 4
    if len(seen) != exp:
        raise RuntimeError(f"observed {len(seen)} distinct smids, expected {exp}")
    return SmRemap.from_smids(sorted(seen), nsmid)


# ---------------------------------------------------------------------------
# %globaltimer
# ---------------------------------------------------------------------------
def globaltimer_resolution(*, n_ctas: int | None = None, max_changes: int = 512,
                           max_ns: int = 20_000_000, device=None) -> dict:
    """Poll %globaltimer on one thread per CTA (1 CTA/SM); log every change."""
    dev = device_index(device)
    k = _k(dev, "gt_resolution")
    G = n_ctas or _props(dev).multi_processor_count
    out = torch.zeros(G * max_changes * 2, dtype=torch.int64, device=f"cuda:{dev}")
    meta = torch.zeros(G * 6, dtype=torch.int64, device=f"cuda:{dev}")
    smem = one_cta_per_sm_smem(dev)
    k(G, 32, out, max_changes, max_ns, meta, smem=smem)
    torch.cuda.synchronize(dev)
    o = out.view(G, max_changes, 2).cpu().numpy().astype(np.int64)
    m = meta.view(G, 6).cpu().numpy().astype(np.int64)
    deltas, cyc_per_ns = [], []
    for g in range(G):
        n = int(m[g, 1])
        t, c = o[g, :n, 0], o[g, :n, 1]
        if n > 2:
            dt = np.diff(t)
            deltas.append(dt)
            cyc_per_ns.append((c[-1] - c[0]) / max(1, (t[-1] - t[0])))
    d = np.concatenate(deltas) if deltas else np.zeros(0, dtype=np.int64)
    vals, cnts = np.unique(d, return_counts=True)
    order = np.argsort(-cnts)[:8]
    reads = m[:, 2].astype(np.float64)
    return {
        "n_ctas": G,
        "changes_per_cta": int(np.median(m[:, 1])),
        "delta_ns": {"min": int(d.min()) if d.size else None,
                     "median": float(np.median(d)) if d.size else None,
                     "mean": float(d.mean()) if d.size else None,
                     "max": int(d.max()) if d.size else None,
                     "p01": float(np.percentile(d, 1)) if d.size else None,
                     "p99": float(np.percentile(d, 99)) if d.size else None},
        "delta_ns_top_values": {int(vals[i]): int(cnts[i]) for i in order},
        "decreases_total": int(m[:, 3].sum()),
        "read_cost_cycles": float(np.median(m[:, 4] / np.maximum(reads, 1))),
        "implied_sm_clock_mhz": {"median": float(np.median(cyc_per_ns) * 1e3) if cyc_per_ns else None,
                                 "min": float(np.min(cyc_per_ns) * 1e3) if cyc_per_ns else None,
                                 "max": float(np.max(cyc_per_ns) * 1e3) if cyc_per_ns else None},
    }


def globaltimer_skew(*, rounds_per_cta: int = 3, delay_ns: int = 3000, device=None) -> dict:
    """Pairwise flag ping between all SMs (1 CTA/SM) to bound %globaltimer offsets.

    d(i->j) = recv_j - send_i = latency + (off_j - off_i). Using min over rounds in both
    directions: off_j - off_i = (d(i->j) - d(j->i)) / 2, latency = (d(i->j) + d(j->i)) / 2.
    """
    dev = device_index(device)
    k = _k(dev, "gt_ping")
    G = _props(dev).multi_processor_count
    R = G * rounds_per_cta
    cuda = f"cuda:{dev}"
    flags = torch.zeros(R, dtype=torch.int32, device=cuda)
    send = torch.zeros(R, dtype=torch.int64, device=cuda)
    recv = torch.zeros(R * G, dtype=torch.int64, device=cuda)
    smids = torch.zeros(G, dtype=torch.int32, device=cuda)
    arrive = torch.zeros(1, dtype=torch.int32, device=cuda)
    err = torch.zeros(1, dtype=torch.int64, device=cuda)
    k(G, 32, flags, send, recv, smids, R, arrive, err, delay_ns, smem=one_cta_per_sm_smem(dev))
    torch.cuda.synchronize(dev)
    if int(err.item()):
        raise RuntimeError("gt_ping timed out (CTAs not co-resident?)")
    s = send.cpu().numpy().astype(np.int64)
    rv = recv.view(R, G).cpu().numpy().astype(np.int64)
    sm = smids.cpu().numpy()
    INF = np.iinfo(np.int64).max
    dmin = np.full((G, G), INF, dtype=np.int64)
    neg = 0
    for r in range(R):
        i = r % G
        d = rv[r] - s[r]
        d[i] = INF
        neg += int((d[np.arange(G) != i] < 0).sum())
        dmin[i] = np.minimum(dmin[i], d)
    # offsets relative to CTA 0, and pairwise one-way latency estimates
    off = np.array([(dmin[0, j] - dmin[j, 0]) / 2 if j else 0.0 for j in range(G)])
    lat = np.array([(dmin[0, j] + dmin[j, 0]) / 2 for j in range(1, G)])
    mask = ~np.eye(G, dtype=bool)
    allmin = dmin[mask]
    return {
        "n_sms": G, "rounds": R,
        "negative_one_way_samples": neg,
        "one_way_min_ns": {"min": int(allmin.min()), "median": float(np.median(allmin)),
                           "max": int(allmin.max())},
        "offset_vs_cta0_ns": {"min": float(off.min()), "max": float(off.max()),
                              "abs_max": float(np.abs(off).max()),
                              "abs_median": float(np.median(np.abs(off)))},
        "latency_est_ns": {"min": float(lat.min()), "median": float(np.median(lat)),
                           "max": float(lat.max())},
        "offsets_by_smid": {int(sm[j]): float(off[j]) for j in range(G)},
    }
