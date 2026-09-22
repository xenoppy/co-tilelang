"""NVML via ctypes (no pynvml in the env): one-shot GPU state and a background
sampler of SM clock / power / temperature / clock-event (throttle) reasons.

Measured on RTX PRO 6000 Blackwell, driver 580.173.02 (2026-09-22):
  * every direct query (clock, instant/avg power, temperature, reasons) is refreshed by
    the driver only every ~500 ms. Polling at 10 ms therefore returns the same value
    ~50 times, and a window shorter than ~0.5 s may see only values from BEFORE it.
    ``summary()`` reports the refresh period it actually observed.
  * nvmlDeviceGetSamples(NVML_TOTAL_POWER_SAMPLES) gives real 20 ms power samples
    (the driver's ring buffer holds ~120 of them, so the sampler drains it periodically).
    Clock samples (PROCESSOR_CLK) are NOT_SUPPORTED.
  * nvmlDeviceGetTotalEnergyConsumption reads ~1/6 of the sampled power (about 100 W
    during a 600 W load) and costs ~0.5 ms per call -> not used.
Use ``cobench.clock.ClockProbe`` for an accurate, microsecond-level SM clock.
"""
from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

NVML_CLOCK_GRAPHICS, NVML_CLOCK_SM, NVML_CLOCK_MEM = 0, 1, 2
NVML_TEMPERATURE_GPU = 0
NVML_FI_DEV_POWER_AVERAGE = 185
NVML_FI_DEV_POWER_INSTANT = 186
NVML_TOTAL_POWER_SAMPLES = 0

# nvmlClocksEventReasons (a.k.a. ClocksThrottleReasons) bit masks
THROTTLE_REASONS = {
    0x1: "GpuIdle",
    0x2: "ApplicationsClocksSetting",
    0x4: "SwPowerCap",
    0x8: "HwSlowdown",
    0x10: "SyncBoost",
    0x20: "SwThermalSlowdown",
    0x40: "HwThermalSlowdown",
    0x80: "HwPowerBrakeSlowdown",
    0x100: "DisplayClockSetting",
}
# Reasons that indicate the clock is being held below what the workload asked for.
PERF_LIMITING = 0x4 | 0x8 | 0x20 | 0x40 | 0x80


class NvmlError(RuntimeError):
    pass


class _Value(ctypes.Union):
    _fields_ = [("dVal", ctypes.c_double), ("uiVal", ctypes.c_uint), ("ulVal", ctypes.c_ulong),
                ("ullVal", ctypes.c_ulonglong), ("sllVal", ctypes.c_longlong),
                ("siVal", ctypes.c_int), ("usVal", ctypes.c_ushort)]


class _FieldValue(ctypes.Structure):
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint),
                ("timestamp", ctypes.c_longlong), ("latencyUsec", ctypes.c_longlong),
                ("valueType", ctypes.c_int), ("nvmlReturn", ctypes.c_int), ("value", _Value)]


class _Sample(ctypes.Structure):
    _fields_ = [("timeStamp", ctypes.c_ulonglong), ("sampleValue", _Value)]


_LIB = None
_LOCK = threading.Lock()


def _lib():
    global _LIB
    with _LOCK:
        if _LIB is None:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            r = lib.nvmlInit_v2()
            if r != 0:
                raise NvmlError(f"nvmlInit_v2 failed: {r}")
            _LIB = lib
    return _LIB


def _call(fn_name, *args):
    r = getattr(_lib(), fn_name)(*args)
    if r != 0:
        raise NvmlError(f"{fn_name} failed with nvmlReturn {r}")


class NvmlDevice:
    """NVML handle for a torch CUDA device (matched by PCI bus id)."""

    def __init__(self, device=None):
        from .cudrv import device_index
        p = torch.cuda.get_device_properties(device_index(device))
        bus = f"{p.pci_domain_id:08x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0".encode()
        self.handle = ctypes.c_void_p()
        _call("nvmlDeviceGetHandleByPciBusId_v2", ctypes.c_char_p(bus), ctypes.byref(self.handle))
        lib = _lib()
        self._reason_fn = ("nvmlDeviceGetCurrentClocksEventReasons"
                           if hasattr(lib, "nvmlDeviceGetCurrentClocksEventReasons")
                           else "nvmlDeviceGetCurrentClocksThrottleReasons")
        self._instant_power = self._field_power(NVML_FI_DEV_POWER_INSTANT) is not None

    # -- scalar queries -----------------------------------------------------
    def _uint(self, fn, *pre):
        v = ctypes.c_uint()
        _call(fn, self.handle, *pre, ctypes.byref(v))
        return v.value

    def clock_mhz(self, which=NVML_CLOCK_SM) -> int:
        return self._uint("nvmlDeviceGetClockInfo", ctypes.c_int(which))

    def max_clock_mhz(self, which=NVML_CLOCK_SM) -> int:
        return self._uint("nvmlDeviceGetMaxClockInfo", ctypes.c_int(which))

    def temperature_c(self) -> int:
        return self._uint("nvmlDeviceGetTemperature", ctypes.c_int(NVML_TEMPERATURE_GPU))

    def power_limit_w(self) -> float:
        return self._uint("nvmlDeviceGetEnforcedPowerLimit") / 1000.0

    def pstate(self) -> int:
        v = ctypes.c_int()
        _call("nvmlDeviceGetPerformanceState", self.handle, ctypes.byref(v))
        return v.value

    def reasons(self) -> int:
        v = ctypes.c_ulonglong()
        _call(self._reason_fn, self.handle, ctypes.byref(v))
        return v.value

    def _field_power(self, field_id) -> float | None:
        fv = (_FieldValue * 1)()
        fv[0].fieldId = field_id
        r = _lib().nvmlDeviceGetFieldValues(self.handle, 1, fv)
        if r != 0 or fv[0].nvmlReturn != 0:
            return None
        return fv[0].value.uiVal / 1000.0

    def power_w(self) -> float:
        """Instantaneous board power if exposed, else nvmlDeviceGetPowerUsage (1 s average).
        Either way refreshed only every ~500 ms on this driver."""
        if self._instant_power:
            p = self._field_power(NVML_FI_DEV_POWER_INSTANT)
            if p is not None:
                return p
        return self._uint("nvmlDeviceGetPowerUsage") / 1000.0

    def power_samples(self, since_us: int = 0) -> list[tuple[int, float]]:
        """Driver-buffered power samples [(timestamp_us_since_epoch, W)] newer than since_us."""
        vt, cnt = ctypes.c_int(), ctypes.c_uint(0)
        f = _lib().nvmlDeviceGetSamples
        r = f(self.handle, ctypes.c_int(NVML_TOTAL_POWER_SAMPLES), ctypes.c_ulonglong(since_us),
              ctypes.byref(vt), ctypes.byref(cnt), None)
        if r != 0 or cnt.value == 0:
            return []
        buf = (_Sample * cnt.value)()
        r = f(self.handle, ctypes.c_int(NVML_TOTAL_POWER_SAMPLES), ctypes.c_ulonglong(since_us),
              ctypes.byref(vt), ctypes.byref(cnt), buf)
        if r != 0:
            return []
        out = []
        for i in range(cnt.value):
            v = buf[i].sampleValue
            raw = {0: v.dVal, 1: v.uiVal, 2: v.ulVal, 3: v.ullVal, 4: v.sllVal}.get(vt.value, v.uiVal)
            if buf[i].timeStamp > since_us:
                out.append((int(buf[i].timeStamp), float(raw) / 1000.0))
        return out

    def state(self) -> dict:
        """One-shot snapshot (for result metadata; values may be up to ~0.5 s old)."""
        return {
            "sm_clock_mhz": self.clock_mhz(NVML_CLOCK_SM),
            "mem_clock_mhz": self.clock_mhz(NVML_CLOCK_MEM),
            "max_sm_clock_mhz": self.max_clock_mhz(NVML_CLOCK_SM),
            "temp_c": self.temperature_c(),
            "power_w": self.power_w(),
            "power_limit_w": self.power_limit_w(),
            "pstate": self.pstate(),
            "reasons": decode_reasons(self.reasons()),
        }


def decode_reasons(mask: int) -> list[str]:
    return [name for bit, name in THROTTLE_REASONS.items() if mask & bit]


def gpu_state(device=None) -> dict:
    return NvmlDevice(device).state()


def _refresh_ms(t: np.ndarray, v: np.ndarray) -> float | None:
    ch = np.nonzero(np.diff(v))[0] + 1
    return float(np.median(np.diff(t[ch])) * 1e3) if ch.size >= 2 else None


@dataclass
class NvmlSampler:
    """Background sampler (context manager)::

        with NvmlSampler(interval_ms=10) as s:
            ... GPU work ...
        s.summary()

    Polls SM clock / power / temperature / reasons every ``interval_ms`` and drains the
    driver's 20 ms power-sample buffer. See module doc for the (500 ms) refresh caveat.
    """
    interval_ms: float = 10.0
    device: object = None
    t: list = field(default_factory=list)
    sm_clock: list = field(default_factory=list)
    power: list = field(default_factory=list)
    temp: list = field(default_factory=list)
    reason: list = field(default_factory=list)
    power_buf: list = field(default_factory=list)   # (us since epoch, W), 20 ms samples

    def __post_init__(self):
        self._dev = NvmlDevice(self.device)
        self._stop = threading.Event()
        self._thread = None
        self._errors = []
        self._last_ps = 0

    def _drain_power(self):
        new = self._dev.power_samples(self._last_ps)
        if new:
            self.power_buf.extend(new)
            self._last_ps = new[-1][0]

    def _sample_once(self):
        d = self._dev
        self.t.append(time.perf_counter())
        self.sm_clock.append(d.clock_mhz(NVML_CLOCK_SM))
        self.power.append(d.power_w())
        self.temp.append(d.temperature_c())
        self.reason.append(d.reasons())

    def _run(self):
        period = self.interval_ms / 1000.0
        nxt = time.perf_counter()
        last_drain = nxt
        while not self._stop.is_set():
            try:
                self._sample_once()
                if time.perf_counter() - last_drain > 0.5:
                    self._drain_power()
                    last_drain = time.perf_counter()
            except Exception as e:  # keep sampling; report at the end
                self._errors.append(repr(e))
                if len(self._errors) > 100:
                    return
            nxt += period
            delay = nxt - time.perf_counter()
            if delay < 0:
                nxt, delay = time.perf_counter(), 0
            self._stop.wait(delay)

    def start(self):
        self._t0 = time.perf_counter()
        self._w0 = time.time()
        self._last_ps = int(self._w0 * 1e6)  # only power samples inside the window
        self._thread = threading.Thread(target=self._run, daemon=True, name="nvml-sampler")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._t1 = time.perf_counter()
        self._w1 = time.time()
        # the driver publishes power samples with a small delay; give it one period
        time.sleep(0.03)
        self._drain_power()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def summary(self) -> dict:
        n = len(self.t)
        dur = self._t1 - self._t0
        out = {"n_samples": n, "duration_s": dur, "interval_ms_target": self.interval_ms,
               "max_sm_clock_mhz": self._dev.max_clock_mhz(NVML_CLOCK_SM),
               "errors": self._errors[:5]}
        if n == 0:
            return out
        t = np.asarray(self.t)
        sm = np.asarray(self.sm_clock, float)
        pw = np.asarray(self.power, float)
        tp = np.asarray(self.temp, float)
        rs = np.asarray(self.reason, dtype=np.uint64)
        out["interval_ms_median"] = float(np.median(np.diff(t)) * 1e3) if n > 1 else None
        out["refresh_ms_observed"] = {"sm_clock": _refresh_ms(t, sm), "power": _refresh_ms(t, pw)}
        out["stale_warning"] = dur < 1.0
        out["sm_clock_mhz"] = {"min": float(sm.min()), "median": float(np.median(sm)),
                               "max": float(sm.max()), "mean": float(sm.mean())}
        out["power_w_polled"] = {"mean": float(pw.mean()), "max": float(pw.max())}
        ps = [(ts, w) for ts, w in self.power_buf if self._w0 * 1e6 <= ts <= self._w1 * 1e6]
        if ps:
            pv = np.array([w for _, w in ps])
            out["power_w"] = {"n": int(pv.size), "mean": float(pv.mean()),
                              "min": float(pv.min()), "max": float(pv.max()), "source": "20ms-samples"}
        else:
            out["power_w"] = {"n": 0, "mean": float(pw.mean()), "min": float(pw.min()),
                              "max": float(pw.max()), "source": "polled"}
        out["temp_c"] = {"min": float(tp.min()), "max": float(tp.max())}
        union = int(np.bitwise_or.reduce(rs))
        perf = (rs & np.uint64(PERF_LIMITING)) != 0
        out["throttle"] = {
            "reasons_seen": decode_reasons(union),
            "any_perf_limiting": bool(perf.any()),
            "frac_samples_perf_limited": float(perf.mean()),
            "frac_by_reason": {name: float(((rs & np.uint64(bit)) != 0).mean())
                               for bit, name in THROTTLE_REASONS.items() if union & bit},
        }
        return out

    def samples(self) -> dict:
        """Raw per-sample arrays (t relative to start, seconds)."""
        return {"t_s": [x - self._t0 for x in self.t], "sm_clock_mhz": list(self.sm_clock),
                "power_w": list(self.power), "temp_c": list(self.temp),
                "reasons": [int(r) for r in self.reason],
                "power_20ms": [((ts / 1e6) - self._w0, w) for ts, w in self.power_buf]}
