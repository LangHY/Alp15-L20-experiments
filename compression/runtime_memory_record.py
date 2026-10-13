"""Phase-level measurements; L20 host/device values are not Jetson unified RAM."""
import contextlib
import json
import os
import threading
import time
from pathlib import Path

import torch


def storage_inventory(module, extra_tensors=()):
    """Count complete backing storages once, including views and shared weights."""
    seen = set()
    totals = {}
    tensors = list(module.parameters()) + list(module.buffers()) + list(extra_tensors)
    logical = 0
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Inventory requires explicit tensors")
        if tensor.device.type == "meta":
            raise ValueError("Meta tensors have no measured storage")
        logical += tensor.numel() * tensor.element_size()
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            totals[str(tensor.device)] = totals.get(str(tensor.device), 0) + storage.nbytes()
    return {"unique_storage_bytes_by_device": totals,
            "logical_tensor_bytes_including_aliases": logical,
            "unique_storages": len(seen),
            "scope": "Explicit model parameters/buffers and supplied runtime tensors only"}


def process_rss_bytes():
    """Current process resident pages, not CUDA or whole-machine memory."""
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')


class MemoryRecorder:
    def __init__(self, output, device=None, interval_s=0.02):
        if interval_s <= 0:
            raise ValueError("Sampling interval must be positive")
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents mixing or overwriting prior experiments.
        self.output.open('x').close()
        self.device = device
        self.interval_s = interval_s

    @contextlib.contextmanager
    def phase(self, name, module=None, extra_tensors=()):
        """One sequential phase per recorder; do not nest peak-counter resets."""
        if getattr(self, '_active', False):
            raise RuntimeError("Memory phases cannot overlap")
        self._active = True
        cuda = self.device is not None
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        stop = threading.Event()
        samples = []
        sampling_errors = []
        def sample():
            while not stop.is_set():
                try:
                    samples.append([time.monotonic(), process_rss_bytes()])
                except Exception as exc:
                    sampling_errors.append(str(exc))
                    return
                stop.wait(self.interval_s)
        worker = threading.Thread(target=sample, daemon=True)
        start = time.monotonic()
        worker.start()
        error = None
        try:
            yield
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            try:
                if cuda:
                    torch.cuda.synchronize(self.device)
                stop.set()
                worker.join()
                row = {"phase": name, "status": "failed" if error else "completed",
                       "elapsed_s": time.monotonic() - start, "error": error,
                       "rss_samples_monotonic_s_bytes": samples,
                       "rss_sampling_interval_requested_s": self.interval_s,
                       "rss_sampling_errors": sampling_errors,
                       "process_rss_sampled_peak_bytes": max((s[1] for s in samples), default=None),
                       "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if cuda else None,
                       "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if cuda else None,
                       "inventory": storage_inventory(module, extra_tensors) if module is not None else None,
                       "limitations": ["RSS peaks are sampled, brief peaks may be missed",
                                        "CUDA allocator excludes driver and other-process allocations",
                                        "L20 RSS plus CUDA is not a Jetson unified-memory measurement",
                                        "Do not add independent phase maxima or inventory to allocator peaks"]}
                with self.output.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row) + '\n')
            finally:
                stop.set()
                worker.join()
                self._active = False
