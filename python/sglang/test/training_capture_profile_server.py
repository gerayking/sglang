"""Profiler-only CPU scopes for attributing capture kernels and DMA events."""

import os
import sys
import threading
import time
from collections import defaultdict
from functools import wraps

import torch

from sglang.srt.training_capture import coordinator
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.kv_exporter import (
    BoundHiCacheKVExporter,
    HiCacheL2KVExporter,
    SelectedLayerKVExporter,
)
from sglang.srt.training_capture.kv_staging import KVStaging
from sglang.srt.training_capture.teacher_batcher import TeacherLogitBatcher
from sglang.srt.training_capture.teacher_staging import TeacherStaging

timings = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "cpu_seconds": 0.0})
timings_lock = threading.Lock()


def timed(name, function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        cpu_started = time.thread_time()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            cpu_elapsed = time.thread_time() - cpu_started
            with timings_lock:
                timings[name]["calls"] += 1
                timings[name]["seconds"] += elapsed
                timings[name]["cpu_seconds"] += cpu_elapsed

    return wrapped


def scoped(name, function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function("training_capture." + name):
            return function(*args, **kwargs)

    return wrapped


coordinator.capture_teacher = scoped("teacher", coordinator.capture_teacher)
coordinator.CaptureCoordinator._admit = scoped(
    "admit", coordinator.CaptureCoordinator._admit
)
TeacherLogitBatcher.enqueue = scoped("teacher_ring", TeacherLogitBatcher.enqueue)
TeacherLogitBatcher._compact = scoped(
    "teacher_background", TeacherLogitBatcher._compact
)
HiCacheL2KVExporter.materialize = scoped(
    "l2_materialize", HiCacheL2KVExporter.materialize
)
HiCacheL2KVExporter.acquire_transient = scoped(
    "l2_transient", HiCacheL2KVExporter.acquire_transient
)
SelectedLayerKVExporter.export = scoped("kv", SelectedLayerKVExporter.export)
BoundHiCacheKVExporter.export = scoped("kv", BoundHiCacheKVExporter.export)
KVStaging.flush = scoped("kv_d2h", KVStaging.flush)
RequestCaptureContext.record_teacher_range = scoped(
    "teacher_d2h", RequestCaptureContext.record_teacher_range
)
RequestCaptureContext.record_positions = scoped(
    "positions_d2h", RequestCaptureContext.record_positions
)

flush_staging = RequestCaptureContext._flush_staging
flush_teacher = scoped("teacher_d2h", flush_staging)


def flush_capture_staging(self, staging):
    # Append flushes are already inside record_teacher_range's scope.
    return (
        flush_teacher(self, staging)
        if isinstance(staging, TeacherStaging)
        else flush_staging(self, staging)
    )


RequestCaptureContext._flush_staging = flush_capture_staging

for method in (
    "before_forward",
    "after_forward",
    "after_result",
    "on_release",
    "_admit",
    "_manifest_fits",
    "_commit",
    "prepare_l2",
    "_drain_l2_releases",
    "_reserve",
    "_retire",
    "_write_snapshot",
):
    setattr(
        coordinator.CaptureCoordinator,
        method,
        timed(method, getattr(coordinator.CaptureCoordinator, method)),
    )
for cls, method in (
    (TeacherLogitBatcher, "enqueue"),
    (TeacherLogitBatcher, "_compact"),
    (RequestCaptureContext, "record_positions"),
    (RequestCaptureContext, "wait_for_copies"),
    (HiCacheL2KVExporter, "acquire_transient"),
):
    setattr(cls, method, timed(method, getattr(cls, method)))

capture_stats = coordinator.CaptureCoordinator.stats


def profile_stats(self):
    result = capture_stats(self)
    with timings_lock:
        result["diagnostic_cpu_wall"] = {
            name: dict(value) for name, value in timings.items()
        }
    return result


coordinator.CaptureCoordinator.stats = profile_stats


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
