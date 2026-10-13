"""Test-only writer prefixes. Held records finish normally after the gate opens.

Never use this entry point for serving: checksum reuse and checksum-only validation
are diagnostic controls, and all publications require independent readback checks.
"""

import functools
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import torch

from sglang.srt.training_capture import (
    payload_hash,
    snapshot,
    snapshot_writer,
    teacher_batcher,
)
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.coordinator import CaptureCoordinator
from sglang.srt.training_capture.kv_exporter import HiCacheL2KVExporter
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import tensor_bytes
from sglang.srt.training_capture.teacher_batcher import TeacherLogitBatcher
from sglang.srt.training_capture.timings import CaptureTimings
from sglang.srt.training_capture.writer_process import ProcessSnapshotWriter

METRICS = {}
THREADS = {}
LOCK = threading.Lock()
LOCAL = threading.local()
ACTIVE_MODE = None


class ProbeCUDA:
    def __getattr__(self, name):
        return getattr(torch.cuda, name)

    def Event(self, *args, **kwargs):
        if (
            ACTIVE_MODE == "blocking_teacher"
            and threading.current_thread().name == "training-teacher-batcher"
        ):
            kwargs["blocking"] = True
        return torch.cuda.Event(*args, **kwargs)


class ProbeTorch:
    cuda = ProbeCUDA()

    def __getattr__(self, name):
        return getattr(torch, name)


# Limit the event flag experiment to teacher_batcher's completion events.
teacher_batcher.torch = ProbeTorch()

after_forward = CaptureCoordinator.after_forward
reserve = CaptureCoordinator._reserve
begin = ProcessSnapshotWriter.begin


def probe_after_forward(self, batch, forward_batch, *args, **kwargs):
    canonical = getattr(forward_batch, "positions_are_canonical", False)
    if ACTIVE_MODE == "original_positions":
        forward_batch.positions_are_canonical = False
    try:
        return after_forward(self, batch, forward_batch, *args, **kwargs)
    finally:
        forward_batch.positions_are_canonical = canonical


def probe_reserve(self):
    LOCAL.catalog = self.catalog
    try:
        return reserve(self)
    finally:
        del LOCAL.catalog


def probe_begin(self, identity):
    if ACTIVE_MODE == "parent_lease":
        return LOCAL.catalog.begin(identity)
    return begin(self, identity)


CaptureCoordinator.after_forward = probe_after_forward
CaptureCoordinator._reserve = probe_reserve
ProcessSnapshotWriter.begin = probe_begin

record_positions = RequestCaptureContext.record_positions


def probe_positions(self, positions, *, start):
    if ACTIVE_MODE == "cpu_positions":
        # Diagnostic AR control: admission rejects custom positions and the
        # context already initializes this buffer to arange(max_tokens).
        return
    return record_positions(self, positions, start=start)


RequestCaptureContext.record_positions = probe_positions

compact_teacher = TeacherLogitBatcher._compact


def deferred_teacher(self, jobs):
    while ACTIVE_MODE == "teacher_deferred" and not self.stop.is_set():
        time.sleep(0.01)
    return compact_teacher(self, jobs)


TeacherLogitBatcher._compact = deferred_teacher


@contextmanager
def scope(name):
    stack = getattr(LOCAL, "timing_stack", None)
    if stack is None:
        stack = LOCAL.timing_stack = []
        with LOCK:
            THREADS[threading.get_native_id()] = {
                "name": threading.current_thread().name,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
            }
    children = [0.0, 0.0]
    stack.append(children)
    wall, cpu = time.perf_counter(), time.thread_time()
    try:
        yield
    finally:
        elapsed, used = time.perf_counter() - wall, time.thread_time() - cpu
        stack.pop()
        if stack:
            stack[-1][0] += elapsed
            stack[-1][1] += used
        with LOCK:
            value = METRICS.setdefault(
                name,
                {
                    "calls": 0,
                    "wall": 0.0,
                    "cpu": 0.0,
                    "self_wall": 0.0,
                    "self_cpu": 0.0,
                },
            )
            value["calls"] += 1
            value["wall"] += elapsed
            value["cpu"] += used
            value["self_wall"] += elapsed - children[0]
            value["self_cpu"] += used - children[1]


def timed(name, function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if threading.current_thread().name != "training-snapshot-writer":
            return function(*args, **kwargs)
        with scope(name):
            return function(*args, **kwargs)

    return wrapped


measure_stage = CaptureTimings.measure


@contextmanager
def measured_stage(self, name):
    with scope("stage/" + name), measure_stage(self, name):
        yield


CaptureTimings.measure = measured_stage
HiCacheL2KVExporter.materialize = timed(
    "l2_materialize", HiCacheL2KVExporter.materialize
)
snapshot.digest_bytes = timed("sha_snapshot", snapshot.digest_bytes)
payload_hash.PayloadHasher.digests = timed(
    "sha_payload", payload_hash.PayloadHasher.digests
)
MooncakeSnapshotStore._registration = timed(
    "store_registration", MooncakeSnapshotStore._registration
)
os.fsync = timed("journal_fsync", os.fsync)
original_validate = snapshot_writer.validate_tensors


class TimedClient:
    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        value = getattr(self.client, name)
        if name in ("batch_put_from", "batch_is_exist", "put_from", "is_exist"):
            return timed("native/" + name, value)
        return value


def probe_validate(manifest, tensors, *, payload_hasher):
    if getattr(LOCAL, "checksum_only", False):
        digests = payload_hasher.digests(
            [tensor_bytes(tensors[obj.key]) for obj in manifest.objects]
        )
        if digests != [obj.sha256 for obj in manifest.objects]:
            raise RuntimeError("probe checksum mismatch")
        return
    return original_validate(manifest, tensors, payload_hasher=payload_hasher)


snapshot_writer.validate_tensors = probe_validate


class CachedHasher:
    """Reuse only one held record's immutable payloads, never across records."""

    workers = 2  # Route snapshot construction through digests(), without workers.

    def __init__(self, original):
        self.original = original
        self.cache = {}

    def digests(self, buffers):
        result = []
        for data in buffers:
            key = (data.obj.__array_interface__["data"][0], data.nbytes)
            if key not in self.cache:
                self.cache[key] = self.original.digests([data])[0]
            result.append(self.cache[key])
        return result


class PrefixComplete(Exception):
    pass


STOPS = {
    "validation": "catalog_register",
    "registered": "store_payload",
    "stored": "catalog_written",
}


def run_prefix(self, record, mode):
    if self.writer_process is not None:
        if mode not in ("copies", "published"):
            raise ValueError("process writer supports copies/published prefixes")
        if record.invalid_reason or record.context.state != "SEALED":
            raise RuntimeError("process probe requires a valid sealed capture")
        self.writer.timings.call("copy_wait", record.context.wait_for_copies)
        if mode == "copies":
            return False
        self._write_snapshot(record)
        for counter in ("writer_started", "copies_completed", "ready"):
            self._count(counter)
        return True
    original_hasher = self.store.payload_hasher
    original_call = self.writer.timings.call
    if mode in ("cached_sha", "cached_sha_checksum_only"):
        self.store.payload_hasher = CachedHasher(original_hasher)
    LOCAL.checksum_only = mode in ("checksum_only", "cached_sha_checksum_only")

    def stop_at(stage, function, *args, **kwargs):
        if stage == STOPS.get(mode):
            raise PrefixComplete
        return original_call(stage, function, *args, **kwargs)

    self.writer.timings.call = stop_at
    try:
        if record.invalid_reason or record.context.state != "SEALED":
            raise RuntimeError(
                f"probe requires sealed capture: {record.invalid_reason}, "
                f"state={record.context.state}"
            )
        self.writer.timings.call("copy_wait", record.context.wait_for_copies)
        if mode == "copies":
            return False
        manifest, tensors = self.writer.timings.call(
            "snapshot_build",
            record.context.prepare_snapshot,
            payload_hasher=self.store.payload_hasher,
            dataset_id=record.lease.dataset_id,
            sample_id=record.lease.sample_id,
            generation_id=record.lease.generation_id,
            teacher=self.teacher,
            kv=self.kv,
            provenance=record.provenance,
            contract_id=self.config.contract_id,
        )
        # Resume with identical created_at and manifest bytes after registration.
        record.context.prepare_snapshot = lambda **unused: (manifest, tensors)
        if mode == "build":
            return False
        self.writer.write(
            manifest,
            tensors,
            record.slot.manifest_buffer,
            record.lease,
            check_current=lambda: self._check_publication(record),
            on_prepared=lambda lease, data: self._publication_prepared(
                record, lease, data
            ),
        )
        for counter in (
            "writer_started",
            "copies_completed",
            "snapshot_built",
            "ready",
        ):
            self._count(counter)
        return True
    except PrefixComplete:
        return False
    finally:
        self.writer.timings.call = original_call
        self.store.payload_hasher = original_hasher
        LOCAL.checksum_only = False


writer_loop = CaptureCoordinator._writer_loop
original_stats = CaptureCoordinator.stats


def stats(self):
    result = original_stats(self)
    with LOCK:
        result["writer_probe"] = {name: dict(value) for name, value in METRICS.items()}
    return result


def prefix_writer_loop(self):
    if self.store is not None:
        self.store.client = TimedClient(self.store.client)
    gate = Path(os.environ["SGLANG_TEST_CAPTURE_WRITER_GATE"])
    get = self.work.get
    pending = deque()
    active_mode = None

    def prefix_get(*args, **kwargs):
        global ACTIVE_MODE
        nonlocal active_mode
        if gate.exists() and not self.writer_stop.is_set():
            if active_mode is None:
                active_mode = json.loads(gate.read_text())["mode"]
                ACTIVE_MODE = active_mode
            if active_mode in (
                "cpu_positions",
                "original_positions",
                "parent_lease",
                "blocking_teacher",
                "blocking_copy_events",
            ):
                return get(*args, **kwargs)
            if active_mode in ("none", "teacher_deferred"):
                time.sleep(0.01)
                raise queue.Empty
            record = get(timeout=0.01)
            if record.context is None or record.invalid_reason:
                return record
            try:
                with scope("prefix/" + active_mode):
                    completed = run_prefix(self, record, active_mode)
            except Exception:
                with scope("prefix_error"):
                    pass
                pending.append((record, False))
                raise
            pending.append((record, completed))
            raise queue.Empty
        active_mode = None
        ACTIVE_MODE = None
        while pending:
            record, completed = pending.popleft()
            if not completed:
                return record
            self._retire(record, complete=True)
            self.work.task_done()
        return get(*args, **kwargs)

    self.work.get = prefix_get
    try:
        writer_loop(self)
    finally:
        self.work.get = get


CaptureCoordinator._writer_loop = prefix_writer_loop
CaptureCoordinator.stats = stats


if os.environ.get("SGLANG_TEST_CAPTURE_CPU_TIMINGS") == "1":
    import psutil

    from sglang.srt.layers import logsumexp
    from sglang.srt.training_capture import teacher

    def cpu_timed(name, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with scope("cpu/" + name):
                return function(*args, **kwargs)

        return wrapped

    for name in (
        "before_forward",
        "after_forward",
        "after_result",
        "_admit",
        "_manifest_fits",
        "_commit",
        "prepare_l2",
        "on_release",
        "_drain_l2_releases",
        "_reserve",
        "_retire",
        "_write_snapshot",
    ):
        setattr(
            CaptureCoordinator, name, cpu_timed(name, getattr(CaptureCoordinator, name))
        )
    for cls, name in (
        (TeacherLogitBatcher, "enqueue"),
        (TeacherLogitBatcher, "_finish"),
        (RequestCaptureContext, "write_async_teacher"),
        (RequestCaptureContext, "wait_for_copies"),
        (RequestCaptureContext, "_wait_for_async_teacher"),
        (HiCacheL2KVExporter, "acquire_transient"),
    ):
        setattr(cls, name, cpu_timed(name, getattr(cls, name)))
    # Time real compaction after the optional diagnostic deferral gate opens.
    compact_teacher = cpu_timed("teacher_compact", compact_teacher)
    teacher_batcher.capture_teacher = cpu_timed(
        "capture_teacher", teacher_batcher.capture_teacher
    )
    teacher._top128 = cpu_timed("top128", teacher._top128)
    logsumexp.row_logsumexp = cpu_timed("logsumexp", logsumexp.row_logsumexp)
    synchronize = torch.cuda.Event.synchronize

    def timed_synchronize(event):
        if threading.current_thread().name == "training-teacher-batcher":
            with scope("cpu/teacher_synchronize"):
                return synchronize(event)
        return synchronize(event)

    torch.cuda.Event.synchronize = timed_synchronize
    prefix_stats = CaptureCoordinator.stats

    def cpu_stats(self):
        result = prefix_stats(self)
        result["diagnostic_process_cpu"] = {"parent": time.process_time()}
        if self.writer_process is not None:
            child = psutil.Process(self.writer_process.process.pid).cpu_times()
            result["diagnostic_process_cpu"]["writer"] = child.user + child.system
        with LOCK:
            result["diagnostic_threads"] = dict(THREADS)
        return result

    CaptureCoordinator.stats = cpu_stats


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
