"""Bounded raw-logit ownership and asynchronous teacher compaction."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from sglang.srt.training_capture.protocol import CaptureError, ContractError
from sglang.srt.training_capture.teacher import capture_teacher
from sglang.srt.training_capture.teacher_staging import (
    TeacherCompletion,
    TeacherStaging,
)

if TYPE_CHECKING:
    from sglang.srt.training_capture.context import RequestCaptureContext


class TeacherRingFull(CaptureError):
    """The capture ring is full; serving may continue after dropping the sample."""


@dataclass(frozen=True)
class _Destination:
    staging: TeacherStaging
    completion: TeacherCompletion
    row: int


@dataclass(frozen=True)
class _Job:
    start: int
    count: int
    destinations: tuple[_Destination, ...]
    ready: torch.cuda.Event
    error: BaseException | None = None


class TeacherLogitBatcher:
    """Copy raw rows into a bounded ring, then compact them on another stream."""

    def append(self, values, row_indices, targets):
        return self.enqueue(values, row_indices, targets)

    def seal(self, staging):
        # The worker records the producer stream after finishing reserved rows.
        pass

    def __init__(
        self,
        *,
        vocab_size: int,
        ring_rows: int,
        batch_rows: int,
        device: torch.device,
        topk_backend: str,
    ):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type != "cuda" or ring_rows < 1 or not 1 <= batch_rows <= ring_rows:
            raise ValueError(
                "async teacher capture requires CUDA and valid ring limits"
            )
        self.vocab_size = vocab_size
        self.capacity = ring_rows
        self.batch_rows = batch_rows
        self.device = device
        self.topk_backend = topk_backend
        with torch.cuda.device(device):
            # SGLang's target logits contract is FP32. Keeping the same dtype is
            # required for bit-equivalent top-k values and logsumexp.
            self.storage = torch.empty(
                (ring_rows, vocab_size), dtype=torch.float32, device=device
            )
            self.stream = torch.cuda.Stream(device=device)
        self.allocated_bytes = self.storage.numel() * self.storage.element_size()
        self.lock = threading.Lock()
        self.head = self.tail = self.used = 0
        self.accepted_rows = self.processed_rows = self.backpressure_rows = 0
        self.batches = self.failures = self.max_occupancy = 0
        self.work: queue.Queue[_Job] = queue.Queue()
        self.stop = threading.Event()
        self.fatal_error: BaseException | None = None
        self.closed = False
        self.thread = threading.Thread(
            target=self._worker, name="training-teacher-batcher", daemon=True
        )
        self.thread.start()

    def _reserve(self, count: int) -> int:
        with self.lock:
            if count > self.capacity - self.used:
                self.backpressure_rows += count
                raise TeacherRingFull("teacher raw-logit ring is full")
            start = self.head
            self.head = (self.head + count) % self.capacity
            self.used += count
            self.accepted_rows += count
            self.max_occupancy = max(self.max_occupancy, self.used)
            return start

    def _release(self, start: int, count: int):
        with self.lock:
            if start != self.tail or count > self.used:
                raise CaptureError("teacher ring release order is corrupt")
            self.tail = (self.tail + count) % self.capacity
            self.used -= count

    def enqueue(
        self,
        raw_logits: torch.Tensor,
        row_indices: list[int],
        targets: list[tuple[RequestCaptureContext, int]],
    ):
        """Own selected raw rows without waiting for background compaction."""
        count = len(row_indices)
        if self.closed or self.stop.is_set():
            raise CaptureError("teacher batcher is closed")
        if (
            count < 1
            or count != len(targets)
            or raw_logits.ndim != 2
            or raw_logits.device != self.device
            or raw_logits.dtype != self.storage.dtype
            or raw_logits.shape[1] < self.vocab_size
            or any(
                type(row) is not int or not 0 <= row < raw_logits.shape[0]
                for row in row_indices
            )
        ):
            raise ContractError("raw teacher rows differ from the async ring contract")
        for context, position in targets:
            context.validate_async_teacher(position=position)

        start = self._reserve(count)
        try:
            ready = torch.cuda.Event()
            producer = torch.cuda.current_stream(self.device)
            scores = raw_logits[:, : self.vocab_size]
            offset = 0
            while offset < count:
                destination = (start + offset) % self.capacity
                take = 1
                while (
                    offset + take < count
                    and destination + take < self.capacity
                    and row_indices[offset + take] == row_indices[offset] + take
                ):
                    take += 1
                self.storage.narrow(0, destination, take).copy_(
                    scores.narrow(0, row_indices[offset], take), non_blocking=True
                )
                offset += take
            ready.record(producer)
        except Exception as error:
            # CUDA may have accepted a prefix of the copies. Do not recycle its
            # storage until process teardown synchronizes the device.
            with self.lock:
                self.failures += 1
                self.fatal_error = error
                self.stop.set()
            raise

        destinations = []
        try:
            for context, position in targets:
                row = context.reserve_async_teacher(position=position)
                destinations.append(
                    _Destination(
                        staging=context.teacher_staging,
                        completion=context.teacher_completion,
                        row=row,
                    )
                )
        except Exception as error:
            self.work.put(_Job(start, count, tuple(destinations), ready, error))
            raise
        offset = 0
        while offset < count:
            take = min(self.batch_rows, count - offset)
            self.work.put(
                _Job(
                    (start + offset) % self.capacity,
                    take,
                    tuple(destinations[offset : offset + take]),
                    ready,
                )
            )
            offset += take

    def _next_batch(self, carry: _Job | None):
        if carry is None:
            try:
                carry = self.work.get(timeout=0.05)
            except queue.Empty:
                return [], None
        jobs = [carry]
        rows = carry.count
        if carry.error is not None:
            return jobs, None
        carry = None
        while rows < self.batch_rows:
            try:
                candidate = self.work.get_nowait()
            except queue.Empty:
                break
            if candidate.error is not None or rows + candidate.count > self.batch_rows:
                carry = candidate
                break
            jobs.append(candidate)
            rows += candidate.count
        return jobs, carry

    def _compact(self, jobs: list[_Job]):
        total = sum(job.count for job in jobs)
        entries = tuple(entry for job in jobs for entry in job.destinations)
        if len(entries) != total:
            raise jobs[0].error or CaptureError(
                "teacher batch destination is incomplete"
            )
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            for job in jobs:
                self.stream.wait_event(job.ready)
            offset = 0
            physical = jobs[0].start
            while offset < total:
                take = min(total - offset, self.capacity - physical)
                rows = capture_teacher(
                    self.storage.narrow(0, physical, take),
                    self.vocab_size,
                    topk_backend=self.topk_backend,
                )
                for source_row in range(take):
                    destination = entries[offset + source_row]
                    destination.staging.write_reserved(
                        rows, row=source_row, start=destination.row
                    )
                offset += take
                physical = 0
            complete = torch.cuda.Event()
            complete.record(self.stream)
        complete.synchronize()

    def _finish(
        self, jobs: list[_Job], error: BaseException | None, *, reusable: bool = True
    ):
        for job in jobs:
            for destination in job.destinations:
                if not reusable:
                    destination.staging.transfer_uncertain = True
                destination.completion.finish(error)
            if reusable:
                self._release(job.start, job.count)
            self.work.task_done()
        with self.lock:
            if error is None:
                self.batches += 1
                self.processed_rows += sum(job.count for job in jobs)
            else:
                self.failures += sum(job.count for job in jobs)

    def _worker(self):
        carry = None
        while not self.stop.is_set() or carry is not None or not self.work.empty():
            jobs, carry = self._next_batch(carry)
            if not jobs:
                continue
            error = self.fatal_error or jobs[0].error
            reusable = self.fatal_error is None
            try:
                if not reusable:
                    pass
                elif error is None:
                    self._compact(jobs)
                else:
                    # Reservation can fail after the producer has queued the
                    # D2D copies. Preserve ring ownership until those copies
                    # finish even though no compaction will consume them.
                    for job in jobs:
                        job.ready.synchronize()
            except Exception as exc:  # noqa: BLE001 - failure belongs to samples
                error = exc
                try:
                    torch.cuda.synchronize(self.device)
                except Exception as sync_error:  # noqa: BLE001
                    error = sync_error
                    with self.lock:
                        self.fatal_error = sync_error
                        self.stop.set()
                    reusable = False
            try:
                self._finish(jobs, error, reusable=reusable)
            except Exception:  # noqa: BLE001 - retain storage after corruption
                # Ring ownership corruption is process-fatal for capture. Keep
                # storage alive and fail all future admissions.
                with self.lock:
                    self.failures += sum(job.count for job in jobs)
                self.stop.set()

    def stats(self):
        with self.lock:
            return {
                "capacity_rows": self.capacity,
                "batch_rows": self.batch_rows,
                "occupied_rows": self.used,
                "max_occupied_rows": self.max_occupancy,
                "accepted_rows": self.accepted_rows,
                "processed_rows": self.processed_rows,
                "backpressure_rows": self.backpressure_rows,
                "batches": self.batches,
                "failures": self.failures,
                "allocated_bytes": self.allocated_bytes,
            }

    def close(self):
        if self.closed:
            return
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=20)
        if self.thread.is_alive() or self.used:
            raise CaptureError("teacher batcher did not drain")
        self.closed = True


class FinalHiddenCapture:
    """Request-owned hidden rows, copied before model outputs can be overwritten."""

    def __init__(self, *, device, spec):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.spec = spec
        self.rows = 0
        self.closed = False
        self.copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

    def append(self, hidden, indices, targets):
        if self.closed:
            raise CaptureError("hidden capture is closed")
        if (
            hidden is None
            or hidden.ndim != 2
            or hidden.dtype != torch.bfloat16
            or hidden.device != self.device
            or hidden.shape[1] != self.spec.hidden_size
            or not indices
            or len(indices) != len(targets)
            or any(
                type(row) is not int or not 0 <= row < hidden.shape[0]
                for row in indices
            )
        ):
            raise ContractError("hidden output differs from the capture contract")
        for context, position in targets:
            context.validate_async_teacher(position=position)
        stream = torch.cuda.current_stream(self.device) if hidden.is_cuda else None
        for row, (context, _) in zip(indices, targets, strict=True):
            staging = context.teacher_staging
            if staging.stream is not None and staging.stream != stream:
                stream.wait_stream(staging.stream)
            offset = context.teacher_rows
            staging.reserve(start=offset, count=1)
            try:
                staging.device_tensors["teacher_final_hidden"][offset].copy_(
                    hidden[row]
                )
            except Exception:
                context.transfer_uncertain = True
                raise
            staging.stream = stream
            context.teacher_rows += 1
            self.rows += 1
        if stream is not None:
            hidden.record_stream(stream)

    def seal(self, staging):
        if staging is None or not staging.end or self.copy_stream is None:
            return
        event = torch.cuda.Event()
        event.record(staging.stream)
        self.copy_stream.wait_event(event)
        staging.stream = self.copy_stream

    def stats(self):
        return {
            "hidden_rows": self.rows,
            "processed_rows": self.rows,
            "pending_rows": 0,
        }

    def close(self):
        if not self.closed and self.copy_stream is not None:
            self.copy_stream.synchronize()
        self.closed = True
