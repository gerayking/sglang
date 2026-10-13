"""Request-owned position ledger, independent of mutable scheduler batches."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from sglang.srt.training_capture.codec import TOP128_CODEC
from sglang.srt.training_capture.host_pool import HostSlot
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.kv_staging import KVStaging
from sglang.srt.training_capture.protocol import (
    ContractError,
    SequenceInfo,
    validate_tensors,
)
from sglang.srt.training_capture.snapshot import (
    SealedCapture,
    SnapshotMetadata,
    prepare_snapshot,
    prepare_snapshot_partition,
)
from sglang.srt.training_capture.teacher import TeacherRows
from sglang.srt.training_capture.teacher_staging import (
    TeacherCompletion,
    TeacherStaging,
)
from sglang.srt.training_capture.topology import CapturePartition


class RequestCaptureContext:
    """A single generation's accepted tokens, teacher rows, KV and completion.

    Only the inference thread mutates a collecting context. After seal/abort it
    transfers ownership to the writer, which never reads the serving request.
    This initial codec uses ordinary, zero-based full-attention positions.
    """

    def __init__(
        self,
        *,
        slot: HostSlot | None,
        prompt_ids: tuple[int, ...],
        max_tokens: int,
        vocab_size: int,
        partition: CapturePartition | None = None,
        teacher_batcher=None,
        codec=TOP128_CODEC,
    ):
        if not prompt_ids or len(prompt_ids) >= max_tokens:
            raise ContractError(
                "capture requires a nonempty prompt and response capacity"
            )
        if any(not 0 <= token < vocab_size for token in prompt_ids):
            raise ContractError("prompt token outside target vocabulary")
        self.partition = partition
        self.codec = codec
        self.owns_aux = partition is None or partition.include_aux
        self.owns_kv = partition is None or bool(partition.heads)
        active = partition is None or partition.active
        if active != (slot is not None):
            raise ContractError("context storage differs from payload ownership")
        if partition is not None and active:
            expected = (
                set(codec.aux_specs(max_tokens, max_tokens)) if self.owns_aux else set()
            )
            for heads in partition.heads:
                for component in ("k", "v"):
                    name = f"target_{component}.{heads.layer_id}"
                    expected.add(name)
                    value = slot.tensors.get(name)
                    if (
                        value is None
                        or value.ndim != 3
                        or value.shape[0] < max_tokens
                        or value.shape[1] != heads.end - heads.start
                    ):
                        raise ContractError(
                            "request buffers differ from local KV ownership"
                        )
            if not partition.active or set(slot.tensors) != expected:
                raise ContractError("request buffers differ from partition ownership")
        self.slot = slot
        self.prompt_length = len(prompt_ids)
        self.token_ids = list(prompt_ids)
        self.pending_tokens = {}
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.kv_end = 0
        self.kv_source_stream = None
        self.teacher_rows = 0
        self.teacher_batcher = teacher_batcher
        self.teacher_completion = TeacherCompletion()
        self.last_event = None
        self.last_stream = None
        self.kv_staging = (
            KVStaging(slot.device_tensors, slot.tensors)
            if slot is not None and slot.device_tensors is not None
            else None
        )
        self.teacher_staging = (
            TeacherStaging(slot.teacher_device_tensors, slot.tensors)
            if slot is not None and slot.teacher_device_tensors is not None
            else None
        )
        if teacher_batcher is not None and (
            not self.owns_aux
            or self.teacher_staging is None
            or self.teacher_staging.capacity < max_tokens
        ):
            raise ContractError(
                "async teacher capture requires full request-owned device staging"
            )
        self.transfer_uncertain = False
        self.l2_exporter = None
        self.l2_lease = None
        self.l2_materialized = False
        self.state = "COLLECTING"
        self.failure_reason = None
        self.sequence = None
        self.sealed = None
        if self.owns_aux:
            self.slot.tensors["position_ids"][:max_tokens].copy_(
                torch.arange(max_tokens)
            )
            self.slot.tensors["loss_mask"][:max_tokens].zero_()
            self.slot.tensors["kv_valid"][:max_tokens].zero_()

    def _collecting(self):
        if self.state != "COLLECTING":
            raise ContractError("capture is no longer collecting")

    @property
    def transfer_uncertain(self):
        return self._transfer_uncertain or bool(
            self.teacher_staging and self.teacher_staging.transfer_uncertain
        )

    @transfer_uncertain.setter
    def transfer_uncertain(self, value):
        self._transfer_uncertain = value

    def _sealed(self):
        if self.state != "SEALED":
            raise ContractError("only a sealed request can prepare a snapshot")

    def _record_completion(self, device):
        if device.type == "cuda":
            try:
                stream = torch.cuda.current_stream(device)
                if self.last_event is not None and stream != self.last_stream:
                    stream.wait_event(self.last_event)
                event = torch.cuda.Event()
                event.record(stream)
                self.last_event = event
                self.last_stream = stream
            except Exception:
                self.transfer_uncertain = True
                raise

    def export_kv(
        self, exporter: SelectedLayerKVExporter, slots: torch.Tensor, *, end: int
    ):
        self._collecting()
        if not self.owns_kv:
            raise ContractError("aux-only context cannot export KV payloads")
        if not self.kv_end <= end <= self.max_tokens:
            raise ContractError("KV positions retracted or exceeded capture capacity")
        if end == self.kv_end:
            return
        start = self.kv_end
        bound_exporter = self.slot.kv_exporter or exporter
        try:
            if self.kv_staging is None:
                bound_exporter.export(slots, self.slot.tensors, start, end)
            else:
                self.kv_staging.export(bound_exporter, slots, start=start, end=end)
        finally:
            if self.kv_staging is not None and self.kv_staging.transfer_uncertain:
                self.transfer_uncertain = True
            # Also fence copies queued before a later layer raises.
            self._record_completion(exporter.device)
        self.kv_end = end

    def record_kv_progress(self, *, end: int):
        """A rank without KV payload records its locally computed prefix."""
        self._collecting()
        if self.owns_kv or not self.kv_end <= end <= self.max_tokens:
            raise ContractError(
                "KV owners must advance through completed export enqueue"
            )
        self.kv_end = end

    def defer_kv_progress(self, *, end: int, device=None):
        """Record coverage whose KV bytes will be materialized from HiCache L2."""
        self._collecting()
        if not self.owns_kv or not self.kv_end <= end <= self.max_tokens:
            raise ContractError("invalid deferred L2 KV coverage")
        if device is not None and torch.device(device).type == "cuda":
            self.kv_source_stream = torch.cuda.current_stream(device)
        self.kv_end = end

    def bind_l2_snapshot(self, exporter, lease):
        """Attach a protected, completed-prefix L2 source before sealing."""
        self._collecting()
        if (
            not self.owns_kv
            or self.l2_lease is not None
            or lease.host_indices.numel() != self.kv_end
        ):
            lease.release()
            raise ContractError("L2 snapshot lease differs from KV coverage")
        self.l2_exporter = exporter
        self.l2_lease = lease

    def record_positions(self, positions: torch.Tensor, *, start: int):
        self._collecting()
        if not self.owns_aux:
            raise ContractError("only the aux owner records position payloads")
        end = start + positions.numel()
        if positions.ndim != 1 or not 0 <= start < end <= self.kv_end:
            raise ContractError("forward positions do not cover the exported KV range")
        try:
            self.slot.tensors["position_ids"][start:end].copy_(
                positions, non_blocking=positions.is_cuda
            )
            if positions.is_cuda:
                positions.record_stream(torch.cuda.current_stream(positions.device))
        finally:
            self._record_completion(positions.device)

    def record_teacher(self, rows: TeacherRows, *, row: int, position: int):
        if position != self.kv_end:
            raise ContractError(
                "teacher prediction is not aligned with the computed KV prefix"
            )
        self.record_teacher_range(rows, row=row, position=position, count=1)

    def record_teacher_range(
        self, rows: TeacherRows, *, row: int, position: int, count: int
    ):
        self._collecting()
        if not self.owns_aux:
            raise ContractError("only the aux owner records teacher payloads")
        if row < 0 or row + count > rows.logits.shape[0]:
            raise ContractError(
                "teacher range is missing, duplicated, or shifted: "
                f"count={count} position={position} "
                f"expected_position={self.prompt_length + self.teacher_rows} "
                f"max_tokens={self.max_tokens} kv_end={self.kv_end} "
                f"row={row} available_rows={rows.logits.shape[0]}"
            )
        self._validate_teacher_range(position=position, count=count)
        index = self.teacher_rows
        try:
            if self.teacher_staging is not None:
                self.teacher_staging.append(rows, row=row, start=index, count=count)
            else:
                for name, source in (
                    ("teacher_topk_ids", rows.token_ids),
                    ("teacher_topk_logits", rows.logits),
                    ("teacher_logsumexp", rows.logsumexp),
                ):
                    self.slot.tensors[name][index : index + count].copy_(
                        source[row : row + count], non_blocking=source.is_cuda
                    )
                    if source.is_cuda:
                        source.record_stream(torch.cuda.current_stream(source.device))
        finally:
            if (
                self.teacher_staging is not None
                and self.teacher_staging.transfer_uncertain
            ):
                self.transfer_uncertain = True
            self._record_completion(
                self.teacher_staging.device
                if self.teacher_staging is not None
                else rows.logits.device
            )
        self.teacher_rows += count

    def _validate_teacher_range(self, *, position: int, count: int):
        if (
            count < 1
            or position != self.prompt_length + self.teacher_rows
            or position + count > self.max_tokens
            or position + count - 1 > self.kv_end
        ):
            raise ContractError(
                "teacher range is missing, duplicated, or shifted: "
                f"count={count} position={position} "
                f"expected_position={self.prompt_length + self.teacher_rows} "
                f"max_tokens={self.max_tokens} kv_end={self.kv_end}"
            )

    def validate_async_teacher(self, *, position: int):
        self._collecting()
        if self.teacher_batcher is None or not self.owns_aux:
            raise ContractError("request does not use async teacher capture")
        self._validate_teacher_range(position=position, count=1)

    def reserve_async_teacher(self, *, position: int):
        self.validate_async_teacher(position=position)
        index = self.teacher_rows
        self.teacher_staging.reserve(start=index, count=1)
        self.teacher_rows += 1
        self.teacher_completion.reserve()
        return index

    def write_async_teacher(self, rows: TeacherRows, *, row: int, start: int):
        self.teacher_staging.write_reserved(rows, row=row, start=start)

    def finish_async_teacher(self, error: BaseException | None):
        self.teacher_completion.finish(error)

    def _wait_for_async_teacher(self):
        if self.teacher_batcher is None:
            return
        self.teacher_completion.wait()

    def trim_terminal_prefix(self):
        """Discard lookahead/verify suffixes past the scheduler's output boundary."""
        self._collecting()
        n = len(self.token_ids)
        response_length = n - self.prompt_length
        if (
            response_length < 1
            or (self.owns_aux and response_length > self.teacher_rows)
            or self.kv_end < n - 1
        ):
            raise ContractError("terminal output has incomplete teacher or KV coverage")
        if self.owns_aux:
            self.teacher_rows = response_length
        self.kv_end = min(self.kv_end, n)

    def commit_token(self, *, position: int, token_id: int):
        self._collecting()
        if (
            position != len(self.token_ids)
            or position >= self.max_tokens
            or position > self.kv_end
            or (self.owns_aux and position >= self.prompt_length + self.teacher_rows)
        ):
            raise ContractError("accepted token has no matching teacher row")
        if not 0 <= token_id < self.vocab_size:
            raise ContractError("accepted token outside target vocabulary")
        if (
            position in self.pending_tokens
            and self.pending_tokens[position] != token_id
        ):
            raise ContractError("scheduler token differs from the observed target path")
        self.token_ids.append(token_id)
        self.pending_tokens.pop(position, None)

    def observe_tokens(self, *, position: int, tokens: list[int]):
        """Preserve the model's path until delayed scheduler results confirm it."""
        self._collecting()
        if position < 0:
            raise ContractError("observed token position must be nonnegative")
        for offset, token in enumerate(tokens):
            index = position + offset
            if index >= self.max_tokens:
                break
            if not 0 <= token < self.vocab_size:
                raise ContractError("observed token outside target vocabulary")
            if index < len(self.token_ids):
                if self.token_ids[index] != token:
                    raise ContractError("observed path differs from committed tokens")
            elif index in self.pending_tokens:
                if self.pending_tokens[index] != token:
                    raise ContractError("observed target path changed before commit")
            elif index == len(self.token_ids) + len(self.pending_tokens):
                self.pending_tokens[index] = token
            else:
                raise ContractError("observed target path contains a gap")

    def seal(self, stop_reason: str):
        self._collecting()
        n = len(self.token_ids)
        r = n - self.prompt_length
        if (
            not r
            or (self.owns_aux and r != self.teacher_rows)
            or self.kv_end not in (n - 1, n)
        ):
            raise ContractError("cannot seal an incomplete response or KV prefix")
        self._flush_kv()
        if self.teacher_batcher is None:
            self._flush_staging(self.teacher_staging)
        else:
            try:
                self.teacher_batcher.seal(self.teacher_staging)
            except Exception:
                self.transfer_uncertain = True
                raise
        self.sequence = SequenceInfo(
            prompt_length=self.prompt_length,
            response_length=r,
            total_length=n,
            stop_reason=stop_reason,
        )
        self.sealed = SealedCapture(
            sequence=self.sequence,
            token_ids=tuple(self.token_ids),
            valid_kv_tokens=self.kv_end,
            teacher_rows=r,
        )
        self.state = "SEALED"

    def _flush_kv(self):
        self._flush_staging(self.kv_staging)

    def _flush_staging(self, staging):
        if staging is None or staging.start == staging.end:
            return
        # Finalization may run outside the model's forward-stream context.
        with (
            torch.cuda.stream(staging.stream)
            if staging.stream is not None
            else nullcontext()
        ):
            try:
                staging.flush()
            finally:
                self._record_completion(staging.device)

    def abort(self, reason: str):
        if self.state in ("COLLECTING", "SEALED"):
            self.state = "FAILED"
            self.failure_reason = reason
        if self.l2_lease is not None:
            self.l2_lease.release()
            self.l2_lease = None

    def wait_for_copies(self):
        self._wait_for_async_teacher()
        if self.teacher_batcher is not None and self.state == "SEALED":
            if self.teacher_staging.end != self.teacher_rows:
                self.teacher_staging.truncate(self.teacher_rows)
            self._flush_staging(self.teacher_staging)
        elif (
            self.teacher_staging is not None and self.teacher_staging.stream is not None
        ):
            # An aborted hidden capture still owns its producer's D2D writes.
            with torch.cuda.stream(self.teacher_staging.stream):
                self._record_completion(self.teacher_staging.device)
        if self.transfer_uncertain:
            raise ContractError("CUDA copy completion is uncertain")
        if self.last_event is not None:
            try:
                self.last_event.synchronize()
            except Exception:
                self.transfer_uncertain = True
                raise
        if self.l2_lease is not None:
            lease, self.l2_lease = self.l2_lease, None
            try:
                self.l2_exporter.materialize(lease, self.slot.tensors, end=self.kv_end)
                self.l2_materialized = True
            except Exception:
                self.transfer_uncertain = True
                raise

        if self.state == "SEALED" and self.slot is not None and not self.slot.host_ready:
            sealed = self.sealed
            if self.owns_aux:
                n, p = sealed.sequence.total_length, sealed.sequence.prompt_length
                torch.arange(
                    p,
                    n,
                    out=self.slot.tensors["logits_positions"][: sealed.teacher_rows],
                )
                self.slot.tensors["token_ids"][:n].copy_(
                    torch.tensor(sealed.token_ids, dtype=torch.int32)
                )
                self.slot.tensors["loss_mask"][:n].zero_()
                self.slot.tensors["loss_mask"][p:n] = 1
                self.slot.tensors["kv_valid"][:n].zero_()
                self.slot.tensors["kv_valid"][: sealed.valid_kv_tokens] = 1
            self.slot.host_ready = True

    def snapshot(self, **metadata):
        """Return a fully validated snapshot for direct consumers."""
        result = self.prepare_snapshot(**metadata)
        validate_tensors(*result, _codec=self.codec)
        self._sealed()
        return result

    def prepare_snapshot(self, *, payload_hasher=None, **metadata):
        """Prepare owned views for mandatory validation by SnapshotWriter."""
        if self.partition is not None:
            raise ContractError(
                "partitioned requests require coordinated snapshot assembly"
            )
        self._sealed()
        self.wait_for_copies()
        result = prepare_snapshot(
            SnapshotMetadata(sequence=self.sealed.sequence, **metadata),
            self.slot.tensors,
            valid_kv_tokens=self.sealed.valid_kv_tokens,
            payload_hasher=payload_hasher,
        )
        self._sealed()
        return result

    def prepare_partition(self, *, payload_hasher=None, **metadata):
        if self.partition is None or not self.partition.active:
            raise ContractError(
                "partition preparation requires an active partitioned request"
            )
        self._sealed()
        self.wait_for_copies()
        result = prepare_snapshot_partition(
            SnapshotMetadata(sequence=self.sealed.sequence, **metadata),
            self.slot.tensors,
            valid_kv_tokens=self.sealed.valid_kv_tokens,
            partition=self.partition,
            token_ids=self.sealed.token_ids,
            payload_hasher=payload_hasher,
        )
        self._sealed()
        return result
