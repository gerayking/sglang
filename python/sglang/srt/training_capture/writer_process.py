"""CPU snapshot publication in a fresh process over bounded shared Host slots."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Connection
from typing import ClassVar

import msgspec
import torch
import torch.multiprocessing as mp
from sglang.srt.training_capture.catalog import (
    CatalogConflict,
    CatalogError,
    CatalogUnavailable,
    HTTPCaptureCatalog,
)
from sglang.srt.training_capture.mooncake_store import (
    MooncakeSnapshotStore,
    TransportError,
)
from sglang.srt.training_capture.protocol import DTYPES, CaptureError, ContractError
from sglang.srt.training_capture.snapshot import prepare_snapshot
from sglang.srt.training_capture.snapshot_writer import (
    PublicationJournal,
    SnapshotWriter,
)
from sglang.srt.training_capture.timings import CaptureTimings


class WriterProcessError(TransportError):
    """IPC completion is unknown. Do not reuse the submitted Host slot."""


class ProcessTimings(CaptureTimings):
    def __init__(self):
        super().__init__()
        self.remote = {}

    def update_remote(self, values):
        with self.lock:
            self.remote = values

    def stats(self):
        with self.lock:
            result = {stage: dict(value) for stage, value in self.values.items()}
            for stage, value in self.remote.items():
                for field, count in value.items():
                    if field == "max_seconds":
                        result[stage][field] = max(result[stage][field], count)
                    else:
                        result[stage][field] += count
            return result


def _connect_writer(config):
    catalog = HTTPCaptureCatalog(
        config.catalog_endpoint,
        bearer_token=(
            os.environ[config.catalog_token_env] if config.catalog_token_env else None
        ),
        timeout=config.http_timeout_seconds,
        attempts=config.http_attempts,
    )
    store = MooncakeSnapshotStore.connect(
        msgspec.to_builtins(config.store),
        replica_num=config.replica_num,
        max_receive_bytes=config.max_host_bytes,
        payload_hash_workers=config.payload_hash_workers,
    )
    try:
        return SnapshotWriter(
            store, catalog, PublicationJournal(config.journal_directory)
        )
    except Exception:
        store.close()
        raise


def _writer_main(
    connection, config, arena, layout, slot_bytes, manifest_offset, factory
):
    # CUDA is hidden by the launcher before any child imports execute.
    torch.set_num_threads(1)
    writer = None
    slots = []
    sequence = 0

    def reply(kind, value=None):
        connection.send(
            (
                kind,
                sequence,
                value,
                writer.timings.stats() if writer else {},
                [
                    index
                    for index, (storage, _, _) in enumerate(slots)
                    if writer and storage.data_ptr() in writer.store.quarantined
                ],
            )
        )

    def checkpoint(kind, value=None):
        connection.send((kind, sequence, value))
        ack_sequence, error = connection.recv()
        if ack_sequence != sequence or error is not None:
            raise ContractError(error or "writer checkpoint sequence mismatch")

    try:
        if arena.device.type != "cpu" or not arena.is_shared():
            raise ContractError("writer requires shared CPU storage")
        writer = factory(config)
        for start in range(0, arena.numel(), slot_bytes):
            storage = arena[start : start + slot_bytes]
            tensors = {
                name: storage[offset : offset + length]
                .view(DTYPES[dtype])
                .reshape(shape)
                for name, (offset, length, dtype, shape) in layout.items()
            }
            slots.append((storage, tensors, storage[manifest_offset:]))
            writer.store.register(storage)
        reply("ok")
        while True:
            sequence, operation, payload = connection.recv()
            try:
                if operation == "write":
                    index, metadata, valid_tokens, lease = payload
                    _, buffers, manifest_buffer = slots[index]
                    manifest, tensors = writer.timings.call(
                        "snapshot_build",
                        prepare_snapshot,
                        metadata,
                        buffers,
                        valid_kv_tokens=valid_tokens,
                        payload_hasher=writer.store.payload_hasher,
                    )
                    result = writer.write(
                        manifest,
                        tensors,
                        manifest_buffer,
                        lease,
                        check_current=lambda: checkpoint("check"),
                        on_prepared=lambda lease, data: checkpoint(
                            "prepared", (lease, data)
                        ),
                    )
                elif operation == "recover_prepared":
                    result = writer.recover_prepared(*payload)
                elif operation == "recover":
                    writer.recover()
                    result = None
                elif operation == "begin":
                    result = writer.catalog.begin(payload)
                elif operation == "close":
                    writer.store.close()
                    writer.journal.close()
                    reply("ok")
                    return
                else:
                    raise ContractError("unknown writer operation")
                reply("ok", result)
            except Exception as error:  # noqa: BLE001 - serialize failures across IPC
                reply("error", (type(error).__name__, str(error)[:1024]))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:  # noqa: BLE001 - report startup failures to the parent
        reply("error", (type(error).__name__, str(error)[:1024]))
    finally:
        if writer is not None:
            try:
                writer.store.close()
            finally:
                writer.journal.close()
        connection.close()


class ProcessSnapshotWriter:
    """One in-flight command; the coordinator's bounded queue owns backpressure.

    Only startup shares the arena handle. Per-request messages contain a slot
    index and metadata, never tensor payloads. ACKs fence slot reuse; prepared
    bytes are acknowledged before any potentially ambiguous publication.
    """

    _retained: ClassVar[set[ProcessSnapshotWriter]] = set()

    def __init__(self, config, pool, *, writer_factory=_connect_writer):
        self.pool = pool
        self.timeout = config.writer_process_timeout_seconds
        self.timings = ProcessTimings()
        self.quarantined = set()
        self.failed = False
        self.closed = False
        self.sequence = 0
        self.rpc_lock = threading.Lock()
        self.indices = {id(slot): index for index, slot in enumerate(pool.slots)}
        self.connection, child = mp.Pipe()
        self.process = None
        try:
            pool.shared_writer_active = True
            # multiprocessing.spawn imports __main__ before its target, which can
            # initialize serving's CUDA runtime. exec supplies a CPU-only env from
            # the outset without changing the serving process's environment.
            self.process = subprocess.Popen(
                [sys.executable, "-m", __name__, str(child.fileno())],
                pass_fds=(child.fileno(),),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
            )
            child.close()
            self.connection.send(
                (bytes(mp.current_process().authkey), sys.path, os.getpid())
            )
            self.connection.send(
                (
                    config,
                    pool.shared_arena,
                    pool.layout,
                    pool.slot_bytes,
                    pool.manifest_offset,
                    writer_factory,
                )
            )
            self._receive()
        except BaseException:
            child.close()
            self._stop()
            raise

    def _stop(self):
        self.connection.close()
        if self.process is None:
            self.pool.shared_writer_active = False
            return
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self.process.poll() is None:
            self._retained.add(self)
            raise WriterProcessError("writer process did not stop; retain Host storage")
        self.pool.shared_writer_active = False
        self._retained.discard(self)

    def _receive(self, *, check_current=None, on_prepared=None):
        deadline = time.monotonic() + self.timeout
        callback_error = None
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("snapshot writer command timed out")
            if not self.connection.poll(0.1):
                if self.process.poll() is not None:
                    raise EOFError("snapshot writer exited without completion ACK")
                continue
            kind, sequence, value, *state = self.connection.recv()
            if sequence != self.sequence:
                raise OSError("snapshot writer reply sequence mismatch")
            if kind in ("check", "prepared"):
                try:
                    if kind == "check":
                        check_current()
                    else:
                        on_prepared(*value)
                    error = None
                except Exception as caught:  # noqa: BLE001 - drain the child ACK first
                    callback_error = caught
                    error = str(caught)[:1024] or type(caught).__name__
                self.connection.send((sequence, error))
                continue
            timings, quarantined = state
            self.timings.update_remote(timings)
            self.quarantined.update(
                self.pool.slots[index].storage.data_ptr() for index in quarantined
            )
            if callback_error is not None:
                raise callback_error
            if kind == "error":
                name, message = value
                errors = (
                    TransportError,
                    ContractError,
                    CatalogConflict,
                    CatalogError,
                    CatalogUnavailable,
                )
                error_type = next(
                    (e for e in errors if e.__name__ == name), CaptureError
                )
                raise error_type(f"snapshot writer: {name}: {message}")
            if kind != "ok":
                raise OSError("unknown snapshot writer reply")
            return value

    def _call(self, operation, payload=None, **callbacks):
        # Lease refill and publication share one ordered command channel.
        with self.rpc_lock:
            return self._call_locked(operation, payload, **callbacks)

    def _call_locked(self, operation, payload, **callbacks):
        if self.failed or self.closed:
            raise WriterProcessError("snapshot writer is unavailable")
        self.sequence += 1
        try:
            self.connection.send((self.sequence, operation, payload))
            return self._receive(**callbacks)
        except (OSError, EOFError) as error:
            self.failed = True
            self._stop()
            raise WriterProcessError(str(error)) from error

    def begin(self, identity):
        return self._call("begin", identity)

    def build_and_write(self, slot, metadata, valid_tokens, lease, **callbacks):
        index = self.indices[id(slot)]
        try:
            return self._call(
                "write", (index, metadata, valid_tokens, lease), **callbacks
            )
        except TransportError:
            self.quarantined.add(slot.storage.data_ptr())
            raise

    def recover_prepared(self, lease, data):
        return self._call("recover_prepared", (lease, data))

    def recover(self):
        return self._call("recover")

    def close(self):
        if self.closed:
            return
        try:
            if not self.failed and self.process.poll() is None:
                self._call("close")
        finally:
            self._stop()
        self.closed = True


if __name__ == "__main__":
    from sglang.srt.utils import kill_itself_when_parent_died

    kill_itself_when_parent_died()
    connection = Connection(int(sys.argv[1]))
    authkey, paths, parent_pid = connection.recv()
    if os.getppid() != parent_pid:
        sys.exit(1)
    # Torch's CPU-storage reducer transfers an existing shared-memory FD through
    # multiprocessing's authenticated resource sharer; no payload is serialized.
    mp.current_process().authkey = authkey
    sys.path[:] = paths
    _writer_main(connection, *connection.recv())
