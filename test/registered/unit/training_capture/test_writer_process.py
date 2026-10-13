"""Spawn isolation, shared payload ownership and ambiguous publication recovery."""

import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
from sglang.srt.training_capture.catalog import CaptureLease, CatalogUnavailable
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.mooncake_store import (
    MooncakeSnapshotStore,
    TransportError,
)
from sglang.srt.training_capture.protocol import (
    CaptureError,
    ContractError,
    decode_manifest,
)
from sglang.srt.training_capture.snapshot import SnapshotMetadata
from sglang.srt.training_capture.snapshot_writer import (
    PublicationJournal,
    SnapshotWriter,
)
from sglang.srt.training_capture.writer_process import (
    ProcessSnapshotWriter,
    WriterProcessError,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.training_capture_utils import (
    BufferStore,
    FakeReplicateConfig,
    make_snapshot,
    read_snapshot,
)

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


class ProcessCatalog:
    def __init__(self, state, failure):
        self.state, self.failure = state, failure

    def objects(self, lease, payload):
        return {}

    def begin(self, identity):
        self.state["begin_pid"] = os.getpid()
        if self.failure == "begin":
            self.failure = None
            raise CatalogUnavailable("begin temporarily unavailable")
        return CaptureLease(
            capture_id="capture-new",
            fencing_token=1,
            dataset_id=identity["dataset_id"],
            sample_id=identity["sample_id"],
            generation_id=identity["generation_id"],
            expires_in_seconds=120,
            renew_after_seconds=20,
        )

    def seal(self, lease, payload):
        return {"state": "PREPARED", "manifest_sha256": payload["manifest"]["sha256"]}

    def publish(self, payload):
        previous = self.state.get("publication")
        if previous is not None:
            assert previous == payload
        self.state["publication"] = payload
        if self.failure == "publish":
            self.failure = None
            raise CatalogUnavailable("committed but reply lost")
        return {"state": "AVAILABLE", "publication_id": "p1", "catalog_cursor": "c1"}


def process_writer_factory(config, *, data, state, failure=None):
    state["pid"] = os.getpid()
    state["cuda_initialized"] = torch.cuda.is_initialized()
    client = BufferStore()
    client.data = data
    if failure == "transport":
        client.put_status = -1
    return SnapshotWriter(
        MooncakeSnapshotStore(client, FakeReplicateConfig()),
        ProcessCatalog(state, failure),
        PublicationJournal(config.journal_directory),
    )


class TestProcessWriter(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manager = mp.get_context("spawn").Manager()
        self.addCleanup(self.manager.shutdown)
        self.data, self.state = self.manager.dict(), self.manager.dict()
        self.manifest, self.tensors = make_snapshot()
        self.config = CaptureConfig(
            dataset_id=self.manifest.dataset_id,
            model_id="test",
            producer_revision="test",
            selected_layer_ids=self.manifest.kv.selected_layer_ids,
            catalog_endpoint="http://localhost:1",
            journal_directory=self.directory.name,
            store=StoreSetup(
                local_hostname="localhost", master_server_addr="localhost:1"
            ),
            snapshot_writer_backend="process",
        )
        self.pool = HostBufferPool(
            kv=self.manifest.kv,
            max_tokens=8,
            slots=1,
            max_bytes=2 << 20,
            registrar=None,
            shared=True,
            pin_memory=False,
        )
        self.slot = self.pool.acquire()
        self.metadata = SnapshotMetadata(
            **{
                field: getattr(self.manifest, field)
                for field in SnapshotMetadata.__struct_fields__
            }
        )
        self.lease = CaptureLease(
            capture_id="capture-1",
            fencing_token=1,
            dataset_id=self.manifest.dataset_id,
            sample_id=self.manifest.sample_id,
            generation_id=self.manifest.generation_id,
            expires_in_seconds=120.0,
            renew_after_seconds=20.0,
        )
        self.prepared = None

    def start(self, *, failure=None):
        writer = ProcessSnapshotWriter(
            self.config,
            self.pool,
            writer_factory=partial(
                process_writer_factory,
                data=self.data,
                state=self.state,
                failure=failure,
            ),
        )
        self.addCleanup(writer.close)
        # Fill after spawn: the writer must observe the same storage, not a copy.
        for descriptor in self.manifest.objects:
            target = self.slot.tensors[descriptor.name]
            value = self.tensors[descriptor.key]
            if descriptor.kind == "kv":
                start, end = descriptor.token_range
                target[start:end].copy_(value)
            else:
                target[: len(value)].copy_(value)
        return writer

    def on_prepared(self, lease, data):
        self.prepared = (lease, data)
        self.assertIsNone(self.pool.acquire())

    def write(self, writer, *, check_current=lambda: None):
        return writer.build_and_write(
            self.slot,
            self.metadata,
            self.metadata.sequence.total_length - 1,
            self.lease,
            check_current=check_current,
            on_prepared=self.on_prepared,
        )

    def readback(self):
        client = BufferStore()
        client.data = self.data
        with_store = MooncakeSnapshotStore(client, FakeReplicateConfig())
        try:
            publication = self.state["publication"]
            manifest, packed = read_snapshot(with_store, publication)
            self.assertEqual(manifest.sequence, self.manifest.sequence)
            for descriptor in manifest.objects:
                value = packed[descriptor.name]
                if descriptor.kind == "kv":
                    start, end = descriptor.token_range
                    value = value[start:end]
                torch.testing.assert_close(value, self.tensors[descriptor.key])
        finally:
            with_store.close()

    def test_shared_payloads_cpu_only_publication_and_ack(self):
        writer = self.start()
        self.assertNotEqual(self.state["pid"], os.getpid())
        self.assertFalse(self.state["cuda_initialized"])
        with patch(
            "sglang.srt.training_capture.writer_process.prepare_snapshot",
            side_effect=AssertionError("construction ran in serving process"),
        ):
            self.write(writer)
        self.assertIsNotNone(self.prepared)
        self.readback()
        self.assertEqual(writer.timings.stats()["snapshot_build"]["calls"], 1)
        self.pool.release(self.slot, transfer_complete=True)
        with self.assertRaisesRegex(CaptureError, "writer stops"):
            self.pool.close()
        self.assertEqual(len(self.pool.slots), 1)
        self.assertIs(self.pool.acquire(), self.slot)
        writer.close()
        self.assertIsNotNone(writer.process.poll())
        PublicationJournal(self.directory.name).close()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_pinned_shared_pool_d2h_and_release(self):
        self.pool = HostBufferPool(
            kv=self.manifest.kv,
            max_tokens=8,
            slots=1,
            max_bytes=2 << 20,
            registrar=None,
            shared=True,
            pin_memory=True,
        )
        self.slot = self.pool.acquire()
        self.addCleanup(self.pool.unpin_shared)
        writer = self.start()
        self.assertTrue(self.slot.storage.is_pinned())
        source = self.slot.tensors["teacher_topk_logits"][:3].cuda()
        self.slot.tensors["teacher_topk_logits"][:3].fill_(float("nan"))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.slot.tensors["teacher_topk_logits"][:3].copy_(
                source, non_blocking=True
            )
        stream.synchronize()
        self.write(writer)
        self.readback()
        writer.close()
        self.pool.release(self.slot, transfer_complete=True)
        self.pool.close()
        self.assertFalse(self.slot.storage.is_pinned())

    def test_cancellation_after_remote_validation_prevents_writes(self):
        writer = self.start()

        def cancelled():
            raise ContractError("cancelled during validation")

        with self.assertRaisesRegex(ContractError, "cancelled"):
            self.write(writer, check_current=cancelled)
        self.assertFalse(self.data)
        self.assertIsNone(self.prepared)
        # Cancellation ACK drained the channel; the next command still works.
        self.write(writer)
        self.readback()

    def test_lease_creation_is_remote_and_serialized_with_publication(self):
        writer = self.start()
        identity = {"dataset_id": "ds", "sample_id": "sample", "generation_id": "gen"}
        checking, release = threading.Event(), threading.Event()

        def check():
            checking.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test did not release publication check")

        with ThreadPoolExecutor(max_workers=2) as threads:
            writing = threads.submit(self.write, writer, check_current=check)
            try:
                self.assertTrue(checking.wait(timeout=10))
                leasing = threads.submit(writer.begin, identity)
                time.sleep(0.02)
                self.assertFalse(leasing.done())
            finally:
                release.set()
            writing.result(timeout=30)
            lease = leasing.result(timeout=30)
        self.assertEqual(lease.sample_id, identity["sample_id"])
        self.assertEqual(self.state["begin_pid"], writer.process.pid)
        self.assertNotEqual(self.state["begin_pid"], os.getpid())
        self.readback()

    def test_failed_lease_creation_does_not_break_publication_channel(self):
        writer = self.start(failure="begin")
        with self.assertRaises(CatalogUnavailable):
            writer.begin({})
        self.assertFalse(writer.failed)
        self.assertFalse(writer.quarantined)
        self.write(writer)
        self.readback()

    def test_ambiguous_publish_keeps_exact_bytes_and_recovers(self):
        writer = self.start(failure="publish")
        with self.assertRaises(CatalogUnavailable):
            self.write(writer)
        self.assertIsNotNone(self.prepared)
        self.assertIsNone(self.pool.acquire())
        writer.recover_prepared(*self.prepared)
        self.readback()
        self.assertEqual(
            decode_manifest(self.prepared[1]).sequence, self.manifest.sequence
        )

    def test_transport_failure_quarantines_shared_slot(self):
        writer = self.start(failure="transport")
        with self.assertRaises(TransportError):
            self.write(writer)
        self.assertIn(self.slot.storage.data_ptr(), writer.quarantined)
        self.pool.release(self.slot, transfer_complete=False)
        self.assertIsNone(self.pool.acquire())
        self.assertNotIn("publication", self.state)

    def test_invalid_payload_is_rejected_before_store_write(self):
        writer = self.start()
        self.slot.tensors["teacher_topk_logits"][0, 0] = float("nan")
        with self.assertRaises(ContractError):
            self.write(writer)
        self.assertFalse(self.data)
        self.assertNotIn("publication", self.state)
        self.assertIsNone(self.prepared)

    def test_worker_death_fails_without_reusing_slot(self):
        writer = self.start()
        writer.process.kill()
        writer.process.wait(timeout=5)
        with self.assertRaises(WriterProcessError):
            self.write(writer)
        self.assertTrue(writer.failed)
        self.assertIn(self.slot.storage.data_ptr(), writer.quarantined)
        self.assertIsNone(self.pool.acquire())

    def test_timeout_stops_child_before_host_pins_can_be_released(self):
        writer = self.start()
        writer.timeout = 0.2
        with self.assertRaises(WriterProcessError):
            self.write(writer, check_current=lambda: time.sleep(0.3))
        self.assertTrue(writer.failed)
        self.assertIsNotNone(writer.process.poll())
        self.assertFalse(self.pool.shared_writer_active)
        self.assertIn(self.slot.storage.data_ptr(), writer.quarantined)


if __name__ == "__main__":
    unittest.main()
