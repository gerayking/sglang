"""Passive rank-local resources, with transport shutdown before buffer release."""

from __future__ import annotations

import os
from typing import ClassVar

import msgspec
import torch
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.kv_exporter import (
    HiCacheL2KVExporter,
    SelectedLayerKVExporter,
)
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import ContractError
from sglang.srt.training_capture.snapshot_writer import PublicationJournal
from sglang.srt.training_capture.teacher_batcher import TeacherLogitBatcher


class CaptureResources:
    # An unsuccessful CUDA or transport stop must retain storage even if its
    # exception is discarded. Process teardown is the final backstop.
    _retained: ClassVar[set[CaptureResources]] = set()

    def __init__(self):
        self.store = self.pool = self.catalog = self.journal = self.exporter = None
        self.l2_exporter = self.teacher_batcher = None
        self.writer_process = None
        self.device = None
        self.closed = False

    @classmethod
    def prepare(
        cls,
        *,
        config,
        kv,
        partition,
        source_pool,
        teacher=None,
        tree_cache=None,
        pin_memory=True,
    ):
        resources = cls()
        partition.local_layers(kv)
        if not partition.active:
            return resources
        try:
            token = (
                os.environ[config.catalog_token_env]
                if config.catalog_token_env
                else None
            )
            resources.catalog = HTTPCaptureCatalog(
                config.catalog_endpoint,
                bearer_token=token,
                timeout=config.http_timeout_seconds,
                attempts=config.http_attempts,
            )
            resources.catalog.check_compatibility(
                contract_id=config.contract_id,
                kv_codec=kv.codec,
                store_protocol=config.store.protocol,
            )
            if partition.heads and config.kv_source == "device":
                resources.exporter = SelectedLayerKVExporter.from_pool(
                    kv, source_pool, partition=partition
                )
            elif partition.heads:
                if tree_cache is None:
                    raise ContractError("HiCache L2 capture requires the tree cache")
                resources.l2_exporter = HiCacheL2KVExporter(
                    kv, tree_cache, source_pool, partition=partition
                )
            if config.snapshot_writer_backend == "thread":
                resources.store = MooncakeSnapshotStore.connect(
                    msgspec.to_builtins(config.store),
                    replica_num=config.replica_num,
                    max_receive_bytes=config.max_host_bytes,
                    payload_hash_workers=config.payload_hash_workers,
                )
            resources._allocate(
                config,
                kv,
                partition,
                pin_memory,
                device=getattr(source_pool, "device", None),
                teacher=teacher,
            )
        except Exception:
            resources.close()
            raise
        return resources

    @classmethod
    def from_connected(
        cls,
        *,
        config,
        kv,
        exporter,
        store,
        catalog,
        pin_memory=True,
        l2_exporter=None,
        teacher=None,
    ):
        """Take ownership of an already connected single-rank transport."""
        if config.snapshot_writer_backend == "process":
            raise ContractError(
                "process writer must connect its Store inside the child"
            )
        resources = cls()
        resources.store, resources.catalog, resources.exporter = (
            store,
            catalog,
            exporter,
        )
        resources.l2_exporter = l2_exporter
        try:
            if catalog is not None:
                catalog.check_compatibility(
                    contract_id=config.contract_id,
                    kv_codec=kv.codec,
                    store_protocol=config.store.protocol,
                )
            if store.payload_hasher.workers != config.payload_hash_workers:
                raise ContractError(
                    "connected Store hash workers differ from capture config"
                )
            resources._allocate(config, kv, None, pin_memory, teacher=teacher)
        except Exception:
            resources.close()
            raise
        return resources

    def _allocate(
        self, config, kv, partition, pin_memory, *, device=None, teacher=None
    ):
        device = self.exporter.device if self.exporter is not None else device
        if device is not None:
            self.device = torch.device(device)
            if self.device.type == "cuda" and self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        async_teacher = config.teacher_capture_backend == "async_ring" and (
            partition is None or partition.include_aux
        )
        self.pool = HostBufferPool(
            kv=kv,
            max_tokens=config.max_sample_tokens,
            slots=config.max_inflight_samples,
            max_bytes=config.max_host_bytes,
            registrar=self.store,
            manifest_bytes=config.manifest_buffer_bytes,
            pin_memory=pin_memory,
            device=self.device,
            kv_d2h_batch_tokens=config.kv_d2h_batch_tokens,
            kv_export_backend=config.kv_export_backend,
            teacher_d2h_batch_tokens=(
                config.max_sample_tokens
                if async_teacher
                else config.teacher_d2h_batch_tokens
            ),
            max_device_bytes=config.max_device_bytes,
            partition=partition,
            shared=config.snapshot_writer_backend == "process",
        )
        if config.kv_export_backend == "hicache" and self.exporter is not None:
            for slot in self.pool.slots:
                slot.kv_exporter = self.exporter.bind(slot)
            torch.cuda.current_stream(self.exporter.device).synchronize()
        if async_teacher:
            if teacher is None:
                raise ContractError("async teacher capture requires teacher identity")
            if self.device is None or self.device.type != "cuda" or not pin_memory:
                raise ContractError(
                    "async teacher capture requires CUDA and pinned Host buffers"
                )
            ring_bytes = (
                config.teacher_gpu_ring_rows
                * teacher.vocab_size
                * torch.float32.itemsize
            )
            required = self.pool.device_allocated_bytes + ring_bytes
            if required > config.max_device_bytes:
                raise ContractError(
                    f"async teacher buffers require {required} bytes, "
                    f"budget is {config.max_device_bytes}"
                )
            self.teacher_batcher = TeacherLogitBatcher(
                vocab_size=teacher.vocab_size,
                ring_rows=config.teacher_gpu_ring_rows,
                batch_rows=config.teacher_background_batch_rows,
                device=self.device,
                topk_backend=config.teacher_topk_backend,
            )
        if config.snapshot_writer_backend == "process":
            from sglang.srt.training_capture.writer_process import ProcessSnapshotWriter

            self.writer_process = ProcessSnapshotWriter(config, self.pool)
        elif partition is None or partition.include_aux:
            self.journal = PublicationJournal(config.journal_directory)

    def close(self):
        """Caller stops workers; both CUDA and Store must stop before release."""
        if self.closed:
            return
        try:
            if self.teacher_batcher is not None:
                self.teacher_batcher.close()
            # A failed request fence can leave copies pending on another stream.
            # Closing the Store client only stops its own transport operations.
            if self.device is not None and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if self.writer_process is not None:
                self.writer_process.close()
            if self.store is not None:
                self.store.close()
            if self.journal is not None:
                self.journal.close()
            if self.pool is not None:
                self.pool.unpin_shared()
        except Exception:
            self._retained.add(self)
            raise
        self.closed = True
        self._retained.discard(self)
