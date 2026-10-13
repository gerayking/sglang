"""Gather selected target KV into request-owned Host or device storage."""

from __future__ import annotations

import queue
import threading
from collections import defaultdict
from collections.abc import Mapping

import torch
from sglang.srt.training_capture.protocol import DTYPES, ContractError, KVSpec
from sglang.srt.training_capture.topology import CapturePartition


class HiCacheL2SnapshotLease:
    """Pinned HiCache Host pages owned by one background materialization."""

    def __init__(self, exporter, host_indices, events, release_token):
        self.exporter = exporter
        self.host_indices = host_indices
        self.events = tuple(events)
        self.release_token = release_token
        self.lock = threading.Lock()
        self.released = False

    def release(self):
        with self.lock:
            if self.released:
                return
            self.released = True
        self.exporter.defer_release(self.release_token)


class HiCacheL2KVExporter:
    """Materialize selected KV from serving HiCache L2, never from device KV."""

    def __init__(
        self,
        kv: KVSpec,
        tree_cache,
        source_pool,
        *,
        partition: CapturePartition | None = None,
    ):
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        controller = getattr(tree_cache, "cache_controller", None)
        if controller is None or controller.write_policy != "write_through":
            raise ContractError("L2 capture requires a write-through HiCache")
        if tree_cache.page_size != 1:
            raise ContractError("L2 capture currently requires a one-token page")
        host_pool = controller.mem_pool_host
        if hasattr(host_pool, "entry_map"):
            entry = host_pool.entry_map.get(PoolName.KV)
            if entry is None:
                raise ContractError("HiCache has no full-KV Host pool")
            host_pool = entry.host_pool
        if not all(hasattr(host_pool, name) for name in ("k_buffer", "v_buffer")):
            raise ContractError("L2 capture requires an MHA/GQA Host KV pool")

        self.kv = kv
        self.tree_cache = tree_cache
        self.controller = controller
        self.host_pool = host_pool
        self.partition = partition
        self.layers = kv.layers if partition is None else partition.local_layers(kv)
        self.release_queue = queue.SimpleQueue()
        self.requires_transient = bool(
            getattr(tree_cache, "supports_mamba", lambda: False)()
        )
        if isinstance(source_pool, HybridLinearKVPool):
            mapping = source_pool.full_attention_layer_id_mapping
            self.layer_indices = {
                layer.layer_id: mapping[layer.layer_id] for layer in self.layers
            }
        else:
            start = getattr(source_pool, "start_layer", 0)
            self.layer_indices = {
                layer.layer_id: layer.layer_id - start for layer in self.layers
            }
        for layer in self.layers:
            local = self.layer_indices[layer.layer_id]
            if not 0 <= local < host_pool.layer_num:
                raise ContractError("selected layer is not present in HiCache L2")
            k = self._layer_rows("k", local)
            v = self._layer_rows("v", local)
            if (
                list(k.shape[1:]) != [layer.num_kv_heads, layer.key_head_dim]
                or list(v.shape[1:]) != [layer.num_kv_heads, layer.value_head_dim]
                or k.dtype != DTYPES[kv.dtype]
                or v.dtype != DTYPES[kv.dtype]
            ):
                raise ContractError(
                    "HiCache L2 layout differs from the capture contract"
                )

    def _layer_rows(self, component, layer_index):
        value = getattr(self.host_pool, f"{component}_buffer")
        layout = self.host_pool.layout
        if layout == "layer_first":
            return value[layer_index]
        if layout == "page_first":
            return value[:, layer_index]
        if layout == "page_first_direct":
            return value[:, layer_index, 0]
        if layout == "page_head":
            return value[:, :, 0, layer_index]
        raise ContractError(f"unsupported HiCache L2 layout: {layout}")

    @staticmethod
    def _legacy_path(node, root):
        path = []
        while node is not root:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def _pending_events(self, nodes):
        pending = {
            node.write_through_pending_id
            for node in nodes
            if node.write_through_pending_id is not None
        }
        events = []
        for ack in tuple(self.controller.ack_write_queue):
            if pending.intersection(ack.node_ids):
                events.append(ack.finish_event)
                pending.difference_update(ack.node_ids)
        if pending:
            raise ContractError("HiCache L2 write completion event is unavailable")
        return events

    def _find_unified_node(self, req, valid_tokens):
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        key = RadixKey(
            (req.origin_input_ids + req.output_ids)[:valid_tokens],
            req.extra_key,
            is_bigram=self.tree_cache.tree_core.is_eagle,
        )
        result = self.tree_cache.match_prefix(MatchPrefixParams(key=key))
        if result.full_kv_hit_length != valid_tokens:
            raise ContractError(
                "HiCache L2 prefix length differs from committed KV coverage"
            )
        return result.best_match_node

    def _acquire_unified(self, req, valid_tokens):
        from sglang.srt.mem_cache.unified_cache.components import ComponentType

        node_id = self._find_unified_node(req, valid_tokens)
        params = self.tree_cache.inc_host_lock_ref(node_id).to_dec_params()
        token = ("unified", node_id, params)
        try:
            node = self.tree_cache.tree_core.node_by_id(node_id)
            nodes = self._legacy_path(node, self.tree_cache.tree_core.root_node)
            values = [
                node.component_data[ComponentType.FULL].host_value for node in nodes
            ]
            if any(value is None for value in values):
                raise ContractError("committed prefix is not resident in HiCache L2")
            return nodes, values, token
        except Exception:
            self._release(token)
            raise

    def _acquire_legacy(self, req, valid_tokens):
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        key = RadixKey(
            (req.origin_input_ids + req.output_ids)[:valid_tokens],
            req.extra_key,
            is_bigram=self.tree_cache.is_eagle,
        )
        result = self.tree_cache.match_prefix(MatchPrefixParams(key=key))
        node = result.best_match_node
        nodes = self._legacy_path(node, self.tree_cache.root_node)
        values = [node.host_value for node in nodes]
        if any(value is None for value in values):
            raise ContractError("committed prefix is not resident in HiCache L2")
        for node in nodes:
            node.protect_host()
        return nodes, values, ("legacy", tuple(nodes), None)

    def acquire(self, req, *, valid_tokens):
        if valid_tokens < 1:
            raise ContractError("L2 snapshot requires a nonempty KV prefix")
        if hasattr(self.tree_cache, "tree_core"):
            nodes, values, token = self._acquire_unified(req, valid_tokens)
        else:
            nodes, values, token = self._acquire_legacy(req, valid_tokens)
        try:
            host_indices = torch.cat(values).to(dtype=torch.int64, copy=True)
            if host_indices.numel() != valid_tokens:
                raise ContractError(
                    "HiCache L2 prefix length differs from committed KV coverage"
                )
            events = self._pending_events(nodes)
            return HiCacheL2SnapshotLease(self, host_indices, events, token)
        except Exception:
            self._release(token)
            raise

    def acquire_transient(self, req, *, valid_tokens, source_stream=None):
        """Stage a hybrid request in HiCache L2 before device slots are freed."""
        if valid_tokens < 1:
            raise ContractError("L2 snapshot requires a nonempty KV prefix")
        available = self.controller.mem_pool_host.available_size()
        if available < valid_tokens:
            needed = valid_tokens - available
            if self.tree_cache.evict_host(needed) < needed:
                raise ContractError("HiCache L2 has no capacity for a snapshot lease")
        device_indices = self.tree_cache.req_to_token_pool.req_to_token[
            req.req_pool_idx, :valid_tokens
        ]
        if source_stream is not None:
            # Under overlap, request release runs on the scheduler stream while
            # the last KV writes can still be pending on the forward stream.
            torch.cuda.current_stream(device_indices.device).wait_stream(source_stream)
        result = self.controller.write_transient(device_indices)
        if result is None:
            raise ContractError("HiCache L2 snapshot allocation failed")
        host_indices, finish_event = result
        if device_indices.is_cuda:
            torch.cuda.current_stream(device_indices.device).wait_event(finish_event)
        return HiCacheL2SnapshotLease(
            self,
            host_indices.to(dtype=torch.int64, copy=True),
            (finish_event,),
            ("transient", host_indices, None),
        )

    @torch.no_grad()
    def materialize(self, lease, destinations, *, end):
        try:
            for event in lease.events:
                event.synchronize()
            indices = lease.host_indices[:end]
            if indices.numel() != end:
                raise ContractError("L2 lease does not cover the requested KV prefix")
            for layer in self.layers:
                local = self.layer_indices[layer.layer_id]
                for component in ("k", "v"):
                    name = f"target_{component}.{layer.layer_id}"
                    source = self._layer_rows(component, local)
                    destination = destinations[name][:end]
                    if (
                        destination.device.type != "cpu"
                        or not destination.is_contiguous()
                    ):
                        raise ContractError(
                            "L2 materialization requires contiguous Host destinations"
                        )
                    torch.index_select(source, 0, indices, out=destination)
        finally:
            lease.release()

    def defer_release(self, token):
        self.release_queue.put(token)

    def _release(self, token):
        kind, owner, params = token
        if kind == "unified":
            self.tree_cache.dec_host_lock_ref(owner, params)
        elif kind == "transient":
            self.controller.mem_pool_host.free(owner)
        else:
            for node in owner:
                node.release_host()

    def drain_releases(self):
        count = 0
        while True:
            try:
                token = self.release_queue.get_nowait()
            except queue.Empty:
                return count
            self._release(token)
            count += 1


class SelectedLayerKVExporter:
    def __init__(
        self,
        kv: KVSpec,
        buffers: Mapping[str, torch.Tensor],
        *,
        partition: CapturePartition | None = None,
    ):
        self.kv = kv
        self.buffers = dict(buffers)
        if not self.buffers:
            raise ContractError("KV exporter requires selected source buffers")
        self.device = next(iter(self.buffers.values())).device
        expected = set()
        layers = kv.layers if partition is None else partition.local_layers(kv)
        for layer in layers:
            for component, dim in (
                ("k", layer.key_head_dim),
                ("v", layer.value_head_dim),
            ):
                name = f"target_{component}.{layer.layer_id}"
                expected.add(name)
                value = self.buffers[name]
                if value.device != self.device:
                    raise ContractError("selected KV buffers must share a device")
                if (
                    value.ndim != 3
                    or list(value.shape[1:]) != [layer.num_kv_heads, dim]
                    or value.dtype != DTYPES[kv.dtype]
                ):
                    raise ContractError(
                        "KV exporter requires unquantized token/head/dim source buffers"
                    )
        if set(self.buffers) != expected:
            raise ContractError("KV source layers differ from the capture contract")

    @classmethod
    def from_pool(cls, kv: KVSpec, pool, *, partition: CapturePartition | None = None):
        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            MHATokenToKVPool,
        )

        is_hybrid = isinstance(pool, HybridLinearKVPool)
        source_pool = pool.full_kv_pool if is_hybrid else pool
        if (
            not isinstance(source_pool, MHATokenToKVPool)
            or source_pool.is_quantized_kv_cache
            or source_pool.use_hnd
            or (is_hybrid and pool.use_mla)
        ):
            raise ContractError(
                "capture requires a dense, unquantized NHD MHA/GQA pool"
            )
        if pool.page_size != kv.source_page_size:
            raise ContractError("configured source page size disagrees with the pool")
        buffers = {}
        layers = kv.layers if partition is None else partition.local_layers(kv)
        for layer in layers:
            if is_hybrid:
                local = layer.layer_id in pool.full_attention_layer_id_mapping
            else:
                local = 0 <= layer.layer_id - pool.start_layer < pool.layer_num
            if not local:
                raise ContractError("selected layer is not local to this worker")
            buffers[f"target_k.{layer.layer_id}"] = pool.get_key_buffer(layer.layer_id)
            buffers[f"target_v.{layer.layer_id}"] = pool.get_value_buffer(
                layer.layer_id
            )
        return cls(kv, buffers, partition=partition)

    def bind(self, slot):
        """Bind immutable, pool-owned destinations before capture admission."""
        return BoundHiCacheKVExporter(self, slot)

    @torch.no_grad()
    def export(
        self,
        slots: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        start: int,
        end: int,
    ) -> None:
        """Enqueue on the producer stream, before any reuse of the source slots.

        Gathered temporaries own storage. Same-stream ordering protects the
        pool read, and record_stream protects temporary storage through D2H.
        The caller records completion and retains Host storage until it fires.
        """
        if (
            slots.ndim != 1
            or slots.dtype not in (torch.int32, torch.int64)
            or slots.numel() != end - start
            or not 0 <= start < end
        ):
            raise ContractError("invalid KV export position mapping")
        # index_select accepts the request pool's int32 indices without an upcast.
        indices = slots.to(device=self.device)
        stream = torch.cuda.current_stream(self.device) if indices.is_cuda else None
        if stream is not None:
            indices.record_stream(stream)
        for name, source in self.buffers.items():
            destination = destinations[name][start:end]
            if (
                destination.shape != (end - start, *source.shape[1:])
                or destination.dtype != source.dtype
            ):
                raise ContractError("destination does not cover the KV range")
            if destination.is_cuda:
                if destination.device != source.device:
                    raise ContractError("KV staging must share the source device")
                torch.index_select(source, 0, indices, out=destination)
                destination.record_stream(stream)
                continue
            if destination.device.type != "cpu":
                raise ContractError("KV export requires Host or CUDA storage")
            if source.is_cuda and not destination.is_pinned():
                raise ContractError(
                    "asynchronous KV export requires pinned Host storage"
                )
            gathered = source.index_select(0, indices)
            destination.copy_(gathered, non_blocking=source.is_cuda)
            if stream is not None:
                gathered.record_stream(stream)


class BoundHiCacheKVExporter:
    """Use existing HiCache kernels with metadata owned by the registered slot."""

    def __init__(self, exporter, slot):
        from sglang.kernels.ops.kvcache.hicache import can_use_hicache_jit_kernel

        self.exporter = exporter
        self.metadata = slot.kv_export_tensors
        self.destinations = (slot.tensors, slot.device_tensors)
        self.host_storage, self.device_storage = slot.storage, slot.device_storage
        self.sources = tuple(exporter.buffers.values())
        self.host_enqueued_bytes = self.device_enqueued_bytes = 0
        if self.metadata is None or exporter.device.type != "cuda":
            raise ContractError(
                "HiCache export metadata must be allocated by the Host pool"
            )
        groups = defaultdict(list)
        for name, source in exporter.buffers.items():
            row_bytes = source.shape[1] * source.shape[2] * source.element_size()
            if (
                source.stride(2) != 1
                or source.stride(1) != source.shape[2]
                or row_bytes % 128
                or source.data_ptr() % 16
                or source.stride(0) * source.element_size() % 16
            ):
                raise ContractError(
                    "HiCache export requires aligned, contiguous KV rows of a multiple of 128 bytes"
                )
            for tensors in self.destinations:
                if tensors is None:
                    continue
                destination = tensors[name]
                if (
                    destination.shape[1:] != source.shape[1:]
                    or destination.dtype != source.dtype
                    or not destination.is_contiguous()
                    or destination.data_ptr() % 16
                    or (
                        destination.device != source.device
                        and not (
                            destination.device.type == "cpu" and destination.is_pinned()
                        )
                    )
                ):
                    raise ContractError(
                        "HiCache export destination differs from the bound KV layout"
                    )
            groups[row_bytes, source.stride(0) * source.element_size()].append(name)
        self.groups = []
        names = []
        for (row_bytes, source_stride), group_names in groups.items():
            if not can_use_hicache_jit_kernel(element_size=row_bytes):
                raise ContractError("HiCache KV export kernel could not be initialized")
            self.groups.append(
                (len(names), len(names) + len(group_names), row_bytes, source_stride)
            )
            names.extend(group_names)
        self.source_rows = min(value.shape[0] for value in exporter.buffers.values())
        self.row_bytes = sum(
            (last - first) * width for first, last, width, _ in self.groups
        )
        self.metadata["export_sources"].copy_(
            torch.tensor(
                [exporter.buffers[name].data_ptr() for name in names],
                dtype=torch.uint64,
            )
        )
        for key, tensors in zip(("export_host", "export_staging"), self.destinations):
            if tensors is not None:
                self.metadata[key].copy_(
                    torch.tensor(
                        [tensors[name].data_ptr() for name in names], dtype=torch.uint64
                    )
                )
        for bits in (32, 64):
            positions = self.metadata[f"export_positions{bits}"]
            torch.arange(positions.numel(), out=positions)

    @torch.no_grad()
    def export(self, slots, destinations, start, end):
        from sglang.kernels.ops.kvcache.hicache import transfer_hicache_all_layer_mla

        if (
            slots.ndim != 1
            or slots.dtype not in (torch.int32, torch.int64)
            or slots.numel() != end - start
            or not 0 <= start < end
        ):
            raise ContractError("invalid KV export position mapping")
        if destinations is self.destinations[0]:
            pointer_key = "export_host"
        elif destinations is self.destinations[1] and destinations is not None:
            pointer_key = "export_staging"
        else:
            raise ContractError("KV destinations were not bound to this capture slot")
        if any(end > destinations[name].shape[0] for name in self.exporter.buffers):
            raise ContractError("destination does not cover the KV range")
        indices = slots.to(device=self.exporter.device).contiguous()
        # The pointer kernel has no pool extent parameter. Keep asynchronous
        # bounds rejection equivalent to index_select before it accesses storage.
        torch._assert_async(
            ((indices >= 0) & (indices < self.source_rows)).all(),
            "capture KV source index out of range",
        )
        stream = torch.cuda.current_stream(self.exporter.device)
        indices.record_stream(stream)
        bits = 32 if indices.dtype == torch.int32 else 64
        positions = self.metadata[f"export_positions{bits}"][start:end]
        for first, last, row_bytes, source_stride in self.groups:
            transfer_hicache_all_layer_mla(
                ptr_dst=self.metadata[pointer_key][first:last],
                indices_dst=positions,
                ptr_src=self.metadata["export_sources"][first:last],
                indices_src=indices,
                cache_src_stride_bytes=source_stride,
                cache_dst_stride_bytes=row_bytes,
                element_size=row_bytes,
            )
        self.device_storage.record_stream(stream)
        if pointer_key == "export_host":
            self.host_enqueued_bytes += (end - start) * self.row_bytes
        else:
            self.device_enqueued_bytes += (end - start) * self.row_bytes
