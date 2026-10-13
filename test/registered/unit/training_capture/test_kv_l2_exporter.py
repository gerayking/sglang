"""HiCache L2-backed training snapshot export."""

import unittest
from types import SimpleNamespace

import msgspec
import torch
from sglang.srt.training_capture.kv_exporter import HiCacheL2KVExporter
from sglang.srt.training_capture.protocol import ContractError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import make_kv_spec

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Event:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class _Node:
    def __init__(self, parent, host_value=None, pending_id=None):
        self.parent = parent
        self.host_value = host_value
        self.write_through_pending_id = pending_id
        self.host_ref_counter = 0

    def protect_host(self):
        self.host_ref_counter += 1

    def release_host(self):
        self.host_ref_counter -= 1


class _Tree:
    page_size = 1
    is_eagle = False

    def __init__(self, host_pool, leaf, acks):
        self.root_node = leaf.parent.parent
        self.leaf = leaf
        self.cache_controller = SimpleNamespace(
            write_policy="write_through",
            mem_pool_host=host_pool,
            ack_write_queue=acks,
        )

    def match_prefix(self, _params):
        return SimpleNamespace(best_match_node=self.leaf)


class _UnifiedNode:
    def __init__(self, node_id, parent, host_value=None, pending_id=None):
        from sglang.srt.mem_cache.unified_cache.components import ComponentType

        self.id = node_id
        self.parent = parent
        self.component_data = {
            ComponentType.FULL: SimpleNamespace(host_value=host_value)
        }
        self.write_through_pending_id = pending_id
        self.host_ref_counter = 0


class _UnifiedTree:
    page_size = 1
    is_eagle = False

    def __init__(self, host_pool, nodes, acks):
        self.nodes = {node.id: node for node in nodes}
        self.root = nodes[0]
        self.leaf = nodes[-1]
        self.tree_core = SimpleNamespace(
            root_node=self.root,
            is_eagle=False,
            node_by_id=lambda node_id: self.nodes[node_id],
        )
        self.cache_controller = SimpleNamespace(
            write_policy="write_through",
            mem_pool_host=host_pool,
            ack_write_queue=acks,
        )

    def match_prefix(self, params):
        self.matched_key = params.key
        return SimpleNamespace(
            best_match_node=self.leaf.id,
            full_kv_hit_length=len(params.key),
        )

    def inc_host_lock_ref(self, node_id):
        node = self.nodes[node_id]
        while node is not self.root:
            node.host_ref_counter += 1
            node = node.parent
        return SimpleNamespace(to_dec_params=lambda: None)

    def dec_host_lock_ref(self, node_id, _params):
        node = self.nodes[node_id]
        while node is not self.root:
            node.host_ref_counter -= 1
            node = node.parent


def _fixture(*, missing_event=False):
    kv = msgspec.structs.replace(make_kv_spec(), source_page_size=1)
    host_pool = SimpleNamespace(
        layout="layer_first",
        layer_num=4,
        k_buffer=torch.arange(4 * 8 * 2 * 4, dtype=torch.bfloat16).reshape(4, 8, 2, 4),
        v_buffer=(
            torch.arange(4 * 8 * 2 * 4, dtype=torch.bfloat16).reshape(4, 8, 2, 4) + 1024
        ),
    )
    root = _Node(None)
    first = _Node(root, torch.tensor([2, 7]), pending_id=41)
    leaf = _Node(first, torch.tensor([5]), pending_id=None)
    event = _Event()
    acks = [] if missing_event else [SimpleNamespace(node_ids=[41], finish_event=event)]
    tree = _Tree(host_pool, leaf, acks)
    exporter = HiCacheL2KVExporter(
        kv, tree, SimpleNamespace(start_layer=0), partition=None
    )
    request = SimpleNamespace(
        origin_input_ids=[11, 12], output_ids=[13], extra_key=None
    )
    return kv, host_pool, (first, leaf), event, exporter, request


def _unified_fixture():
    kv, host_pool, _, event, _, request = _fixture()
    root = _UnifiedNode(0, None)
    first = _UnifiedNode(1, root, torch.tensor([2, 7]), pending_id=41)
    leaf = _UnifiedNode(2, first, torch.tensor([5]))
    tree = _UnifiedTree(
        host_pool,
        (root, first, leaf),
        [SimpleNamespace(node_ids=[41], finish_event=event)],
    )
    exporter = HiCacheL2KVExporter(
        kv, tree, SimpleNamespace(start_layer=0), partition=None
    )
    request.last_node = root.id
    return tree, (first, leaf), exporter, request


class TestHiCacheL2KVExporter(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_transient_copy_waits_for_overlapped_forward(self):
        _, _, _, _, exporter, request = _fixture()
        request.req_pool_idx = 0
        exporter.tree_cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(3, device="cuda")[None]
        )
        exporter.host_pool.available_size = lambda: 8
        source = torch.zeros(3, device="cuda")
        destination = torch.empty(3, pin_memory=True)
        producer = torch.cuda.Stream()
        torch.cuda.synchronize()

        def copy_to_l2(indices):
            destination.copy_(source, non_blocking=True)
            finished = torch.cuda.Event()
            finished.record()
            return torch.arange(3), finished

        exporter.controller.write_transient = copy_to_l2
        with torch.cuda.stream(producer):
            torch.cuda._sleep(50_000_000)
            source.fill_(7)
        lease = exporter.acquire_transient(
            request, valid_tokens=3, source_stream=producer
        )
        lease.events[0].synchronize()
        torch.testing.assert_close(destination, torch.full((3,), 7.0))

    def test_waits_for_l2_and_materializes_selected_layers(self):
        kv, host_pool, nodes, event, exporter, request = _fixture()
        lease = exporter.acquire(request, valid_tokens=3)
        self.assertEqual([node.host_ref_counter for node in nodes], [1, 1])

        destinations = {
            f"target_{kind}.{layer.layer_id}": torch.empty(
                3, layer.num_kv_heads, dim, dtype=torch.bfloat16
            )
            for layer in kv.layers
            for kind, dim in (
                ("k", layer.key_head_dim),
                ("v", layer.value_head_dim),
            )
        }
        exporter.materialize(lease, destinations, end=3)

        self.assertTrue(event.synchronized)
        indices = torch.tensor([2, 7, 5])
        for layer in kv.layers:
            for kind in ("k", "v"):
                expected = getattr(host_pool, f"{kind}_buffer")[
                    layer.layer_id
                ].index_select(0, indices)
                torch.testing.assert_close(
                    destinations[f"target_{kind}.{layer.layer_id}"],
                    expected,
                    rtol=0,
                    atol=0,
                )

        # The writer can finish on a background thread. Tree refcounts are
        # released later by the scheduler thread.
        self.assertEqual([node.host_ref_counter for node in nodes], [1, 1])
        self.assertEqual(exporter.drain_releases(), 1)
        self.assertEqual([node.host_ref_counter for node in nodes], [0, 0])
        self.assertEqual(exporter.drain_releases(), 0)

    def test_missing_write_completion_event_releases_l2_pages(self):
        _, _, nodes, _, exporter, request = _fixture(missing_event=True)
        with self.assertRaisesRegex(ContractError, "completion event"):
            exporter.acquire(request, valid_tokens=3)
        self.assertEqual([node.host_ref_counter for node in nodes], [0, 0])

    def test_unified_cache_resolves_committed_prefix_instead_of_request_cursor(self):
        tree, nodes, exporter, request = _unified_fixture()

        lease = exporter.acquire(request, valid_tokens=3)

        self.assertEqual(list(tree.matched_key.token_ids), [11, 12, 13])
        self.assertEqual([node.host_ref_counter for node in nodes], [1, 1])
        lease.release()
        self.assertEqual(exporter.drain_releases(), 1)
        self.assertEqual([node.host_ref_counter for node in nodes], [0, 0])


if __name__ == "__main__":
    unittest.main()
