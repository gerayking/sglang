"""Teacher variants coexist without patches and preserve owned output lifetimes."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
from sglang.srt.training_capture import protocol
from sglang.srt.training_capture.codec import HIDDEN_CODEC, TOP128_CODEC
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.snapshot_writer import PublicationJournal
from sglang.srt.training_capture.teacher_batcher import FinalHiddenCapture
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.training_capture_utils import Registrar, make_snapshot

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTeacherCodec(unittest.TestCase):
    def setUp(self):
        self.template, self.top128_tensors = make_snapshot()

    def hidden_snapshot(self, device="cpu", *, graph=False):
        device = torch.device(device)
        pool = HostBufferPool(
            kv=self.template.kv,
            max_tokens=8,
            slots=1,
            max_bytes=2 << 20,
            registrar=Registrar(),
            pin_memory=device.type == "cuda",
            device=device,
            teacher_d2h_batch_tokens=8,
            max_device_bytes=1 << 20,
            codec=HIDDEN_CODEC,
        )
        slot = pool.acquire()
        batcher = FinalHiddenCapture(device=device, spec=HIDDEN_CODEC.teacher)
        context = RequestCaptureContext(
            slot=slot,
            prompt_ids=(3, 4),
            max_tokens=8,
            vocab_size=256,
            teacher_batcher=batcher,
            codec=HIDDEN_CODEC,
        )
        expected = []
        try:
            for layer in self.template.kv.layers:
                for component in ("k", "v"):
                    slot.tensors[f"target_{component}.{layer.layer_id}"].fill_(0.5)
            slot.tensors["token_ids"].fill_(-1)
            source = torch.zeros(2, 2560, dtype=torch.bfloat16, device=device)
            output = source.clone()
            cuda_graph = None
            if graph:
                cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph):
                    output.copy_(source)
            for position in range(2, 5):
                context.defer_kv_progress(end=position)
                source.fill_(position)
                if cuda_graph is not None:
                    cuda_graph.replay()
                else:
                    output.copy_(source)
                expected.append(torch.full((2560,), position, dtype=torch.bfloat16))
                batcher.append(output, [1], [(context, position)])
                # The next forward is allowed to overwrite its own output immediately.
                output.fill_(-1)
                context.commit_token(position=position, token_id=position + 10)
            context.seal("length")
            self.assertFalse(slot.host_ready)
            self.assertEqual(slot.tensors["token_ids"][0], -1)
            frozen_tokens = context.sealed.token_ids
            context.token_ids[-1] = 99
            manifest, tensors = context.snapshot(
                dataset_id=self.template.dataset_id,
                sample_id=self.template.sample_id,
                generation_id=self.template.generation_id,
                teacher=self.template.teacher,
                kv=self.template.kv,
                provenance=self.template.provenance,
                contract_id=HIDDEN_CODEC.contract_id,
            )
            self.assertTrue(slot.host_ready)
            self.assertEqual(
                tuple(slot.tensors["token_ids"][:5].tolist()), frozen_tokens
            )
            self.assertFalse(set(TOP128_CODEC.teacher.tensor_names) & set(slot.tensors))
            hidden = tensors[manifest.key_prefix + "aux/teacher_final_hidden"]
            torch.testing.assert_close(hidden, torch.stack(expected), rtol=0, atol=0)
            self.assertEqual(slot.tensors["kv_valid"][:5].tolist(), [1, 1, 1, 1, 0])
            return manifest, {key: value.clone() for key, value in tensors.items()}
        finally:
            batcher.close()
            if device.type == "cuda":
                torch.cuda.synchronize()
            pool.release(slot, transfer_complete=True)
            pool.close()

    def test_codecs_coexist_and_reject_the_other_wire_contract(self):
        original_decoder = protocol.decode_manifest
        hidden, tensors = self.hidden_snapshot()
        data = protocol.canonical_bytes(hidden)
        HIDDEN_CODEC.validate_tensors(HIDDEN_CODEC.decode_manifest(data), tensors)
        TOP128_CODEC.validate_tensors(self.template, self.top128_tensors)
        self.assertIs(protocol.decode_manifest, original_decoder)
        with self.assertRaises(protocol.ContractError):
            protocol.decode_manifest(data)
        with self.assertRaises(protocol.ContractError):
            HIDDEN_CODEC.decode_manifest(protocol.canonical_bytes(self.template))
        with self.assertRaises(protocol.ContractError):
            TOP128_CODEC.validate_manifest(hidden)

    def test_hidden_checksums_and_alignment_are_mandatory(self):
        manifest, tensors = self.hidden_snapshot()
        key = manifest.key_prefix + "aux/teacher_final_hidden"
        tensors[key][0, 0] += 1
        with self.assertRaisesRegex(protocol.ContractError, "checksum"):
            HIDDEN_CODEC.validate_tensors(manifest, tensors)
        manifest, tensors = self.hidden_snapshot()
        key = manifest.key_prefix + "aux/logits_positions"
        tensors[key][0] += 1
        objects = [
            msgspec.structs.replace(
                obj, sha256=protocol.digest_bytes(protocol.tensor_bytes(tensors[key]))
            )
            if obj.key == key
            else obj
            for obj in manifest.objects
        ]
        with self.assertRaisesRegex(protocol.ContractError, "accepted response"):
            HIDDEN_CODEC.validate_tensors(
                msgspec.structs.replace(manifest, objects=objects), tensors
            )

    def test_hidden_journal_uses_its_codec_without_global_install(self):
        from sglang.srt.training_capture.catalog import CaptureLease

        manifest, _ = self.hidden_snapshot()
        lease = CaptureLease(
            capture_id="c1",
            fencing_token=1,
            dataset_id=manifest.dataset_id,
            sample_id=manifest.sample_id,
            generation_id=manifest.generation_id,
            expires_in_seconds=120,
            renew_after_seconds=20,
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = PublicationJournal(directory, codec=HIDDEN_CODEC)
            data = protocol.canonical_bytes(manifest)
            try:
                journal.save(lease, data)
                self.assertEqual(list(journal.pending()), [(lease, data)])
            finally:
                journal.close()
            journal = PublicationJournal(directory)
            try:
                with self.assertRaises(protocol.ContractError):
                    list(journal.pending())
            finally:
                journal.close()

    def test_hidden_backend_requires_matching_contract_and_resources(self):
        config = CaptureConfig(
            dataset_id="d",
            model_id="test",
            producer_revision="test",
            selected_layer_ids=[0],
            catalog_endpoint="http://localhost:1",
            journal_directory="/tmp/hidden-journal",
            store=StoreSetup(
                local_hostname="localhost", master_server_addr="localhost:1"
            ),
            contract_id=HIDDEN_CODEC.contract_id,
            teacher_capture_backend="final_hidden",
            kv_source="hicache_l2",
            snapshot_writer_backend="process",
            max_device_bytes=1 << 20,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(msgspec.json.encode(config))
            self.assertEqual(CaptureConfig.load(path), config)
            for changes in (
                {"contract_id": TOP128_CODEC.contract_id},
                {"teacher_capture_backend": "async_ring"},
                {"kv_source": "device"},
                {"max_device_bytes": 0},
            ):
                with self.subTest(changes=changes):
                    path.write_bytes(
                        msgspec.json.encode(msgspec.structs.replace(config, **changes))
                    )
                    with self.assertRaises(protocol.ContractError):
                        CaptureConfig.load(path)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_hidden_graph_output_is_owned_before_replay_and_d2h(self):
        manifest, tensors = self.hidden_snapshot("cuda", graph=True)
        HIDDEN_CODEC.validate_tensors(manifest, tensors)

    def test_hidden_dtype_is_rejected_before_binding_or_allocation(self):
        from sglang.srt.training_capture.coordinator import CaptureCoordinator

        with (
            patch(
                "sglang.srt.training_capture.coordinator.CaptureConfig.load",
                return_value=SimpleNamespace(contract_id=HIDDEN_CODEC.contract_id),
            ),
            patch(
                "sglang.srt.training_capture.coordinator.bind_rank_target_contract"
            ) as bind,
            self.assertRaisesRegex(protocol.ContractError, "requires BF16"),
        ):
            try:
                CaptureCoordinator.create(
                    config_path="capture.json",
                    model=None,
                    model_config=SimpleNamespace(dtype=torch.float16),
                    tokenizer_path=None,
                    pool=None,
                    req_to_token=None,
                )
            finally:
                bind.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_aborted_hidden_waits_for_producer_before_recycling(self):
        pool = HostBufferPool(
            kv=self.template.kv,
            max_tokens=4,
            slots=1,
            max_bytes=2 << 20,
            registrar=Registrar(),
            device=torch.device("cuda"),
            teacher_d2h_batch_tokens=4,
            max_device_bytes=1 << 20,
            codec=HIDDEN_CODEC,
        )
        slot = pool.acquire()
        batcher = FinalHiddenCapture(device="cuda", spec=HIDDEN_CODEC.teacher)
        context = RequestCaptureContext(
            slot=slot,
            prompt_ids=(3,),
            max_tokens=4,
            vocab_size=256,
            teacher_batcher=batcher,
            codec=HIDDEN_CODEC,
        )
        try:
            context.defer_kv_progress(end=1)
            hidden = torch.ones(1, 2560, dtype=torch.bfloat16, device="cuda")
            stream, done = torch.cuda.Stream(), torch.cuda.Event()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda._sleep(100_000_000)
                batcher.append(hidden, [0], [(context, 1)])
                done.record()
            context.abort("cancelled")
            context.wait_for_copies()
            self.assertTrue(done.query())
            self.assertFalse(slot.host_ready)
        finally:
            batcher.close()
            torch.cuda.synchronize()
            pool.release(slot, transfer_complete=True)
            pool.close()


if __name__ == "__main__":
    unittest.main()
