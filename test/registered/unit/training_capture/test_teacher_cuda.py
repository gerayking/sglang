"""Raw teacher values and full-vocabulary normalization on CUDA."""

import math
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.protocol import ContractError
from sglang.srt.training_capture.teacher import capture_teacher, warmup_teacher_capture
from sglang.srt.training_capture.teacher_batcher import (
    TeacherLogitBatcher,
    TeacherRingFull,
)
from sglang.srt.training_capture.teacher_staging import TeacherCompletion
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import Registrar, make_kv_spec

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTeacherCuda(CustomTestCase):
    def test_async_ring_quarantines_failed_completion(self):
        done = threading.Event()

        class Completion(TeacherCompletion):
            def finish(self, error):
                super().finish(error)
                done.set()

        class Destination:
            teacher_staging = SimpleNamespace(transfer_uncertain=False)
            teacher_completion = Completion()

            def validate_async_teacher(self, *, position):
                pass

            def reserve_async_teacher(self, *, position):
                self.teacher_completion.reserve()
                return 0

        target = Destination()
        batcher = TeacherLogitBatcher(
            vocab_size=513,
            ring_rows=2,
            batch_rows=2,
            device=torch.device("cuda"),
            topk_backend="torch",
        )
        try:
            with (
                patch.object(batcher, "_compact", side_effect=RuntimeError("kernel")),
                patch("torch.cuda.synchronize", side_effect=RuntimeError("fence")),
            ):
                batcher.enqueue(torch.randn(1, 513, device="cuda"), [0], [(target, 1)])
                self.assertTrue(done.wait(timeout=5))
                batcher.thread.join(timeout=5)
            self.assertFalse(batcher.thread.is_alive())
            self.assertTrue(target.teacher_staging.transfer_uncertain)
            self.assertEqual(str(target.teacher_completion.error), "fence")
            self.assertEqual(batcher.stats()["occupied_rows"], 1)
            with self.assertRaisesRegex(Exception, "closed"):
                batcher.enqueue(torch.randn(1, 513, device="cuda"), [0], [(target, 1)])
            with self.assertRaisesRegex(Exception, "did not drain"):
                batcher.close()
        finally:
            batcher.stop.set()
            batcher.thread.join(timeout=5)
            torch.cuda.synchronize()

    def test_async_ring_owns_raw_rows_and_flushes_at_request_end(self):
        vocab = 513
        pool = HostBufferPool(
            kv=make_kv_spec(),
            max_tokens=4,
            slots=2,
            max_bytes=4 << 20,
            registrar=Registrar(),
            device=torch.device("cuda"),
            teacher_d2h_batch_tokens=4,
            max_device_bytes=4 << 20,
        )
        batcher = TeacherLogitBatcher(
            vocab_size=vocab,
            ring_rows=2,
            batch_rows=2,
            device=torch.device("cuda"),
            topk_backend="torch",
        )
        slots = [pool.acquire(), pool.acquire()]
        for slot in slots:
            for name in (
                "teacher_topk_ids",
                "teacher_topk_logits",
                "teacher_logsumexp",
            ):
                slot.tensors[name].fill_(-1)
        contexts = [
            RequestCaptureContext(
                slot=slot,
                prompt_ids=(3,),
                max_tokens=4,
                vocab_size=vocab,
                teacher_batcher=batcher,
            )
            for slot in slots
        ]
        try:
            raw = torch.randn(3, vocab + 7, device="cuda")
            expected = capture_teacher(raw, vocab, [2, 0])
            contexts[0].kv_end = 2
            contexts[1].kv_end = 1
            delayed = torch.cuda.Stream()
            delayed.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(delayed):
                torch.cuda._sleep(200_000_000)
                batcher.enqueue(
                    raw,
                    [2, 0],
                    [(contexts[0], 1), (contexts[1], 1)],
                )
                raw.fill_(float("nan"))
            with self.assertRaises(TeacherRingFull):
                batcher.enqueue(raw, [1], [(contexts[0], 2)])
            for index, context in enumerate(contexts):
                context.commit_token(position=1, token_id=index + 7)
                context.seal("length")
                self.assertEqual(context.slot.tensors["teacher_topk_ids"][0, 0], -1)
                context.wait_for_copies()
                torch.testing.assert_close(
                    context.slot.tensors["teacher_topk_ids"][0],
                    expected.token_ids[index].cpu(),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    context.slot.tensors["teacher_topk_logits"][0],
                    expected.logits[index].cpu(),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    context.slot.tensors["teacher_logsumexp"][0],
                    expected.logsumexp[index].cpu(),
                    rtol=0,
                    atol=0,
                )

            class RejectReservation:
                def validate_async_teacher(self, *, position):
                    pass

                def reserve_async_teacher(self, *, position):
                    raise ContractError("injected reservation failure")

            failed_stream = torch.cuda.Stream()
            failed_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(failed_stream):
                torch.cuda._sleep(200_000_000)
                with self.assertRaisesRegex(ContractError, "injected"):
                    batcher.enqueue(
                        torch.randn(1, vocab, device="cuda"),
                        [0],
                        [(RejectReservation(), 1)],
                    )
            batcher.work.join()
            self.assertTrue(failed_stream.query())
            self.assertEqual(batcher.stats()["occupied_rows"], 0)
            self.assertEqual(batcher.stats()["processed_rows"], 2)
            self.assertEqual(batcher.stats()["backpressure_rows"], 1)
        finally:
            batcher.close()
            for slot in slots:
                pool.release(slot, transfer_complete=True)
            pool.close()

    def check_flashinfer(self, raw, vocab, indices=None):
        scores = raw[:, :vocab]
        if indices is not None:
            scores = scores[indices]
        rows = capture_teacher(raw, vocab, indices, topk_backend="flashinfer")
        ids = rows.token_ids.long()
        expected = scores.topk(128, dim=-1).values.float()
        torch.testing.assert_close(
            rows.logits, expected, rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(
            rows.logits, scores.gather(1, ids).float(), rtol=0, atol=0, equal_nan=True
        )
        ordered = ids.sort(-1).values
        self.assertTrue((ordered[:, 1:] != ordered[:, :-1]).all())
        torch.testing.assert_close(
            rows.logsumexp,
            scores.float().double().logsumexp(-1).float(),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
        return rows

    def test_flashinfer_padding_selection_dtypes_and_owned_values(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64):
            for vocab in (128, 32768, 151936):
                for indices in (None, [], [2], [3, 0, 3, 1]):
                    with self.subTest(dtype=dtype, vocab=vocab, indices=indices):
                        raw = torch.randn(
                            8, (vocab + 17) * 2, device="cuda", dtype=dtype
                        )[::2, ::2]
                        raw[:, vocab:] = 10000
                        rows = self.check_flashinfer(raw, vocab, indices)
                        saved = [
                            value.clone()
                            for value in (rows.token_ids, rows.logits, rows.logsumexp)
                        ]
                        raw.fill_(float("nan"))
                        for value, expected in zip(
                            (rows.token_ids, rows.logits, rows.logsumexp), saved
                        ):
                            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_flashinfer_ties_and_nonfinite_values(self):
        raw = torch.zeros(7, 32769, device="cuda")
        raw[1].fill_(-10000)
        raw[2].fill_(10000)
        raw[3].fill_(-float("inf"))
        raw[4, 100] = float("inf")
        raw[5, 200] = float("nan")
        raw[6, :200] = 10
        rows = self.check_flashinfer(raw, 32769)
        expected = torch.arange(128, device="cuda", dtype=torch.int32).expand(4, -1)
        torch.testing.assert_close(rows.token_ids[:4], expected, rtol=0, atol=0)
        torch.testing.assert_close(rows.token_ids[6], expected[0], rtol=0, atol=0)
        self.assertFalse(torch.isfinite(rows.logsumexp[3:6]).any())

    def test_flashinfer_graph_and_concurrent_stream_workspaces(self):
        vocab = 151936
        warmup_teacher_capture(vocab, "cuda", topk_backend="flashinfer")
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        inputs = [torch.randn(8, vocab + 17, device="cuda") for _ in streams]
        graphs, outputs = [], []
        for raw, stream in zip(inputs, streams):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    capture_teacher(raw, vocab, topk_backend="flashinfer")
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                rows = capture_teacher(raw, vocab, topk_backend="flashinfer")
            graphs.append(graph)
            outputs.append(rows)
        for iteration in range(4):
            expected = []
            eager = []
            for raw, graph, stream in zip(inputs, graphs, streams):
                with torch.cuda.stream(stream):
                    raw.normal_().add_(iteration * 10)
                    expected.append(raw[:, :vocab].clone())
                    torch.cuda._sleep(1000000)
                    graph.replay()
                    eager.append(capture_teacher(raw, vocab, topk_backend="flashinfer"))
                    raw.fill_(float("nan"))
            for stream in streams:
                stream.synchronize()
            for rows, scores in zip(outputs + eager, expected + expected):
                torch.testing.assert_close(
                    rows.logits, scores.topk(128).values, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    rows.logits, scores.gather(1, rows.token_ids.long()), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    rows.logsumexp,
                    scores.double().logsumexp(-1).float(),
                    rtol=1e-6,
                    atol=1e-6,
                )

    def check_rows(self, raw, vocab, indices=None):
        scores = raw[:, :vocab]
        if indices is not None:
            scores = scores.index_select(
                0,
                (
                    torch.tensor(indices, dtype=torch.long, device=raw.device)
                    if isinstance(indices, list)
                    else indices
                ),
            )
        expected_values, expected_ids = scores.topk(128, dim=-1)
        # The public contract normalizes FP32 scores, including FP64 fallback.
        expected_lse = scores.float().double().logsumexp(-1).float()
        rows = capture_teacher(raw, vocab, indices)
        self.assertEqual(rows.token_ids.dtype, torch.int32)
        self.assertEqual(rows.logits.dtype, torch.float32)
        self.assertEqual(rows.logsumexp.dtype, torch.float32)
        torch.testing.assert_close(rows.token_ids, expected_ids.int(), rtol=0, atol=0)
        torch.testing.assert_close(
            rows.logits, expected_values.float(), rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(
            rows.logsumexp, expected_lse, rtol=1e-6, atol=1e-6, equal_nan=True
        )
        return rows

    def test_dtypes_padding_and_strided_rows_and_columns(self):
        torch.manual_seed(31)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for vocab in (128, 16385, 151936):
                for strided in (False, True):
                    with self.subTest(dtype=dtype, vocab=vocab, strided=strided):
                        raw = torch.randn(
                            8, (vocab + 17) * 2, device="cuda", dtype=dtype
                        )
                        raw = raw[::2, ::2] if strided else raw[:4, : vocab + 17]
                        raw[:, vocab:] = 10000
                        self.check_rows(raw, vocab)

    def test_selected_duplicate_and_empty_rows_and_float64_fallback(self):
        for dtype in (torch.float32, torch.float64):
            raw = torch.randn(4, 513, device="cuda", dtype=dtype)
            for indices in ([3, 0, 3, 1], []):
                with self.subTest(dtype=dtype, indices=indices):
                    self.check_rows(
                        raw,
                        511,
                        torch.tensor(indices, device="cuda", dtype=torch.int64),
                    )
            self.check_rows(raw[:0], 511)

    def test_offsets_masked_values_and_nonfinite_rows(self):
        vocab = 32769
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            raw = torch.zeros(9, vocab, device="cuda", dtype=dtype)
            raw[1].fill_(10000)
            raw[2].fill_(-10000)
            raw[3, 100] = 10000
            raw[4].fill_(-math.log(vocab))
            raw[5].fill_(-float("inf"))
            raw[5, 17] = 2.5
            raw[6].fill_(-float("inf"))
            raw[7, 100] = float("inf")
            raw[8, 200] = float("nan")
            with self.subTest(dtype=dtype):
                self.check_rows(raw, vocab)

    def test_host_row_selection_and_source_reuse(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for indices in ([], [2], [1, 2, 3], [3, 0, 3, 1]):
                with self.subTest(dtype=dtype, indices=indices):
                    raw = torch.randn(8, 1030, device="cuda", dtype=dtype)[::2, ::2]
                    rows = self.check_rows(raw, 511, indices)
                    saved = (
                        rows.token_ids.clone(),
                        rows.logits.clone(),
                        rows.logsumexp.clone(),
                    )
                    raw.fill_(float("nan"))
                    for actual, expected in zip(
                        (rows.token_ids, rows.logits, rows.logsumexp), saved
                    ):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for indices in ([-1], [4], [1, 4], [True], [1.0]):
            with self.subTest(invalid=indices), self.assertRaises(ContractError):
                capture_teacher(torch.zeros(4, 128, device="cuda"), 128, indices)

    def test_warmed_graph_replay_and_owned_outputs(self):
        vocab = 151936
        warmup_teacher_capture(vocab, "cuda")
        raw = torch.randn(4, vocab + 64, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                capture_teacher(raw, vocab)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            rows = capture_teacher(raw, vocab)
        for offset in (0.0, 9.0):
            raw.copy_(torch.randn_like(raw) + offset)
            expected = raw[:, :vocab].clone()
            graph.replay()
            raw.fill_(-float("inf"))
            torch.testing.assert_close(
                rows.logits, expected.gather(1, rows.token_ids.long()), rtol=0, atol=0
            )
            torch.testing.assert_close(
                rows.logsumexp,
                expected.double().logsumexp(-1).float(),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_host_contiguous_rows_in_graph(self):
        raw = torch.randn(8, 16400, device="cuda")
        warmup_teacher_capture(16385, "cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rows = capture_teacher(raw, 16385, [2, 3, 4])
        for offset in (0.0, 9.0):
            raw.copy_(torch.randn_like(raw) + offset)
            expected = raw[2:5, :16385].clone()
            graph.replay()
            raw.fill_(float("nan"))
            torch.testing.assert_close(
                rows.logits, expected.gather(1, rows.token_ids.long()), rtol=0, atol=0
            )
            torch.testing.assert_close(
                rows.logsumexp,
                expected.double().logsumexp(-1).float(),
                rtol=1e-6,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
