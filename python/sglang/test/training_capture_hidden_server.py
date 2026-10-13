"""Test-only, bounded post-norm hidden capture with the normal L2/process writer."""

import os
import sys
import threading

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.training_capture import context as context_module
from sglang.srt.training_capture import host_pool, resources
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.coordinator import CaptureCoordinator
from sglang.srt.training_capture.protocol import ContractError
from sglang.srt.training_capture.writer_process import ProcessSnapshotWriter
from sglang.test import training_capture_writer_probe_server as probe
from sglang.test.training_capture_hidden_codec import (
    HIDDEN_SIZE,
    NORMAL_AUX,
    install,
    validate_tensors,
    writer_factory,
)

install()
probe.original_validate = validate_tensors
LOCAL = threading.local()


class ContextCUDA:
    def __getattr__(self, name):
        return getattr(torch.cuda, name)

    def Event(self, *args, **kwargs):
        if probe.ACTIVE_MODE == "blocking_copy_events":
            kwargs["blocking"] = True
        return torch.cuda.Event(*args, **kwargs)


class ContextTorch:
    cuda = ContextCUDA()

    def __getattr__(self, name):
        return getattr(torch, name)


context_module.torch = ContextTorch()


def lm_head_hidden(
    self,
    hidden_states,
    hidden_states_before_norm,
    aux_hidden_states,
    pruned_states,
    pruned_states_before_norm,
    aux_pruned_states,
    sample_indices,
    logits_metadata,
):
    # Returned in LogitsProcessorOutput, including on CUDA graph replay.
    if pruned_states.shape[-1] != HIDDEN_SIZE or aux_hidden_states is not None:
        raise ContractError("hidden probe supports Qwen3.5-4B AR only")
    return pruned_states if sample_indices is None else pruned_states[sample_indices]


LogitsProcessor._get_hidden_states_to_store = lm_head_hidden
pool_init = host_pool.HostBufferPool.__init__
hidden_aux = host_pool.aux_specs


def allocate_pool(self, **kwargs):
    host_pool.aux_specs = lambda n, r: NORMAL_AUX(n, r) | hidden_aux(n, r)
    try:
        pool_init(self, **kwargs)
    finally:
        host_pool.aux_specs = hidden_aux
    extra = kwargs["slots"] * kwargs["max_tokens"] * HIDDEN_SIZE * 2
    if self.device_allocated_bytes + extra > kwargs["max_device_bytes"]:
        raise ContractError("hidden staging exceeds device memory budget")
    for slot in self.slots:
        slot.teacher_device_tensors = {
            "teacher_final_hidden": torch.empty(
                (kwargs["max_tokens"], HIDDEN_SIZE),
                device=kwargs["device"],
                dtype=torch.bfloat16,
            )
        }
    self.device_allocated_bytes += extra


host_pool.HostBufferPool.__init__ = allocate_pool


class HiddenBatcher:
    def __init__(self, *, device, **kwargs):
        self.rows = 0
        self.copy_stream = torch.cuda.Stream(device=device)

    def enqueue(self, logits, indices, records):
        hidden = LOCAL.hidden
        if (
            hidden is None
            or hidden.dtype != torch.bfloat16
            or hidden.shape[-1] != HIDDEN_SIZE
        ):
            raise ContractError("missing post-norm hidden output")
        stream = torch.cuda.current_stream(hidden.device)
        for row, (context, position) in zip(indices, records, strict=True):
            context.validate_async_teacher(position=position)
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
        hidden.record_stream(stream)

    def stats(self):
        return {
            "hidden_rows": self.rows,
            "processed_rows": self.rows,
            "pending_rows": 0,
        }

    def close(self):
        self.copy_stream.synchronize()


resources.TeacherLogitBatcher = HiddenBatcher
after_forward = CaptureCoordinator.after_forward


def capture_hidden(self, batch, forward_batch, logits_output, **kwargs):
    LOCAL.hidden = logits_output.hidden_states
    try:
        return after_forward(self, batch, forward_batch, logits_output, **kwargs)
    finally:
        LOCAL.hidden = None


CaptureCoordinator.after_forward = capture_hidden
seal = RequestCaptureContext.seal


def seal_hidden(self, stop_reason):
    staging = self.teacher_staging
    if staging is not None and staging.end:
        # One producer fence per completed request. All D2H uses a separate stream.
        try:
            event = torch.cuda.Event()
            event.record(staging.stream)
            staging.stream = self.teacher_batcher.copy_stream
            staging.stream.wait_event(event)
        except Exception:
            self.transfer_uncertain = True
            raise
    return seal(self, stop_reason)


RequestCaptureContext.seal = seal_hidden
if os.environ.get("SGLANG_TEST_CAPTURE_CPU_TIMINGS") == "1":
    HiddenBatcher.enqueue = probe.cpu_timed("hidden_enqueue", HiddenBatcher.enqueue)
    RequestCaptureContext.seal = probe.cpu_timed("hidden_seal", seal_hidden)
    for method in ("_flush_staging", "_record_completion"):
        setattr(
            RequestCaptureContext,
            method,
            probe.cpu_timed("hidden" + method, getattr(RequestCaptureContext, method)),
        )
    event_synchronize = torch.cuda.Event.synchronize
    wait_for_copies = RequestCaptureContext.wait_for_copies

    def copy_context(self):
        LOCAL.copy_context = self
        try:
            return wait_for_copies(self)
        finally:
            LOCAL.copy_context = None

    RequestCaptureContext.wait_for_copies = copy_context

    def synchronize_copy(event):
        if threading.current_thread().name == "training-snapshot-writer":
            context = getattr(LOCAL, "copy_context", None)
            name = (
                "hidden_d2h_synchronize"
                if context is not None and event is context.last_event
                else "l2_synchronize"
            )
            with probe.scope("cpu/" + name):
                return event_synchronize(event)
        return event_synchronize(event)

    torch.cuda.Event.synchronize = synchronize_copy

writer_init = ProcessSnapshotWriter.__init__


def initialize_writer(self, config, pool, **kwargs):
    return writer_init(self, config, pool, writer_factory=writer_factory)


ProcessSnapshotWriter.__init__ = initialize_writer


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
