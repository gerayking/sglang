"""Test-only timing instrumentation; hidden capture itself uses runtime interfaces."""

import os
import sys

from sglang.srt.training_capture import context as context_module
from sglang.srt.training_capture.teacher_batcher import FinalHiddenCapture
from sglang.test import training_capture_writer_probe_server as probe


class ContextCUDA:
    def __getattr__(self, name):
        return getattr(probe.torch.cuda, name)

    def Event(self, *args, **kwargs):
        if probe.ACTIVE_MODE == "blocking_copy_events":
            kwargs["blocking"] = True
        return probe.torch.cuda.Event(*args, **kwargs)


class ContextTorch:
    cuda = ContextCUDA()

    def __getattr__(self, name):
        return getattr(probe.torch, name)


context_module.torch = ContextTorch()
if os.environ.get("SGLANG_TEST_CAPTURE_CPU_TIMINGS") == "1":
    FinalHiddenCapture.append = probe.cpu_timed(
        "hidden_enqueue", FinalHiddenCapture.append
    )
    FinalHiddenCapture.seal = probe.cpu_timed("hidden_seal", FinalHiddenCapture.seal)


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
