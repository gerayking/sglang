"""Test-only writer gate for measuring interference with concurrent serving."""

import os
import queue
import sys
import time
from pathlib import Path

from sglang.srt.training_capture.coordinator import CaptureCoordinator

writer_loop = CaptureCoordinator._writer_loop


def gated_writer_loop(self):
    gate = Path(os.environ["SGLANG_TEST_CAPTURE_WRITER_GATE"])
    get = self.work.get

    def gated_get(*args, **kwargs):
        if gate.exists() and not self.writer_stop.is_set():
            time.sleep(0.01)
            raise queue.Empty
        return get(*args, **kwargs)

    self.work.get = gated_get
    try:
        writer_loop(self)
    finally:
        self.work.get = get


CaptureCoordinator._writer_loop = gated_writer_loop


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
