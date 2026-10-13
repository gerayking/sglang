"""Progressive real writer work, with final Store readback for every sample."""

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from unittest.mock import patch

import benchmark_training_capture as bench
import requests
import torch

MODES = (
    "paused",
    "cpu_positions",
    "teacher_deferred",
    "none",
    "copies",
    "build",
    "validation",
    "registered",
    "stored",
    "published",
    "cached_sha",
    "checksum_only",
    "cached_sha_checksum_only",
    "concurrent",
    "original_positions",
    "parent_lease",
    "blocking_teacher",
    "blocking_copy_events",
)


@contextmanager
def probe_server(*args, **kwargs):
    launch = bench.test_utils._launch_server_process

    def annotated(command, *launch_args):
        return launch(
            [sys.executable, "-m", "sglang.test.training_capture_writer_probe_server"]
            + command[2:],
            *launch_args,
        )

    with patch.object(bench.test_utils, "_launch_server_process", annotated):
        yield


def delta(after, before):
    return {
        name: {
            key: value - before.get(name, {}).get(key, 0)
            for key, value in fields.items()
        }
        for name, fields in after.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES[3:]))
    parser.add_argument("--snapshot-writer-backend", choices=("thread", "process"))
    parser.add_argument("--teacher-gpu-ring-rows", type=int)
    parser.add_argument("--device-mib", type=int)
    parser.add_argument("--cpu-timings", action="store_true")
    parser.add_argument("--interleave-paused", action="store_true")
    cli = parser.parse_args()
    args = argparse.Namespace(**json.loads(cli.config_report.read_text())["config"])
    args.output_dir = cli.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.writer_mode = "deferred"
    args.request_details = False
    args.archive_readback = False
    args.segment_mib = 8192
    args.teacher_gpu_ring_rows = max(32, args.teacher_gpu_ring_rows)
    args.repeats = cli.repeats
    if cli.snapshot_writer_backend is not None:
        args.snapshot_writer_backend = cli.snapshot_writer_backend
    else:
        args.snapshot_writer_backend = getattr(
            args, "snapshot_writer_backend", "thread"
        )
    if cli.teacher_gpu_ring_rows is not None:
        args.teacher_gpu_ring_rows = cli.teacher_gpu_ring_rows
    if cli.device_mib is not None:
        args.device_mib = cli.device_mib
    if (
        "teacher_deferred" in cli.modes
        and args.teacher_gpu_ring_rows < args.num_prompts * args.output_len
    ):
        raise ValueError("deferred teacher requires space for every raw logits row")
    if args.snapshot_writer_backend == "process" and any(
        mode
        not in (
            "paused",
            "none",
            "copies",
            "published",
            "concurrent",
            "cpu_positions",
            "teacher_deferred",
            "original_positions",
            "parent_lease",
            "blocking_teacher",
            "blocking_copy_events",
        )
        for mode in cli.modes
    ):
        raise ValueError("unsupported process writer prefix")
    if args.capture_slots < args.num_prompts:
        raise ValueError("one reserved capture slot per measured request is required")
    torch.set_num_threads(1)
    results = []
    report = {
        "status": "running",
        "modes": cli.modes,
        "repeats": cli.repeats,
        "cpu_timings": cli.cpu_timings,
        "interleave_paused": cli.interleave_paused,
        "driver_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "source_sha256": {
            str(path.relative_to(bench.ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in [
                Path(__file__),
                bench.ROOT
                / "python/sglang/test/training_capture_writer_probe_server.py",
                bench.ROOT / "python/sglang/srt/model_executor/forward_batch_info.py",
                bench.ROOT / "python/sglang/srt/managers/cache_controller.py",
                bench.ROOT
                / "python/sglang/kernels/jit/csrc/kvcacheio/staged_write_back.cuh",
            ]
            + sorted((bench.ROOT / "python/sglang/srt/training_capture").glob("*.py"))
        },
        "measurements": results,
        "scope": "Single H100, local TCP Store, test HTTP Catalog. Prefix records stay held until client exits; partial prefixes repeat outside client timing. Paused retains the capture subsystem but admits no samples. CPU positions uses the initialized AR position buffer. Checksum shortcuts are diagnostic only. Final readback independently validates every published sample.",
    }
    output = args.output_dir / "report.json"
    original_measure = bench.measure
    original_wait = bench.wait_capture_idle
    original_store = bench.isolated_store
    resources = {}

    @contextmanager
    def retained_store(*store_args):
        with original_store(*store_args) as handles:
            resources.update(reader=handles[1], catalog=handles[2])
            yield handles

    def sweep(phase_args, url, directory):
        initial, _ = original_wait(url)
        gate = directory / "writer-gate"
        count = phase_args.num_prompts
        for repetition in range(cli.repeats):
            modes = cli.modes if repetition % 2 == 0 else list(reversed(cli.modes))
            if cli.interleave_paused:
                modes = [
                    item
                    for mode in modes
                    if mode != "paused"
                    for item in ("paused", mode)
                ] + ["paused"]
            for phase_index, mode in enumerate(modes):
                label = f"{phase_index}-" if cli.interleave_paused else ""
                current = directory / f"round-{repetition}-{label}{mode}"
                current.mkdir()
                before, _ = original_wait(url)
                requests.post(
                    url + "/control_training_capture",
                    json={"action": "pause" if mode == "paused" else "resume"},
                    timeout=30,
                ).raise_for_status()
                catalog = resources["catalog"]
                with catalog.condition:
                    previous_publications = set(catalog.publications)
                requests.post(url + "/flush_cache", timeout=30).raise_for_status()
                state = {}
                wait_calls = 0

                def release_before_drain(
                    *wait_args,
                    before=before,
                    mode=mode,
                    state=state,
                    current=current,
                    **wait_kwargs,
                ):
                    nonlocal wait_calls
                    wait_calls += 1
                    if wait_calls == 2:
                        started = time.monotonic()
                        while True:
                            held = bench.capture_state(url)
                            bench.write_json(current / "held.json", held)
                            probe = delta(held["writer_probe"], before["writer_probe"])
                            if probe.get("prefix_error", {}).get("calls", 0):
                                raise RuntimeError("writer prefix failed")
                            failed = {
                                key: value - before["counters"].get(key, 0)
                                for key, value in held["counters"].items()
                                if key.startswith(("failed_", "writer_failed_"))
                                and value > before["counters"].get(key, 0)
                            }
                            if failed:
                                raise RuntimeError(f"capture failed: {failed}")
                            done = probe.get("prefix/" + mode, {}).get("calls", 0)
                            if (
                                mode
                                in (
                                    "none",
                                    "concurrent",
                                    "paused",
                                    "cpu_positions",
                                    "teacher_deferred",
                                    "original_positions",
                                    "parent_lease",
                                    "blocking_teacher",
                                    "blocking_copy_events",
                                )
                                or done == count
                            ):
                                break
                            if time.monotonic() - started > 30:
                                raise TimeoutError(
                                    f"prefix incomplete: {mode}: {done}/{count}"
                                )
                            time.sleep(0.05)
                        state.update(
                            held=held,
                            probe=probe,
                            prefix_tail_seconds=time.monotonic() - started,
                        )
                        if mode == "teacher_deferred" and (
                            held["teacher_batcher"]["processed_rows"]
                            != before["teacher_batcher"]["processed_rows"]
                        ):
                            raise RuntimeError(
                                "teacher compaction ran before gate release"
                            )
                        gate.unlink(missing_ok=True)
                    return original_wait(*wait_args, **wait_kwargs)

                measured_args = copy(phase_args)
                measured_args.writer_mode = "concurrent"
                try:
                    if mode not in ("concurrent", "paused"):
                        bench.write_json(gate, {"mode": mode})
                        time.sleep(0.2)
                    with patch.object(bench, "wait_capture_idle", release_before_drain):
                        result = original_measure(measured_args, url, current)
                finally:
                    gate.unlink(missing_ok=True)
                result.update(
                    mode=mode,
                    repetition=repetition,
                    phase_index=phase_index,
                    capture_before=before,
                    **state,
                )
                counters = result["capture_counter_delta"]
                expected = 0 if mode == "paused" else count
                if (
                    counters.get("ready", 0) != expected
                    or counters.get("admitted", 0) != expected
                ):
                    raise RuntimeError(f"lost samples: {mode}: {counters}")
                with catalog.condition:
                    publications = [
                        value
                        for key, value in catalog.publications.items()
                        if key not in previous_publications
                    ]
                result["readback"] = bench.validate_publications(
                    resources["reader"], publications, measured_args
                )
                if result["readback"].get("validated_samples", 0) != expected:
                    raise RuntimeError("per-phase readback missed samples")
                bench.write_json(current / "measurement.json", result)
                results.append(result)
                bench.write_json(output, report)
                print(
                    json.dumps(
                        {
                            "repetition": repetition,
                            "mode": mode,
                            "tok_s": result["serving"]["output_throughput"],
                            "ttft_ms": result["serving"]["median_ttft_ms"],
                            "ready": counters.get("ready", 0),
                        }
                    ),
                    flush=True,
                )
        final, _ = original_wait(url)
        aggregate = copy(results[-1])
        aggregate["capture_counter_delta"] = {
            key: value - initial["counters"].get(key, 0)
            for key, value in final["counters"].items()
        }
        aggregate["capture_router_delta"] = {
            key: value - initial.get("request_router", {}).get(key, 0)
            for key, value in final.get("request_router", {}).items()
        }
        phase_args.num_prompts = count * len(results)
        return aggregate

    try:
        with (
            patch.dict(
                os.environ,
                {"SGLANG_TEST_CAPTURE_CPU_TIMINGS": "1"} if cli.cpu_timings else {},
            ),
            patch.object(bench, "measure", sweep),
            patch.object(bench, "diagnostic_server", probe_server),
            patch.object(bench, "isolated_store", retained_store),
        ):
            report["verification"] = bench.run_phase(
                args, args.output_dir / "server", 1.0
            )
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        bench.write_json(output, report)


if __name__ == "__main__":
    main()
