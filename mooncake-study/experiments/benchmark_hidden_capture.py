"""Compare full hidden publication to paused capture; reuse the writer sweep."""

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import benchmark_training_capture as bench
import benchmark_writer_prefixes as sweep


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--feature", choices=("hidden", "top128"), required=True)
    cli, remainder = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remainder]
    if cli.feature == "hidden":
        from sglang.test import training_capture_catalog
        from sglang.test import training_capture_hidden_codec as codec

        training_capture_catalog.decode_manifest = codec.decode_manifest
        catalog_init = bench.TestCaptureCatalog.__init__

        def hidden_catalog(self):
            catalog_init(self)
            self.capabilities["contracts"][0].update(
                contract_id=codec.CONTRACT,
                payload_format="experimental_target_kv_final_hidden_v1",
            )

        bench.TestCaptureCatalog.__init__ = hidden_catalog
        bench.decode_manifest = codec.decode_manifest
        bench.validate_tensors = codec.validate_tensors
        config = bench.capture_config

        def hidden_config(*args):
            return config(*args) | {
                "contract_id": codec.CONTRACT,
                "teacher_capture_backend": "final_hidden",
            }

        bench.capture_config = hidden_config

        @contextmanager
        def hidden_server(*args, **kwargs):
            launch = bench.test_utils._launch_server_process

            def annotated(command, *launch_args):
                return launch(
                    [sys.executable, "-m", "sglang.test.training_capture_hidden_server"]
                    + command[2:],
                    *launch_args,
                )

            with patch.object(bench.test_utils, "_launch_server_process", annotated):
                yield

        sweep.probe_server = hidden_server
        validate = bench.validate_publications
        archive_count = 0

        def readback(reader, publications, args, **kwargs):
            nonlocal archive_count
            result = validate(reader, publications, args, **kwargs)
            if publications and archive_count < 2 and "trace_ids" not in kwargs:
                result["hidden_archive"] = validate(
                    reader,
                    publications[:8],
                    args,
                    archive_directory=args.output_dir
                    / f"hidden-readback-{archive_count}",
                )
                archive_count += 1
            return result

        bench.validate_publications = readback
    try:
        sweep.main()
    finally:
        report_path = (
            Path(remainder[remainder.index("--output-dir") + 1]) / "report.json"
        )
        if report_path.exists():
            report = json.loads(report_path.read_text())
            report["teacher_feature"] = cli.feature
            report["hidden_probe_sources"] = {
                str(path.relative_to(bench.ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in (
                    Path(__file__),
                    bench.ROOT / "python/sglang/test/training_capture_hidden_codec.py",
                    bench.ROOT / "python/sglang/test/training_capture_hidden_server.py",
                )
            }
            bench.write_json(report_path, report)


if __name__ == "__main__":
    main()
