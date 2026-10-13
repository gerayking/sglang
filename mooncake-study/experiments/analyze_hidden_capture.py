"""Paired throughput deltas and exclusive CPU timers for the hidden experiment."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(path):
    report = json.loads(path.read_text())
    rows = report["measurements"]
    groups = defaultdict(list)
    for row in rows:
        if row["mode"] == "paused":
            continue
        controls = [
            value
            for value in rows
            if value["mode"] == "paused" and value["repetition"] == row["repetition"]
        ]
        before = [
            value for value in controls if value["phase_index"] < row["phase_index"]
        ]
        after = [
            value for value in controls if value["phase_index"] > row["phase_index"]
        ]
        if not before or not after:
            continue
        neighbors = [before[-1], after[0]]
        rate = row["serving"]["output_throughput"]
        baseline = statistics.mean(
            value["serving"]["output_throughput"] for value in neighbors
        )
        control_rates = [value["serving"]["output_throughput"] for value in neighbors]
        baseline_span = 100 * (max(control_rates) - min(control_rates)) / baseline
        tokens = row["serving"]["total_output_tokens"]
        timers = {}
        for name, fields in row.get("probe", {}).items():
            control_cpu = statistics.mean(
                value.get("probe", {}).get(name, {}).get("self_cpu", 0)
                for value in neighbors
            )
            timers[name] = {
                "self_cpu_ms": fields["self_cpu"] * 1000,
                "control_adjusted_self_cpu_ms": (fields["self_cpu"] - control_cpu)
                * 1000,
                "self_wall_ms": fields["self_wall"] * 1000,
                "calls": fields["calls"],
            }
        groups[row["mode"]].append(
            {
                "repetition": row["repetition"],
                "throughput": rate,
                "baseline": baseline,
                "baseline_span_percent": baseline_span,
                "baseline_unstable": baseline_span > 3.0,
                "loss_percent": 100 * (1 - rate / baseline),
                "extra_serving_ms": 1000 * tokens * (1 / rate - 1 / baseline),
                "mean_ttft_ms": row["serving"]["mean_ttft_ms"],
                "mean_tpot_ms": row["serving"]["mean_tpot_ms"],
                "validated_samples": row["readback"].get("validated_samples", 0),
                "prefix_tail_seconds": row.get("prefix_tail_seconds", 0),
                "timers": timers,
            }
        )
    result = {"source": str(path), "status": report["status"], "modes": {}}
    for mode, values in groups.items():
        names = set().union(*(value["timers"] for value in values))
        summary = {
            key: statistics.mean(value[key] for value in values)
            for key in (
                "throughput",
                "baseline",
                "loss_percent",
                "extra_serving_ms",
                "mean_ttft_ms",
                "mean_tpot_ms",
            )
        }
        summary["loss_percent_range"] = [
            min(value["loss_percent"] for value in values),
            max(value["loss_percent"] for value in values),
        ]
        summary["validated_samples"] = sum(
            value["validated_samples"] for value in values
        )
        summary["timers_mean"] = {
            name: {
                field: statistics.mean(
                    value["timers"].get(name, {}).get(field, 0) for value in values
                )
                for field in (
                    "self_cpu_ms",
                    "control_adjusted_self_cpu_ms",
                    "self_wall_ms",
                    "calls",
                )
            }
            for name in sorted(names)
        }
        summary["runs"] = values
        summary["unstable_control_repetitions"] = [
            value["repetition"] for value in values if value["baseline_unstable"]
        ]
        result["modes"][mode] = summary
    modes = result["modes"]
    if all(name in modes for name in ("none", "copies", "concurrent")):
        result["stage_extra_serving_ms"] = {
            "foreground_and_l2_retention": modes["none"]["extra_serving_ms"],
            "parent_d2h_and_l2_materialization": modes["copies"]["extra_serving_ms"]
            - modes["none"]["extra_serving_ms"],
            "publication_retirement_and_refill": modes["concurrent"]["extra_serving_ms"]
            - modes["copies"]["extra_serving_ms"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = [analyze(path) for path in args.reports]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for report in result:
        print(report["source"], report["status"])
        for mode, value in report["modes"].items():
            print(
                mode,
                {
                    key: value[key]
                    for key in (
                        "throughput",
                        "baseline",
                        "loss_percent",
                        "loss_percent_range",
                        "extra_serving_ms",
                        "validated_samples",
                    )
                },
            )
        print(report.get("stage_extra_serving_ms", {}))


if __name__ == "__main__":
    main()
