# Reducing residual producer work

The follow-up attribution, including unprofiled thread CPU accounting and
same-server stage controls, is recorded in `RESIDUAL_CAPTURE_ATTRIBUTION.md`.

This change addresses the three follow-up items in
`PROCESS_WRITER_RESIDUAL_DIAGNOSIS.md`: L2 copy dispatch, per-step capture work,
and producer reservation work. Snapshot schema and teacher values are unchanged.
Operation counts and parent CPU work are reduced, and archived payloads match
the preceding version exactly. End-to-end throughput improvement is not yet
established: normalized capture overhead improves in the final sweep, but the
absolute baseline and captured throughput both change between versions.

## Implementation

1. `kernels/jit/csrc/kvcacheio/staged_write_back.cuh`: coalesce consecutive Host
   destination pages when copying from the contiguous GPU staging buffer. Each
   tensor and staging chunk remains separate. Gaps and descending destination
   addresses terminate a run. Reject the small-page batch-copy path before
   constructing all of its per-page descriptor arrays.
2. `model_executor/forward_batch_info.py` and `training_capture/coordinator.py`:
   the ordinary sequence-position constructors explicitly mark their output as
   canonical. Autoregressive capture then uses the existing Host `arange`
   positions. Overridden/unknown position constructors still copy actual
   positions. Cache the immutable capture configuration fingerprint once per
   coordinator, instead of recomputing it for every admitted request. PD Prefill
   initializes the same cache because it reuses the provenance builder.
3. `training_capture/writer_process.py` and `coordinator.py`: dispatch
   `Catalog.begin` to the existing CPU writer process. A lock serializes lease
   refill and publication on the command pipe, including all intermediate
   cancellation and prepared-manifest acknowledgments. Parent-side Host slot
   acquisition and conservative lease deadlines remain in the coordinator.

The following dependencies remain necessary:

- L2 staging waits for the actual forward stream, and serving waits before
  reusing the device KV slots.
- Teacher completion and shared Host payload completion precede publication.
- The writer receives the exact prepared-manifest acknowledgment before an
  ambiguous publication can occur; completion ACKs fence Host slot reuse.
- SHA-256, tensor validation, journal operations and publication are unchanged.

Moving `begin` does not move heartbeat, cancellation, all scheduler bookkeeping,
or teacher compaction into the child. Refill can wait behind a publication on
the serialized command pipe. The existing bounded Host pool and fail-open
admission policy still apply.

## Verification

One H100 80 GB, TP=1, Qwen3.5-4B BF16, language model only, HiCache L2
write-through, page size 1, selected KV layers 3/15/31. Throughput phases use
128 requests, input/output 32/32, concurrency 4, 100% capture, 128 Host slots,
32 raw-logit ring rows and teacher batch size 4. Transport is local TCP Mooncake
with the HTTP test Catalog. This is not an RDMA or multi-GPU result.

The targeted suite passed **143 tests and 161 subtests** on the H100: 126 tests
and 125 subtests in the main run, followed by 17 tests and 36 subtests for PD
compatibility after adding the Prefill fingerprint initialization.

- Coordinator admission, lifecycle, canonical and overridden position handling.
- Request contexts, asynchronous teacher capture and L2 export.
- Process writer completion, cancellation, failures and concurrent lease/write
  commands, including an injected Catalog.begin failure followed by a good write.
- CUDA MHA/MLA L2 write-back with continuous, reversed and fragmented pages,
  crossing staging boundaries and checking that untouched pages remain intact.

The profiler workload is separate: 8 requests with 1 input and 32 output tokens.
The before trace is the preceding residual-diagnosis trace; the after trace uses
this implementation. Both have 66 target forwards.

| Activity | Before | After |
| --- | ---: | ---: |
| L2 D2H calls | 528 | 16 |
| L2 D2H bytes | 8,650,752 | 8,650,752 |
| L2 D2H device time | 1.358 ms | 0.221 ms |
| Position D2H calls | 264 | 0 |
| Raw-logit ring D2D calls | 64 | 64 |
| Parent `_reserve` thread CPU, 21 calls | 7.745 ms | 2.408 ms |

These are work totals, not additive serving-latency reductions. Profiling also
adds CPU dispatch overhead. L2 still stages all eight full-attention layers;
selected-layer materialization and payload size are unchanged. Coalescing saves
dispatches, not transferred bytes. The teacher still computes full-vocabulary
top-128/LSE through the existing bounded background ring.

## Throughput interpretation

The final sweep fixes CPU and memory placement to the H100's NUMA node 1 using
`numactl --cpunodebind=1 --membind=1`. Both scheduler and CPU writer affinities
were checked. It uses identical diagnostic probes and three repeats per source
version, with the paused/concurrent order reversed in the middle round.

| Version | Paused token/s observations | Full capture token/s observations | Loss vs own paused mean |
| --- | --- | --- | ---: |
| Before | 578.36, 578.76, 580.58 | 549.35, 548.28, 540.87 | 5.71% |
| After | 554.19, 556.44, 543.91 | 529.60, 527.89, 527.19 | 4.22% |

The normalized difference is 1.49 percentage points. However, paused throughput
also moves from 579.24 to 551.51 token/s, and full capture moves from 546.17 to
528.23 token/s. NUMA binding has not removed the cross-run baseline difference.
This is directional evidence for lower incremental capture overhead, **not a
demonstration of improved absolute throughput or proof that regressions are
excluded**. The cause of the remaining baseline change is not isolated. All 768
captured samples in this final pair pass Store readback.

An initial three-repeat sweep uses the same server for paused/full capture and
reverses their order in the middle round. The source before this change was
frozen before editing; both variants retain the final-forward/L2 correctness
fence.

| Version | Paused token/s mean | Full capture token/s mean | Loss vs own paused |
| --- | ---: | ---: | ---: |
| Before | 574.70 | 552.39 | 3.88% |
| After, initial sweep | 547.36 | 523.74 | 4.32% |

The paused baseline changed substantially between server launches. These
numbers do **not** establish an end-to-end throughput improvement. All 384
captured samples in each sweep passed Store readback; none were dropped.

A follow-up sweep restores one old path at a time in the same running server.
All modes below retain the coalesced L2 path. The three measurements are shown
in full because averaging alone would obscure the changing request latency.

| Mode | Three token/s observations | Mean |
| --- | --- | ---: |
| Paused | 542.78, 558.96, 544.78 | 548.84 |
| All changes | 519.65, 525.06, 519.37 | 521.36 |
| Restore position D2H | 509.66, 510.94, 547.75 | 522.78 |
| Restore parent Catalog.begin | 520.74, 504.46, 552.72 | 525.98 |

The last-round controls improve while restoring the old work. Their mean TTFT
falls to 98.72/97.33 ms, compared with 110.65 ms for that round's full new path.
The experiment does not isolate the cause of this drift, and does not establish
a stable throughput benefit for either change. It must not be summarized using
only the favorable first two rounds. All 1152 captured samples pass readback.

The ordinary server entry point also completed an off/on/off sweep, with all
three phases started independently: 573.80 / 520.53 / 542.00 token/s. Its two
off baselines differ by 5.54%; capture is 6.70% below their mean. This run is
retained as an unfavorable performance result, not discarded as noise.

After stopping the producer, all 128 captured snapshots were read from Mooncake,
archived as manifest plus safetensors, and loaded back from disk with validation.
The archive contains 1792 tensor objects and 105,005,056 payload bytes. An
additional local pass verifies container, manifest and individual tensor hashes.
All 128 full token sequences match the earlier fenced process-writer archive;
all 1792 corresponding object descriptors and payload SHA-256 values match
exactly. This covers KV, positions, masks, token IDs, top-128 IDs/logits and LSE.
It is a cross-version equivalence check, not independent teacher recomputation.

## Artifacts and reproduction

Artifacts are kept outside the implementation checkout:

```
/gpfs/user/fuxuanwei/sglang/mooncake-study/experiments/residual-opt-20261012/
  before/report.json
  after/report.json
  components/report.json
  numa-before/report.json
  numa-after/report.json
  numa-binding.txt
  numa-{before,after}-processes.json
  normal/report.json
  normal/round-0-phase-1-1.0/snapshot-archive/
  archive-comparison.json
  summary.json
  profile/report.json
  profile/{off,on}/{prefill,decode}/*.trace.json.gz
  source-before/python/sglang/...
  source-after/python/sglang/...
  residual-opt-tests.log
  residual-opt-pd-tests.log
  implementation.patch
  compare_archives.py
```

The diagnostic script records source hashes, including the L2 C++ header and
ForwardBatch position constructor. For a frozen source tree, run:

```bash
python mooncake-study/experiments/benchmark_writer_prefixes.py \
  --config-report /tmp/process-writer-fenced/report.json \
  --output-dir /tmp/residual-opt-after \
  --snapshot-writer-backend process --repeats 3 \
  --modes paused concurrent
```

To isolate position and lease work inside one running server, use
`--modes paused concurrent original_positions parent_lease`. These last two
controls exist only in the diagnostic server: they restore per-step position
copies or parent-side Catalog.begin respectively. All controls retain the new
L2 copy path and complete normal publication/readback. Never use diagnostic
server entry points for production serving.

For the NUMA pair, use separate frozen source trees and prepend
`numactl --cpunodebind=1 --membind=1` to each benchmark command. Keep the probe
implementation identical: the after tree in this pair uses the original probe,
without the additional position/lease restoration wrappers. Production code
is the same as the current autoregressive capture path. This extra control did
not eliminate the baseline drift.
