# Last-layer hidden capture: performance and attribution

Date: 2026-10-13. This is an experimental Qwen3.5-4B capture entry point,
not a production schema or a SpecForge training adapter.

## Saved Proposal and Payload Breakdown

The proposal is parked as of 2026-10-13: preserve the current experimental
implementation and measurements for later work. The saved path captures the
post-final-norm hidden for each response prediction, selected target KV layers,
and token/mask/position fields. It uses bounded GPU staging, request-end hidden
D2H, KV materialization from HiCache L2, and a separate CPU writer process for
construction, validation and Mooncake publication. A consumer would still need
an explicit adapter and LM-head/top-128 enrichment for the intended DSpark loss.
Hidden-only capture has not been measured.

All 16 archived manifests in `attribution/hidden-readback-{0,1}` have the same
payload sizes. For this workload (32 prompt + 32 response tokens, BF16 KV from
layers 3/15/31, and BF16 response hidden of width 2,560):

| Payload | Bytes per sample | KiB | Share of tensor bytes |
| --- | ---: | ---: | ---: |
| Selected KV | 786,432 | 768 | 82.67% |
| Final-layer response hidden | 163,840 | 160 | 17.22% |
| Token IDs, masks and positions | 1,024 | 1 | 0.11% |
| Total | 951,296 | 929 | 100% |

The auxiliary 1,024 bytes comprise token IDs (256), loss mask (64), position IDs
(512), KV-valid mask (64), and prediction positions (128). This table excludes
manifest JSON, transport framing, store bookkeeping, and reserved staging
capacity. It describes stored tensor payload, not total device-to-host traffic.

With the same layer geometry, selected KV costs 12,288 bytes per stored KV token
(`3 layers * 2 K/V * 4 heads * 256 dimensions * 2 bytes`); hidden costs 5,120
bytes per response token. Thus the relative sizes also depend on prompt length,
response length, KV coverage, selected layers, and dtype.

Payload share is not serving-overhead share. The measured largest overhead
group is publication, retirement and lease refill (45.9% of the diagnostic
extra serving duration). Foreground staging/bookkeeping plus L2 retention is
31.9%, and parent D2H/completion plus L2 materialization is 22.1%. These controls
do not isolate KV versus hidden versus token/mask costs; no such per-field
performance attribution is established. In particular, small token/mask payload
does not prove negligible bookkeeping cost.

## Workload and isolation

- One H100 80 GB on `node025`, pod
  `job-83e3834bcf07-20261013140313-master-0`, CPU affinity 48-55.
- Qwen3.5-4B, BF16, TP1, greedy decoding, concurrency 4.
- Each timed phase has 128 requests, 32 input tokens and 32 output tokens.
- Decode CUDA graphs enabled. HiCache L2 remains enabled in every control,
  with write-through policy and page size 1. Selected KV layers: 3, 15, 31.
- Request slots: 128; Host budget: 512 MiB; device budget: 128 MiB.
- The existing process writer performs hashing, validation, Mooncake writes,
  and Catalog publication. The test uses local TCP Mooncake and a test HTTP
  Catalog, not cross-machine RDMA or concurrent disk persistence.
- Each active phase is bracketed by paused-capture phases in the same server.
  Paused capture keeps the subsystem allocated but admits no training samples.
- All deferred work is drained and all published tensors read back and validated
  outside the serving measurement. Selected samples are also archived to disk.

The requested top-128 comparison was stopped when the user redirected the
experiment to the hidden path's residual overhead. It is not used in this report.

## Capture implementation

The experimental logits-processor hook returns the actual post-final-norm,
pre-LM-head hidden tensor through `LogitsProcessorOutput.hidden_states`.
CUDA graph replay returns the graph's current tensor, not a Python hook's
stale last value. The normal serving hidden-return API can prefer pre-norm
states, which would be the wrong tensor for reconstructing logits.

For each accepted response prediction, the producer copies one BF16 vector
into bounded, request-owned GPU storage. The original response position ledger
checks alignment. At completion, one producer event makes a separate copy
stream wait for all rows; the parent writer thread then issues one batched
hidden D2H, waits for completion, and materializes selected KV from L2 into
the shared Host arena. The CPU child receives the slot index and metadata,
then builds, validates, writes and publishes the snapshot normally.

The hidden vector has 2,560 elements: 5,120 bytes per response token. It is
larger than compact top-128 IDs/scores/LSE (1,028 bytes), but avoids capturing
248,320 FP32 vocabulary scores (993,280 bytes) and avoids online top-k/LSE
compaction. Serving still computes its normal LM head for generation.

The separate wire format `experimental_target_kv_final_hidden_v1` contains
`teacher_final_hidden[R,2560]`, has `logits: null`, and preserves KV, token IDs,
positions and masks. The ordinary top-128 decoder rejects it. Training on it
would require a subsequent LM-head/top-128 enrichment step and an explicit
consumer adapter; that work is not included in these measurements.

The subsequent paired SpecForge change (`gerayking/SpecForge`, local PR commit
`5ef53e1`) implements that consumer and trainer-side reconstruction. A separate
single-H100 smoke replayed 16 validated serving archives into a live TCP Store,
trained on four samples, and resumed from step 2 through step 4. It does not
extend these serving-performance measurements to simultaneous streaming training.

## Initial Measurement

The uninstrumented smoke run measured 559.707 output tokens/s with full hidden
capture against a bracketing paused baseline of 578.319 tokens/s: **3.218% loss**,
or 235.5 ms additional serving duration per 128 requests. This was one active
phase, not an established multi-run performance estimate.

## Cumulative Stage Controls

Three repetitions reverse stage order in alternating rounds. These runs add
CPU timing instrumentation, so their absolute loss is not interchangeable
with the single uninstrumented 3.218% result.

| Work enabled during serving | Output tokens/s | Paired baseline | Loss | Range |
| --- | ---: | ---: | ---: | ---: |
| Hidden GPU staging, token/mask bookkeeping, L2 retention; writer held | 574.80 | 582.30 | 1.287% | 0.787-1.670% |
| Above, plus hidden D2H completion and L2 materialization | 571.99 | 584.67 | 2.168% | 1.970-2.281% |
| Full publication, retirement and lease refill | 561.64 | 584.66 | 3.937% | 3.843-4.031% |

Subtracting the paired extra serving durations gives the following approximate
increments per 128 requests. These are stage-control differences, not additive
kernel timings or exact attribution of the earlier 3.218% point.

| Increment | Extra duration | Fraction of diagnostic total |
| --- | ---: | ---: |
| Foreground staging/bookkeeping and L2 retention | 91.7 ms | 31.9% |
| Parent hidden D2H/completion and L2 materialization | 63.6 ms | 22.1% |
| Publication, retirement and lease refill | 131.8 ms | 45.9% |

All 1,152 active samples in these controls were published and independently
read back. Held stages perform the remaining work after the timed client exits;
they are diagnostic controls and are not a sustained-throughput serving mode.

## CPU Evidence

Exclusive CPU time in the full-publication stage, averaged over three runs,
with neighboring paused controls subtracted where applicable:

| Scope | CPU time per 128 requests |
| --- | ---: |
| `wait_for_copies`, excluding nested L2 materialization | 125.1 ms |
| `hidden_enqueue` | 73.1 ms |
| `after_forward`, excluding enqueue | 71.2 ms |
| `_reserve` | 58.1 ms |
| L2 materialization | 42.2 ms |
| `_write_snapshot` in the parent, including IPC/callback handling | 38.9 ms |
| `_manifest_fits` | 31.5 ms |
| `acquire_transient` | 29.7 ms |

CPU sums do not equal serving latency: much of this work overlaps GPU execution,
and some runs on other threads/processes. In particular, `wait_for_copies` CPU
time must not be reported as PCIe transfer duration. These timer windows also
include client startup and reservation maintenance outside the timed HTTP
workload; they identify operations, not exact critical-path contributions.

Moving construction and publication to a child does not remove all parent work.
The parent still submits the shared slot, handles publication checkpoints,
retires the slot and replenishes Catalog reservations. `begin` and `write`
share one ordered IPC channel. These remaining parent Python operations can
interact with the scheduler; the child itself does not share the parent's GIL.

## Completion Wait and Retirement Controls

A second sweep kept construction, validation and real publication enabled.
`published` held completed slots until the client finished, while `concurrent`
retired/refilled them normally. `blocking_copy_events` kept normal retirement
but created the hidden D2H completion event with `blocking=True`.

Two repetitions captured and read back another 768 samples. The second round
contained an anomalous paused baseline of 539.45 tokens/s, compared with adjacent
controls near 584. Both affected pairs have approximately 8% control spread.
The analysis preserves all results and flags control pairs with more than 3%
spread; their averages must not be used for precise causal attribution.

The first round's controls were stable:

| Mode | Throughput | Paired loss |
| --- | ---: | ---: |
| Publication completed, retirement/refill deferred | 564.24 | 2.524% |
| Full normal path | 562.87 | 3.433% |
| Full path with blocking D2H completion event | 563.32 | 3.603% |

The retirement control suggests that parent-side retirement/refill contributes
within the publication group. One stable round is insufficient to assign it
a precise share or to attribute the entire group to a particular IPC operation.

The completion-wait CPU result reproduced in both rounds:

| Operation | Normal event CPU | Blocking event CPU |
| --- | ---: | ---: |
| Hidden D2H event synchronize, per 128 requests | 102.56 ms | 1.98 ms |
| Same operation's wall time | 105.12 ms | 106.35 ms |

Blocking removes **98.1% of the CPU time spent waiting**, but does not shorten
the GPU completion dependency. No consistent serving throughput improvement
was demonstrated. The actual D2H submission/stream bookkeeping consumed about
17 ms CPU and L2 event synchronization about 1.3 ms CPU per phase. These are
CPU timings, not measured device transfer durations.

The best-supported optimization target is the publication/parent-management
group: reduce per-request IPC/checkpoint handling and batch reservation/slot
management while preserving publication and buffer-lifetime guarantees.
Blocking completion waits are a CPU-efficiency improvement, not a demonstrated
fix for the serving throughput loss. No production optimization was applied.

## Integrity Checks

The stage sweep archived 16 distinct samples. All 512 response hidden rows,
when multiplied only by the tied BF16 embedding/LM-head weight, reproduced the
corresponding greedy response token (512/512, maximum logit gap 0). No transformer
prefill was performed for this check. All 16 corrupted-payload checks failed as
expected, and all 16 manifests were rejected by the legacy top-128 decoder.
This checks prediction alignment and transport integrity; it does not establish
bitwise equality of full logits across arbitrary GEMM shapes or hardware.
The focused sweep independently passed the same checks for another 16 distinct
samples and 512 hidden rows. Across the two attribution sweeps, all 1,920 captured
samples passed full Mooncake readback; including the initial smoke gives 2,048.

## Artifacts and Reproduction

Before preparing the incremental fork PR, the proposed tree passed 171 focused
capture runtime tests and 235 subtests in 270.26 seconds on the same H100. The
HiCache page-first write-back suite passed all 16 CUDA tests in 62.20 seconds,
including contiguous runs, gaps and staging boundaries. Scoped Ruff F/E9 checks
and `git diff --check` also passed. JUnit reports are retained locally under
`/gpfs/user/fuxuanwei/sglang/mooncake-study/experiments/hidden-pr-validation-20261013`.

Artifacts are under
`/gpfs/user/fuxuanwei/sglang/mooncake-study/experiments/hidden-capture-20261013`:

- `smoke/report.json`: initial uninstrumented measurement.
- `attribution/report.json`: all 21 stage/control phases and CPU counters.
- `attribution/hidden-validation.json`: independent LM-head alignment checks.
- `attribution-source.tar.gz`: exact experiment entry points used for that sweep.
- `focused/report.json`: retirement and blocking-event controls, including the
  anomalous paused baseline.
- `focused/hidden-validation.json`: the additional 512-row LM-head check.
- `focused-source.tar.gz`: exact focused-sweep experiment entry points.
- `summary.json`: paired-baseline calculations.
- `environment.json`: runtime package versions and hardware assignment.

Entry points are `benchmark_hidden_capture.py`, `analyze_hidden_capture.py`,
`validate_hidden_capture.py`, and the test-only modules
`sglang.test.training_capture_hidden_{server,codec}`.

The pod uses the existing `capture` Python environment, with
`PYTHONPATH=/tmp/current/python`, `OMP_NUM_THREADS=1`,
`PYTHONDONTWRITEBYTECODE=1`, and `SGLANG_CACHE_DIR=/tmp/sglang-cache`.
The environment's FlashInfer Python/cubin packages have different versions,
so it also requires `FLASHINFER_DISABLE_VERSION_CHECK=1`; attention is Triton,
and the actual serving/readback runs completed under this environment.

```bash
python mooncake-study/experiments/benchmark_hidden_capture.py \
  --feature hidden \
  --config-report mooncake-study/experiments/hidden-capture-qwen35-4b.json \
  --output-dir /tmp/hidden-attribution --repeats 3 \
  --modes none copies concurrent --interleave-paused \
  --snapshot-writer-backend process --device-mib 128 --cpu-timings

python mooncake-study/experiments/benchmark_hidden_capture.py \
  --feature hidden \
  --config-report mooncake-study/experiments/hidden-capture-qwen35-4b.json \
  --output-dir /tmp/hidden-focused --repeats 2 \
  --modes published concurrent blocking_copy_events --interleave-paused \
  --snapshot-writer-backend process --device-mib 128 --cpu-timings
```
