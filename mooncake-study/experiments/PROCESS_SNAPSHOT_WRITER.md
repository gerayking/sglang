# Process Snapshot Writer

## Configuration

Add to an existing training capture configuration:

```json
{
  "kv_source": "hicache_l2",
  "snapshot_writer_backend": "process",
  "writer_process_timeout_seconds": 120.0
}
```

Keep HiCache enabled with write-through and one-token pages. The process backend
currently supports TP1/PP1, colocated autoregressive serving, with either inline
teacher capture or the asynchronous GPU ring. TP/PP, P/D and speculative modes
reject this backend during server argument validation. The default remains
`thread`; the snapshot schema and Mooncake/SpecForge reader interfaces do not
change. Device-source capture also works, but does not implement the L2 path.

## Ownership

```text
serving GPU -> HiCache L2 DRAM -> selected-layer CPU gather -+
teacher GPU staging -> request-end D2H --------------------+-> shared Host slot
                                                             |
                       metadata + slot index over IPC -------+
                                                             v
CPU publisher: construct + SHA-256 + validate + Store writes + publication
                                                             |
                 completion ACK -> serving may reuse slot <--+
```

The existing serving-side background thread waits for teacher copies and L2
readiness, then materializes L2 data into the snapshot slot. The scheduler does
not wait for the publisher. This change does not move GPU work or L2 data
preparation out of the serving process.

For transient L2 snapshots under overlap, capture remembers the KV producer
stream. At request release, the scheduling stream waits for that producer before
the controller enqueues its D2H copy. Previously only the scheduling stream was
fenced, so the final lookahead KV row could be copied before its forward finished.
The dependency uses CUDA stream events, with no new CPU synchronization or
per-token event allocation.

The bounded Host pool uses one Torch shared-memory arena. Serving pins that
mapping with `cudaHostRegister`; the child registers its own slot views with
its own Mooncake client. CPU tensor storage handles are transferred once at
startup through Torch's shared-storage reducer. Per-sample IPC carries only
metadata and a slot index, never KV or teacher tensor payloads. Provision
`/dev/shm` for the actual pool allocation, bounded by `max_host_bytes`; the
existing GPU staging budget is unchanged.

The publisher starts a fresh Python interpreter with CUDA hidden in its launch
environment. Setting the environment only inside a multiprocessing target is
too late: module imports can initialize CUDA before the target starts. The
serving process's environment is unchanged. The publisher uses one Torch CPU
thread, owns the Mooncake connection and publication journal, and is stopped
when its parent dies.

## Failure And Shutdown

- The existing bounded request queue and Host slots provide backpressure. One
  command is in flight over IPC; an occupied slot remains unavailable until a
  completion ACK. Full capture capacity affects admission, not generation.
- After remote validation, the child asks the parent to check cancellation,
  sealing and the current lease deadline before registering/writing objects.
  Catalog fencing remains authoritative throughout publication.
- After payload WRITTEN acknowledgements, exact prepared manifest bytes are
  sent to and acknowledged by the parent before journal/seal/publication. The
  existing prepared-publication recovery protocol uses these exact bytes.
- Store transport uncertainty quarantines the parent slot too. Worker exit or
  command timeout disables further capture, stops the child and quarantines the
  submitted slot. There is no automatic worker restart; restart the producer to
  recover durable journal entries. An ambiguous result never permits reuse.
- Shutdown stops producer activity and the background threads, synchronizes
  CUDA, stops the child/transport, and only then unregisters Host CUDA pins.
  Unconfirmed process or CUDA shutdown retains the owning resources.

`/server_info` capture state includes `snapshot_writer_backend` and
`snapshot_writer_pid`. Stage timings merge serving copy/queue waits with child
construction, validation and publication timings. The numbers are cumulative
wall time and do not measure GPU kernel duration.

## Validation

The CPU process tests cover shared-buffer visibility after startup, a CPU-only
child, cancellation after validation, malformed tensor rejection, ambiguous
publish replay, Store quarantine, worker death, timeout and journal-lock release.
CUDA tests exercise shared Host pinning, D2H visibility in the publisher and
unpinning after shutdown, plus a deliberately delayed forward before L2 copy.
The benchmark supports `--snapshot-writer-backends thread process` inside one
off/on/off bracket and reverses the variant order on alternating rounds. Keep
all schema checks, hashes and publication operations enabled in both modes.

Disk archives in this benchmark are produced by the readback consumer after
the measured serving phase. They are not a production periodic archive daemon
and do not turn Mooncake DRAM retention into automatic disk durability.

## H100 Result, 2026-10-12

The final, stream-fenced run used Qwen3.5-4B BF16, one H100 80GB, overlap and
decode CUDA graphs, HiCache L2 write-through/page size 1, selected layers
3/15/31, 32 input + 32 output tokens, concurrency 4, and 100% capture. The Host
pool had 128 slots (243,449,856 bytes), the teacher ring 32 rows, and the teacher
background batch limit 4 rows. Transport was local TCP with the HTTP test
Catalog. Each phase measured 128 requests; archive I/O and readback were outside
client timing.

| Mode | Output tok/s | Versus off mean | Median TTFT ms | p99 TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Off before | 575.54 | | 87.78 | 125.61 |
| Thread writer | 512.62 | -11.51% | 114.19 | 130.38 |
| Process writer | 546.56 | -5.65% | 95.54 | 124.21 |
| Off after | 583.04 | | 85.76 | 113.24 |

The off mean was 579.29 tok/s; off phases differed by 1.30%. Process throughput
was 6.62% higher than thread throughput. This is one final controlled bracket,
not a zero-overhead or production SLO claim. RDMA and longer production
workloads were not measured in this run.

Both capture phases admitted and published all 128 requests, with zero teacher
backpressure or failures. After producer shutdown, all 256 snapshots passed
Store readback, SHA/schema validation, archive write and disk readback. This
covers 3,584 payload objects and 210,010,112 payload bytes. Post-client writer
drain was about 9 ms in both modes.

Cross-mode comparison matched all 128 prompts. For the 127 requests with
identical complete token paths, all 1,778 payload-object hashes matched,
including the final KV row, token IDs, masks and teacher values. One generated
response differed starting at position 56 and was excluded from that bitwise
comparison; each of its snapshots still passed its own readback checks.

Earlier forward/reverse measurements are retained separately as diagnostic
evidence. They preceded the L2 producer-stream fence and exposed final-row
differences, so they are not pooled into the final result or treated as validated
training data.

The broader regression covered the process writer, configuration, coordinator,
resources, buffer registration, L2 export, cohort and publication paths, with
176 passing tests. Two PD subtests exposed a fixture declaring a one-token
limit while expecting two or three response tokens. After correcting the
fixture's request limits on both P and D, the PD suite passed 17 tests and 36
subtests. The final CUDA regression passed both shared pinned D2H/publication
and delayed-forward L2 ordering tests (2 passed in 22.08s).

Local evidence is under
`/gpfs/user/fuxuanwei/sglang/mooncake-study/experiments/process-writer-20261012/`:
`fenced-run/report.json`, the phase archives, `archive-comparison-fenced.json`,
`unit-tests-final.log`, `pd-regression-final.log`, and `cuda-regression-final.log`.
The report records producer source hashes and the full invocation configuration.
