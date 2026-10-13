# Asynchronous Teacher Ring

## Path

`teacher_capture_backend="async_ring"` uses the following ownership sequence for
ordinary colocated autoregressive capture:

```
serving stream: selected FP32 logits -> bounded GPU ring -> producer event
background stream: wait event -> batched top-128/LSE -> request GPU staging
request writer: wait completion -> one compact D2H -> Mooncake -> optional archive
```

The forward callback does not wait for compaction or Host transfer. A full ring
fails the affected capture sample and leaves generation running. Raw rows still
have to enter the ring in the producing forward because the serving logits
buffer is reused. The ring and request staging are charged to
`max_device_bytes`.

Add these fields to an existing capture configuration (increase the device
budget when also allocating KV staging or larger request pools):

```json
{
  "teacher_capture_backend": "async_ring",
  "teacher_gpu_ring_rows": 32,
  "teacher_background_batch_rows": 16,
  "max_device_bytes": 134217728
}
```

Async mode reserves `max_sample_tokens` compact rows per auxiliary-owner slot,
overriding `teacher_d2h_batch_tokens` to keep all rows on GPU until finalization.
Raw ring storage is `teacher_gpu_ring_rows * vocab_size * 4` bytes. Compact
teacher staging is `max_inflight_samples * max_sample_tokens * 1028` bytes.
Both are fixed at startup. Temporary top-k/LSE kernel allocations are additional
PyTorch workspace and are not included in that persistent staging budget.
Speculative and P/D capture reject this backend. The manifest and reader
contract stay compatible with existing Mooncake and SpecForge consumers.

## 2026-10-12 H100 smoke

The validation used one H100 80GB, Qwen3.5-4B TP1, HiCache L2 write-through,
three selected layers, 16 requests of 32+8 tokens, concurrency four, and a 100%
sample ratio. The ring held eight raw rows and the background batch limit was
four rows. Local TCP Mooncake and the HTTP test Catalog were used; disk archive
validation ran after client timing.

| Phase | Output tok/s |
| --- | ---: |
| Off before | 239.00 |
| Async capture | 208.55 |
| Off after | 242.58 |

Capture throughput is 86.61% of the 240.79 tok/s off-phase mean. Fourteen of 16
requests were admitted and all 14 reached READY, passed Mooncake readback, were
archived, and passed disk readback. The readback covered 196 payload objects,
6,881,280 KV bytes, 123,424 auxiliary bytes, and 7,004,704 total payload bytes.
The other two requests hit bounded capture admission while all serving requests
completed. This is a correctness smoke at an intentionally extreme capture
ratio, not a production SLO result. The low-priority job failed after the run,
so the phase summaries are the retained evidence for this invocation.

Focused validation passed 28 CPU tests plus 57 subtests, 90
coordinator/cohort/resource/startup tests, and two CUDA ring tests. The ownership
test delays the producer stream, proves deterministic full-ring rejection,
mutates the source logits after enqueue, and verifies exact compact teacher data
only becomes visible on Host after request completion. A reservation failure
also waits for the producer event before recycling the ring. The second test
injects kernel and completion-barrier failures and verifies that the ring stays
occupied, the request buffer is quarantined, and new enqueue attempts fail.

The final CUDA regression passed `2 passed, 9 deselected` in 25.31s on resident
H100 pod `job-0c9c7149b00f-20261012140459-master-0`. This allocation runs the
existing experiment worker with 60% idle load and pauses that load for tasks.
The original smoke predates the final failure-path guards and removal of the
background thread's process-wide `torch.set_num_threads` call. Its throughput
must not be presented as a performance result for those final changes. A longer
comparison with inline capture and normal sample ratios is still required.
