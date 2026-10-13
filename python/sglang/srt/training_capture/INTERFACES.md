# Training Capture Interface Contract

Status: interface contract with partial implementation. The baseline analysis
describes the SGLang PR at `81c9506b6` and its SpecForge consumer at `5ef53e1`.
The progress notes below distinguish implemented boundaries from migration
targets; proposed types and signatures are not all available APIs yet.

Implementation progress: `codec.py` now provides immutable `TeacherSpec` and
explicit `SnapshotCodec` selection. Hidden and top-128 use layout-driven Host
allocation, and the hidden runtime no longer depends on global replacements.
The existing batcher adapter exposes `append`/`seal`; workers retain staging and
`TeacherCompletion` receipts instead of request contexts. `SealedCapture` and
`HostReadyCapture` separate semantic completion from Host readiness, with slot
epochs checked at the process boundary. Auxiliary tensor filling runs after
copy completion. Controller extraction, the complete `CapturePlan`/`BufferLease`
API, and typed control replies below are still migration targets, not claims
about the current implementation.

The first implementation target is the existing single-rank autoregressive
HiCache L2 capture path. Final-hidden capture has an explicit model integration
and remains experimental within its documented model/dtype/topology limits. Existing
top-128, TP/PP, P/D, and speculative paths must retain their behavior; sharing an
interface does not qualify a new combination as supported.

## 1. Component Responsibilities

| Component | Owns | Does not own |
| --- | --- | --- |
| SGLang capture | Observing the accepted target path; teacher capture; KV lifetime; bounded staging; completion fences | Training progress, consumer assignment, long-term retention |
| Snapshot publisher | Validating an immutable sample; Store writes; publication journal; recovery | Serving requests, CUDA, token acceptance |
| Mooncake Store | Registered tensor transport and object storage according to the selected backend | Token semantics, producer state, exactly-once training consumption |
| Capture Catalog | Fenced producer leases, object registration, atomic publication visibility, retention coordination | Accessing SGLang tensors or deciding whether a CUDA copy completed |
| SpecForge | Validating and reading published samples; reconstructing teacher targets; training and checkpoint progress | Serving buffer ownership or declaring a producer sample complete |

The Catalog is an external control service used with Mooncake, not a feature
implicitly provided by every Mooncake Store deployment. A successful Store write
does not by itself establish disk durability. Archival and its durable receipt
remain a separate operation.

## 2. Baseline Implementation and Intended Boundaries

| Current code | Intended responsibility |
| --- | --- |
| `CaptureCoordinator` | A small serving facade plus a separate background capture controller |
| `RequestCaptureContext` | Request token ledger and semantic finalization; transfer mechanics move out |
| `TeacherLogitBatcher` and hidden test patches | Two implementations of a typed teacher-capture interface |
| `HiCacheL2KVExporter` | KV source protection and materialization, with explicit lease ownership |
| `HostBufferPool`, `HostSlot`, staging classes | Layout-driven buffers and independently releasable resource leases |
| `ProcessSnapshotWriter` | Bounded submission and typed completion/control messages |
| `SnapshotWriter`, codec, Store adapter | CPU publication pipeline, retaining existing transaction semantics |

At the baseline revision, the coordinator also initializes resources, renews
leases, prepares L2 sources, dispatches writes, handles publication callbacks,
and retires slots.
The request context also manages CUDA events and materialization. Buffer
allocation and teacher staging know the top-128 tensor names directly.

The baseline hidden experiment patches the logits processor, allocation, batcher,
request sealing, codec, and writer factory. The refactor replaces those patches
with explicit configuration and dependencies, rather than moving them into a
different production module unchanged.

These are logical boundaries, not a requirement for one new file per interface.
Reuse the existing package and its concrete implementations.

## 3. Execution Domains

| Domain | Allowed work | Forbidden work |
| --- | --- | --- |
| Serving scheduler / forward hooks | Bounded admission; map model rows; enqueue teacher copies; commit tokens; protect KV; enqueue completion | Network or disk I/O; payload hashing; CPU waits for CUDA or writer completion |
| Parent materializer | Wait for local completion fences; teacher D2H; gather L2 KV; fill aux tensors; submit Host-ready descriptors | Read a live `Req`; modify the accepted token ledger; publish incomplete data |
| Parent capture controller | Prefetch/renew Catalog leases; prepare admission credits; process cancellation and IPC replies; schedule resource release | Infer token acceptance; reuse buffers without completion evidence |
| CPU publisher process | Map shared buffers; validate/hash/encode; write Store; journal/seal/publish/recover | Initialize CUDA; dereference parent pointers; access scheduler objects |
| SpecForge workers | Fetch complete published samples and normalize them for training | Depend on the producer process remaining alive |

The teacher backend may use its existing parent-side worker and CUDA stream.
That worker owns only its transfer jobs and receipts, never mutable sessions.
This boundary does not require a new thread for every logical component.

Network latency, CUDA completion waits, and hashing must not extend a serving
callback. Short local synchronization and necessary device stream dependencies
still have a cost; this contract does not promise zero serving overhead.

## 4. Data Structures

All cross-domain identifiers are explicit. Except for the serving adapter,
interfaces must not accept a live scheduler `Req` or `ScheduleBatch`.

| Type | Fields and invariant |
| --- | --- |
| `CapturePlan` | Immutable `plan_id`, model/tokenizer/weights identity, `TeacherSpec`, existing `KVSpec`, schema ID, owner partition, `BufferLayout`, budgets, capability set. Built and validated once per model version. |
| `TeacherSpec` | Tagged union: `Top128Spec` or `FinalHiddenSpec`. Includes semantic stage, dtype, output transform and row alignment. Hidden dimension comes from a validated model binding, not a Qwen-specific constant. |
| `TensorLayout` | Name, dtype, capacity shape, byte offset, alignment, storage domain, owner. One definition drives allocation and serialization. |
| `SessionKey` | Capture/sample/generation identity and owner. Distinct from the Catalog fencing token and local buffer epoch. |
| `BufferRef` | Arena ID, slot ID, reuse epoch, layout ID. CPU shared-memory references use offsets, never parent virtual addresses. |
| `AdmissionCredit` | A valid fenced Catalog lease plus reserved device/Host/L2 budgets and downstream queue capacity. Consumed once by one session; stale or revoked credits cannot admit work. |
| `CaptureSession` | Scheduler-owned prompt/accepted-token ledger, pending row references, committed KV coverage, actual positions, stop reason and semantic state. No Store client, CUDA wait, or publication logic. |
| `BufferLease` | Reference, bounded capacity, current owner and outstanding readers/writers. The allocator is the sole authority that makes a slot reusable. |
| `SealedCapture` | Frozen session identity, plan, committed token ledger, teacher-row selection, KV validity and resource references. Logical content is final; bytes may still be in flight. Parent-local only. |
| `HostReadyCapture` | Bounded serializable descriptor containing identity, plan/layout/owner IDs, sequence lengths and stop reason, provenance, KV coverage, shared `BufferRef`, lease reference and deadline. All referenced Host bytes are complete and immutable. |
| `PublicationReceipt` | Catalog publication ID/cursor, sample generation, manifest key, size and digest. It says nothing about an unrelated disk archive or optimizer checkpoint. |

`plan_id` includes semantic and allocation choices and is checked across parent,
child and model replacement. Wire model identity remains the existing teacher
fingerprint. A slot epoch prevents a delayed ACK from releasing a newer sample
that happens to reuse the same slot number.

Large tensor bytes stay in preallocated storage. Frozen token ledgers and row
metadata have explicit token/byte limits; sealing may do bounded CPU work, not
an unbounded scan or metadata allocation.

`Top128Spec` fixes raw FP32 logits, global unpadded vocabulary IDs, top-k 128,
and full-vocabulary log-sum-exp at temperature 1. `FinalHiddenSpec` identifies
post-final-norm/pre-LM-head values, their dimension and dtype, and the exact
teacher/output transform needed by the reader. A codec's supported dtypes and
semantics are validated at startup; an internal union does not expand an
existing wire format's allowed values.

Plan construction also binds the model's required outputs before CUDA graph
capture. Disabled capture requests no extra model outputs. Unsupported model,
topology, dtype or generation mode combinations fail configuration validation;
they must not silently select another teacher variant.

## 5. Serving Facade and Session

Signatures below describe proposed Python contracts, not executable stubs.
Calls labeled non-waiting never wait for network, CUDA completion, or queue
capacity. They can return a capture-local rejection while serving continues.

```python
class CaptureRuntime:
    def try_open(self, request: RequestView) -> OpenResult: ...
    def on_forward(self, rows: ForwardCaptureRows) -> CaptureResult: ...
    def on_commit(self, path: AcceptedPath) -> CaptureResult: ...
    def on_kv_release(self, key: SessionKey, source: KVSourceView) -> CaptureResult: ...
    def finish(self, key: SessionKey, reason: StopReason) -> CaptureResult: ...
    def abort(self, key: SessionKey, reason: DropReason) -> None: ...
```

- `RequestView` contains only identity, prompt, generation limits and capture
  eligibility. `try_open` consumes a pre-reserved admission credit; it never
  calls `Catalog.begin`. `OpenResult` is `Admitted(handle)` or `Skipped(reason)`.
- `ForwardCaptureRows` binds session keys and prediction positions to model
  output rows, actual token positions, committed-prefix coverage, and a producer
  stream. Output tensors are borrowed only for this call. Mapping must survive
  batch reordering and overlap; background workers never retain the batch.
- `AcceptedPath` identifies the ordered tokens actually accepted by serving and
  their teacher row references. Observed/verified candidates are not accepted
  tokens. The speculative adapter supplies acceptance explicitly.
- `on_kv_release` protects the required source before serving reuses its pages.
  Its adapter runs at the appropriate before/after-cache hook. `finish` closes
  the semantic ledger but cannot make a sample eligible until this protection
  succeeds. Both events are explicit; call order is not inferred from a boolean.
- The hook completing both semantic finalization and KV protection emits
  exactly one `SealedCapture`. Neither hook waits for the other, D2H, or
  publication. `CaptureResult` is `Recorded`,
  `Ignored`, or `Dropped(reason)`; an invariant failure invalidates the sample.
- All session mutations are serialized by the serving adapter. Forward and
  result streams may differ; completion ordering is explicit. Workers return
  immutable receipts instead of changing `teacher_rows` on a live session.

The first implementation retains existing scheduler hook positions. This is
not permission to move KV retention until after device slots have been freed.

## 6. Teacher Capture Interface

```python
class TeacherCapture:
    def layout(self, spec: TeacherSpec, limits: CaptureLimits) -> TeacherLayout: ...
    def append(self, rows: ForwardCaptureRows, targets: RowTargets) -> AppendResult: ...
    def seal(self, selection: AcceptedRows) -> TeacherTicket: ...
    def cancel(self, work: TeacherWork) -> None: ...
```

`layout` runs at initialization. `append` and `seal` enqueue work without waiting
for its completion. `AppendResult` is `Accepted(row_refs)` or
`Rejected(reason, drain_ticket)`; partial enqueue failures must retain ownership
of any already submitted work. `TeacherTicket` identifies a fixed set of rows,
their output buffers, and a local completion fence, not a published sample.

`TeacherWork` is either pre-seal row references or a sealed `TeacherTicket`.
`cancel` suppresses future sample use; it does not cancel CUDA operations or
free buffers still referenced by them. Both forms enter the same drain and
resource-release lifecycle.

| Backend | Per-forward action | Completed teacher payload |
| --- | --- | --- |
| `Top128Capture` | Own selected raw logit rows in a bounded ring; compact on its worker/stream | Raw top-128 logits, global vocab IDs, full-vocabulary log-sum-exp |
| `FinalHiddenCapture` | Copy selected post-final-norm, pre-LM-head rows into bounded request-owned staging | Final hidden rows; no online top-k computation |

Both backends expose the same row-selection and completion contract. They do
not need the same internal ring layout. They never sample tokens, publish data,
or decide KV validity. The model binding explicitly exposes the required output
and its semantics; no global monkey patch or thread-local hidden side channel.

Capturing hidden still happens during forward. Delaying the D2H until request
completion does not allow the original model output view to be retained without
copying it. Copies must be ordered before that output's next overwrite,
including CUDA graph replay; allocator `record_stream` alone does not protect a
static graph output from being rewritten.

## 7. KV Source and Host Materialization

```python
class KVSnapshotSource:
    def retain(self, source: KVSourceView, coverage: KVCoverage) -> RetainResult: ...
    def materialize(self, lease: KVLease, dst: HostTensorViews) -> KVReceipt: ...
    def release(self, lease: KVLease) -> None: ...

class CaptureMaterializer:
    def prepare(self, sealed: SealedCapture) -> HostReadyCapture: ...
```

`retain` executes on the cache-owning serving thread before source reuse. It
returns a protected `KVLease`, or rejects the capture. It can enqueue an
offload and establish stream ordering, but cannot perform a host wait.
`materialize` executes only in the parent background domain and may wait for
the lease's completion events. `release` schedules an idempotent release on the
cache-owning thread after source readers finish.
If `retain` fails after enqueueing an offload, its rejected result includes a
drain ticket for the partial operation. Rejection is not permission to free
the operation's source or destination immediately.

The L2 implementation either retains existing protected HiCache pages or uses
bounded transient Host pages for the supported hybrid-model path. Protection
and pending D2H completion are separate conditions. The existing transient path
may call Host eviction to acquire capacity; the proposed policy must instead
reserve capture capacity or reject admission when its budget is unavailable.
This is a behavioral change to implement and measure, not a current guarantee.

`prepare` owns the sealed resources, waits for teacher producers, performs the
request-end teacher D2H, gathers selected L2 KV into shared Host slots, and fills
token/mask/position tensors from the immutable ledger. It publishes the
`HostReadyCapture` only after every writer to those bytes has completed. On
failure it reports the sample failure and drains or quarantines outstanding
work; it does not simply return buffers to the pool.

L2 gathering stays in the parent in the first refactor. Moving it into the CPU
child would require an explicit shared HiCache arena and a cross-process page
lease protocol; passing a Torch view or a pointer is not that protocol.

## 8. Resource and Queue Contract

```python
class CaptureBuffers:
    def try_acquire(self, plan_id: str) -> BufferLease | None: ...
    def transfer(self, lease: BufferLease, owner: ResourceOwner) -> None: ...
    def complete(self, ref: BufferRef, proof: CompletionProof) -> None: ...
    def quarantine(self, ref: BufferRef, reason: str) -> None: ...
```

`CompletionProof` is produced only by the owning transfer/cache/transport
adapter. A generic caller cannot set `done=True`. Every operation validates the
slot epoch; duplicate receipts are idempotent and stale receipts cannot affect
the current occupant.

| Resource | Earliest safe release |
| --- | --- |
| Raw-logit ring rows | Teacher compaction has stopped reading those rows |
| Request teacher GPU staging | All D2H reads of that staging are complete |
| Serving device KV slots used for transient offload | Required offload reads are ordered before reuse on every relevant stream |
| Protected L2 pages | Materialization has finished reading them, then cache-owner release runs |
| Shared Host payload slot | Child/Store no longer reads it and required recovery metadata is retained |
| Manifest storage | Its Store transfer no longer reads it and exact recovery bytes are retained |

The first implementation may retain a combined Host slot through publication
ACK to preserve current behavior. Early independent recycling is an optional
optimization after its completion proof is tested.

Admission reserves bounded downstream capacity for a sample, not just its GPU
rows. Budgets cover device bytes, Host bytes, protected/transient L2 pages,
sample count, metadata, and terminal completion messages. Jobs can exceed
their reservation only by invalidating that capture, not by waiting for more
capacity in a serving callback. An abandoned sample still consumes its credit
until its outstanding operations drain.

## 9. Controller, Publisher and IPC

```python
class CaptureController:
    def take_credit(self) -> AdmissionCredit | None: ...
    def cancel(self, key: SessionKey, reason: DropReason) -> None: ...
    def poll(self, max_events: int) -> tuple[ControlEvent, ...]: ...

class PublisherClient:
    def try_submit(self, sample: HostReadyCapture) -> SubmitResult: ...
    def poll(self, max_events: int) -> tuple[PublisherEvent, ...]: ...

class SnapshotPublisher:
    def publish(self, sample: HostReadyCapture) -> PublicationReceipt: ...
    def recover(self) -> tuple[RecoveryResult, ...]: ...
```

The controller exclusively owns Catalog producer lease begin/renew/fail and
the local cancellation registry. It prepares admission credits in the
background. The publisher owns object registration, writes, sealing,
publication and journal recovery. A parent control loop, separate from blocking
publication work, keeps lease renewal and replies moving during a slow write.

`try_submit` uses a bounded parent queue; it never writes to a potentially
blocking Pipe from a serving callback. On `Accepted`, the queue takes ownership
of the immutable Host slot; on `Full` or `Unavailable`, the materializer retains
it and applies its bounded retry/drop policy. IPC I/O runs on a background
transport worker. The first version can keep one in-flight write command.

Every message carries a protocol version, command ID, session identity and,
where relevant, slot epoch. Startup shares arena handles and validates layout
and plan IDs. Per-sample messages contain metadata and offsets, not tensor
payloads, live callbacks, CUDA events or arbitrary executable factories.

Typed messages replace the current callback channel without removing its
checks:

| Message | Purpose |
| --- | --- |
| `SubmitHostReady` | Transfer a complete immutable Host descriptor to the child |
| `CheckCurrent` / `AuthorizationReply` | Check the controller's generation, cancellation, deadline and lease view at the existing pre-write checkpoint |
| `Prepared` / `PreparedAck` | Retain the exact manifest bytes needed to reconcile an ambiguous seal or publish; ACK alone is not a disk durability receipt |
| `Published` | Report the verified Catalog receipt |
| `WriteFailed` | Report a definite failure and which resource completions are proven |
| `OutcomeUnknown` | Preserve recovery identity and quarantine resources whose transfer ownership is unknown |

The child keeps the durable journal before a potentially ambiguous seal/publish,
as today. The parent must not infer publication success from a buffer-release
ACK. Network retries reuse identical manifest bytes and idempotency keys.

Cancellation before submission drops the capture and drains its operations.
After submission, cancellation races with external publication: the controller
requests fenced Catalog failure/revocation and reconciles the result. If
publication already committed, the result remains published. A local cancel
flag alone cannot revoke an earlier authorization or retract a published sample.
Timeouts are `OutcomeUnknown` until reconciled, not proof of failure. Existing
Catalog operations and fencing remain authoritative; no new atomic revocation
guarantee is assumed.

## 10. Codec and SpecForge Boundary

```python
class SnapshotCodec:
    def layout(self, plan: CapturePlan) -> BufferLayout: ...
    def prepare(self, sample: HostReadyCapture, views: HostTensorViews) -> PreparedSnapshot: ...
    def validate(self, snapshot: PreparedSnapshot) -> None: ...
    def encode(self, snapshot: PreparedSnapshot) -> bytes: ...
```

The codec knows tensor names, semantic validation and manifest encoding; it does
not allocate serving KV, schedule CUDA work or publish Catalog state. It uses
`TeacherCapture.layout` plus KV/aux layouts as the single allocation definition.
Layout derivation is pure metadata work and cannot initialize CUDA or require
the model instance in the child.
Full payload validation and checksums stay in the child. Sharing computed
digests is allowed only while the underlying bytes remain immutable.

The first refactor preserves both existing wire formats exactly:

| Teacher variant | Contract ID | Payload format |
| --- | --- | --- |
| Top-128 | `maas-target-kv-top128-v1` | `maas_target_kv_v1` |
| Final hidden | `experimental-target-kv-final-hidden-v1` | `experimental_target_kv_final_hidden_v1` |

The internal name `prediction_positions` maps to the existing wire tensor
`logits_positions`; this is not a schema rename. Hidden manifests continue to
carry `logits: null`, `hidden_size`, `hidden_semantics` and
`teacher_final_hidden`. Unknown contracts are rejected rather than interpreted
as a familiar tensor layout. Preserve topology, model identity, codec, checksums
and accepted-token provenance.

For response token at sequence index `p`, the associated teacher row predicts
`token_ids[p]`; in ordinary AR it comes from the final hidden at `p - 1`. The
first response row comes from the final prompt position. `loss_mask` describes
prompt/response training selection, while `kv_valid` describes actual computed
KV: the final generated token may lack KV. These are independent masks.

SpecForge continues to use `SnapshotRefReader` and
`MooncakeSnapshotFeatureStore.get` with immutable publication metadata, followed
by local `release`. Hidden rows are projected through the matching frozen LM
head and output transform to recover top-128/LSE for CE + TV training without
target transformer prefill. Finite values, weights identity and row alignment
remain mandatory; wire compatibility does not assert bitwise GEMM equivalence.

The current publication-file reader and local feature-handle release are not a
distributed streaming consumer ACK protocol. Future Transfer Queue/Catalog
claims must own consumer leases, retries, rank assignment, checkpoint progress
and replay policy outside these producer interfaces. Reading or releasing a
sample does not delete it or establish exactly-once optimizer updates.

## 11. State Machines and Failure Rules

Semantic session state is separate from transfer and publication state:

```text
Session:      COLLECTING -> SEALED
                       \-> ABORTED
Materialize:  PENDING -> HOST_READY
                     \-> FAILED_DRAINING -> RELEASED or QUARANTINED
Publication:  QUEUED -> WRITING -> PREPARED -> PUBLISHED
                          \--------> FAILED or OUTCOME_UNKNOWN -> reconciliation
```

These are local state names. The current wire manifest uses `state=READY` and
the Catalog publish receipt uses `state=AVAILABLE`; they are not renamed.
Independent buffer leases track readers until actual completion, even after
the session becomes `ABORTED`.
After sealing, the session is immutable. Late cancellation is a controller and
publication event rather than an attempt to mutate that sealed session.

| Condition | Required behavior |
| --- | --- |
| No admission credit / ring capacity | Skip or invalidate capture; do not stall generation |
| Invalid teacher/KV/token alignment | Never publish; drain resources and count the reason |
| Partial CUDA submission or uncertain completion | Retain/quarantine referenced storage until safe completion or teardown |
| Writer exit / broken IPC | Stop admitting affected work; reconcile publication; do not assume transport readers stopped |
| Expired producer lease | Reject further publication under that fence and reconcile any ambiguous response |
| Model replacement | Drain or safely terminate all old-plan work before rebinding model identity |
| Shutdown | Stop admission, drain/cancel, resolve transfers, stop child/transport, then unregister/free storage |

The first distributed migration reuses existing owner partitions, per-owner
receipts and cohort assembly. A local `HostReadyCapture` from one TP/PP rank is
not a globally publishable sample. An owner adapter must assemble matching
receipts before global publication; the generic single-rank publisher must
reject a partial distributed sample.

## 12. Migration and Acceptance

1. Introduce explicit teacher specs, layout derivation and codec selection.
   Route the hidden experiment through these dependencies and remove its
   allocation/codec/batcher monkey patches. Keep the experimental wire ID.
2. Make session mutation single-owner. Move transfer state to tickets and
   resource leases, and move aux tensor filling into materialization.
3. Add `SealedCapture` / `HostReadyCapture` and typed IPC over the existing
   bounded process writer. Preserve check/prepared acknowledgments and recovery.
4. Isolate controller lease/refill work from serving hooks. Migrate existing
   distributed adapters without silently enabling new hidden capabilities.

Required checks when implementing: AR first/last-row alignment; stop/abort and
speculative trimming; overlap/CUDA-graph output lifetime; delayed completion
and stale-epoch ACK; ring/L2/Host exhaustion; child failure at each publication
stage; lost seal/publish reply recovery; legacy and hidden SpecForge readback;
training checkpoint resume. No new runtime tests are needed for this document.

Measure serving callback CPU time, D2D/D2H bytes and completion time, L2 gather,
queue residence, IPC/control work and writer stages separately. Compare TTFT,
TPOT and throughput against capture-disabled and admission-paused controls on
the same workload. Do not attribute cross-process slowdown to a shared GIL or
interpret summed overlapping stage durations as serving critical-path time.
