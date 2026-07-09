# MoE wgrad in FlyDSL — an optimization walkthrough

A study log of how the permute-free MoE weight-gradient (wgrad) kernel evolved from
a naive first cut to something within striking distance of the Triton baseline on
AMD CDNA4 (gfx950). The emphasis is on **why** each version was slow and **what
concept** fixed it — not on the code itself.

## The problem

For each expert `e`, accumulate a weight gradient by contracting over the tokens
routed to that expert:

```
dW[e][n, k] = sum over routed slots s of expert e:  grad[s, n] * x[token(s), k]
```

It's a **grouped GEMM**: one `[N, K]` matmul per expert, where the shared
contraction dimension is "routed slots" (variable length per expert). "Permute-free"
means we never materialize a reordered copy of `x`/`grad` in global memory — instead a
`sorted_ids` array maps each contiguous slot back to its original token, and we
**gather** operands on the fly. That gather is the whole story of this kernel.

The contraction is over the slot axis, but both operands are stored as
`[slot, feature]` (feature contiguous). The matrix core wants the contraction axis
(slots) to be the *inner* MFMA dimension. So somewhere we must **transpose**
`[slot, feature] → [feature, slot]`. Where and how we do that transpose is the
second half of the story.

Baseline to beat: the Triton `transfree` kernel at roughly **490–560 TF/s**.

---

## v0 — direct gather straight to registers (no LDS)

**Idea.** The simplest thing that works: each lane computes the global addresses for
its operands, gathers them directly into registers, and feeds the matrix core.

**What was lacking.**
- The gathers are **uncoalesced strided loads**. Because the contraction axis (slots)
  is scattered through `sorted_ids`, adjacent lanes read addresses that are far apart.
  The memory system can't merge them into wide cache-line transactions, so effective
  bandwidth collapses.
- No reuse: every lane re-fetches operands that a neighbor also needs.

This established the correctness reference and made the bottleneck obvious: **memory,
not math.**

---

## v1 — register blocking (bigger tiles, amortized gathers)

**Idea.** Make each workgroup responsible for a larger output tile so a gathered
operand feeds more MFMA work before being discarded. Classic arithmetic-intensity
lever: reuse each loaded byte across more flops.

**Result.** ~**194 TF/s** at 64×64; but scaling the tile *up* to 128×128 made it
*worse* (~79 TF/s).

**What was lacking.** Register blocking raises reuse but does nothing about the
**uncoalesced** access pattern — the loads are still strided gathers. Worse, bigger
register tiles inflate VGPR pressure, which lowers occupancy, which removes the GPU's
main tool for hiding the (still enormous) gather latency. So the two effects fought
each other. We were ~2.6× behind Triton and stuck.

Lesson: **you can't out-block a coalescing problem.** The access pattern itself has to
change.

---

## v2 — stage through LDS + hardware transpose read

Two concepts arrive together here, and they're the core of the whole design.

**Concept 1 — coalesced fill via LDS.** Instead of gathering directly to registers,
first do a **coalesced** load of a `[slot, feature]` tile into LDS (shared memory).
The trick: lanes cooperate so that *along the contiguous feature axis* the loads are
wide and merged (each thread grabs an 8×bf16 = 16-byte vector, one
`global_load_dwordx4`). The expensive `sorted_ids` indirection is applied per *row*
(slot), but the bytes within a row come out contiguous. We convert a scattered gather
into a batch of wide, coalesced row-loads.

**Concept 2 — `ds_read_tr16_b64` hardware transpose.** CDNA has an LDS read that
transposes on the way out. We store the tile as `[slot, feature]` but read it back as
`[feature, slot]` directly into the MFMA fragment layout — no VGPR shuffles, no extra
passes. The transpose that v0/v1 would have paid for in register movement becomes
free, done by the LDS hardware.

So the pipeline per contraction step became:

```
coalesced global load  ->  LDS [slot, feature]
ds_read_tr16_b64       ->  MFMA fragment [feature, slot]
mfma                   ->  accumulate
```

**What was still lacking.** v2 as first written used a **single warp** per output
tile. One warp can't issue enough memory traffic or MFMA to saturate a CU, and there
was no latency hiding. Correct and much better structured, but throughput was modest.

---

## v2.1 — multi-warp cooperative workgroup

**Idea.** Put a `warps_n × warps_k` grid of warps on one `block_n × block_k` output
tile. **All warps cooperatively fill the shared LDS tile once per step** (amortizing
the gather across the whole team), then **each warp owns a sub-tile** of 16×16 MFMA
atoms and transpose-reads only its own feature columns out of the shared tile.

**Why the contraction step is 32.** The MFMA is `16×16×32`: its contraction (`K`)
dimension is 32. One step stages a `[32 slots, block_feature]` tile — exactly one MFMA
`K` chunk — and that single staged tile feeds *every* warp's atoms. The expensive
global gather is paid once per 32 slots and shared by the entire workgroup, instead of
each warp gathering for itself. That amortization is the core win of going multi-warp.

**Result.** ~**287 TF/s** best case — about **0.57×** Triton. Real progress, but a
plateau. Profiling pointed at the loop shape itself.

**What was lacking.** The loop was fully serial:

```
fill  ->  barrier  ->  mfma  ->  barrier  ->  (repeat)
```

The matrix cores sit idle during `fill` (waiting on global-load latency), and the
memory system sits idle during `mfma`. Nothing overlaps. This is a **software
pipelining** problem.

---

## v2.2 — async prefetch (2-stage register pipeline)

*(preserved as `moe_wgrad_flydsl_v2_copy2.py` for study)*

**Idea.** Hide the global-load latency behind compute by **prefetching the next step's
operands while the current step's MFMA runs.** Split the fill into two phases — a
`gather` (global → registers) and a `store` (registers → LDS) — so the loads can be
issued far ahead of when the data is needed in LDS. The prefetched tile is carried
across the loop iteration in **registers** (the loop's carried values):

```
regs = gather(step 0)              # prologue
for each step:
    store_lds(regs)                # write current tile to LDS
    barrier                        # (B1) LDS visible
    regs_next = gather(step+1)     # issue NEXT global loads -> in flight...
    mfma(from LDS)                 # ...overlapping this compute
    barrier                        # (B2) done reading LDS before overwrite
    regs = regs_next
```

"Software pipelining" here literally means *loop-carried registers*: the next tile
rides in VGPRs from one iteration to the next. It's cheap because each thread only
carries a vector or two per operand.

**Result.** ~**400 TF/s** — a ~1.4× jump, to **~0.77×** Triton.

**A telling side effect.** With latency now hidden by the pipeline, **fewer warps
started winning** (4-warp beat 8-warp). Before, we leaned on high occupancy (many
waves) to hide latency; once the pipeline does that job explicitly, extra waves mostly
add barrier contention. Two different tools for the same purpose — and you don't need
both cranked.

**What was still lacking.** One thing the prefetch does *not* hide: the
`store → barrier → compute` chain. With a **single** LDS buffer, we can't write the
next tile until every warp has finished reading the current one — so the LDS-write and
its barrier are still serialized against compute. The global latency is hidden; the
LDS-write latency is not.

---

## v2.3 — ping-pong (double-buffered LDS)

*(current version, `moe_wgrad_flydsl_v2.py`)*

**Idea.** Use **two** LDS buffers and alternate. While the matrix core reads buffer A,
write the prefetched next tile into buffer B. Because reads and writes hit *different*
buffers, they don't conflict — so we don't need a barrier between "write next" and
"compute current," and the loop collapses to **one barrier per step**:

```
prologue: gather(0), store -> buffer 0, barrier
for each step:
    cur = step % 2;  nxt = 1 - cur
    gather(step+1)                 # next global loads in flight
    mfma(read buffer[cur])         # compute current
    store(buffer[nxt])             # stage next into the OTHER buffer
    barrier                        # single barrier suffices
```

**Result.** ~**395–419 TF/s** — a further few percent, landing at **0.74–0.83×**
Triton depending on shape.

**The instructive bug.** The first attempt failed *only on multi-step experts* while
single-step cases passed. Cause: the LDS allocator reservation used `TILE * 2`, where
the `*2` was **bytes-per-bf16**, not double-buffering — so each operand had room for
*one* tile, and "buffer 1" silently overlapped the neighboring region. Single-step
experts never touched buffer 1, so they masked the bug. Fix: reserve `TILE * 2 * 2`.
Two lessons: (1) an off-by-a-factor in a size calculation hides until the second
buffer is actually used; (2) tests must include the multi-iteration case or the
double-buffer path is never exercised.

**What the small gain tells us.** Ping-pong only bought a few percent — meaning the
`store→barrier` serialization was *not* the dominant remaining cost. The big win was
already banked by hiding **global** latency (v2.2). This is itself a useful profiling
signal: it says the remaining gap is elsewhere.

---

## Profiling turn: ATT tells us exactly where the stalls are

At v2.3 we stopped guessing and captured an **ATT (Advanced Thread Trace)** of the
kernel *and* the Triton baseline on the same shape (DSV3-Down, B8 M4096, N7168 K2048),
then aggregated per-instruction `Stall` cycles by instruction class. The verdict was
unambiguous:

| stall class | FlyDSL v2.3 | Triton |
|---|---:|---:|
| **LDS reads** (`ds_read_tr`) | **26.0M (48%)** | 5.5M (19%) |
| waitcnt | 9.5M | 5.3M |
| LDS write | 7.2M | 3.5M |
| barrier | 4.9M | 7.1M |
| **global loads** | **0.4M (0.7%)** | 3.4M (12%) |
| TOTAL | 53.8M | 29.2M |

Two things jumped out. First, our **global-load stall was already tiny (0.7%)** — the
prefetch + ping-pong pipeline had done its job; we'd *won* the memory-latency battle.
Second, **LDS reads were 48% of all stalls** at ~117 cycles each vs Triton's ~20 —
the signature of heavy **bank conflicts** on the transpose read. This is why you
profile: the bottleneck was not where "more pipelining" would help at all.

## v2.4 — LDS padding (kill the bank conflicts)

**Concept.** LDS is 32 banks × 4 bytes. Our `[slot, feature]` tile had a row stride of
128 bf16 = 64 dwords, and `64 mod 32 == 0`, so every slot at a given feature mapped to
the *same bank*. The transpose read gathers a column (fixed feature, varying slot) →
all lanes hit one bank → serialized. Adding a few elements of **row padding** so the
stride isn't a multiple of 32 dwords makes consecutive slots step across banks.

Constraint: `ds_read_tr16_b64` is an 8-byte read, so the padded stride must stay
4-bf16-aligned → `LDS_PAD` must be a multiple of 4. Swept `{0,4,8,12,16}`.

**Result.** `LDS_PAD = 8` took the big shape from **506 → ~665 TF/s (+31%)**. The
follow-up ATT confirmed the mechanism: LDS-read stall collapsed **26.0M → 2.3M (~11×)**,
per-read latency **117 → 14 cycles** (now *faster* than Triton), and total stall fell to
**29.4M — on par with Triton's 29.2M**. The sweep was non-monotonic (`pad=12` worse than
8 and 16), a reminder that conflicts depend on the exact `stride mod 32`, not "more pad."

Note what padding is *not*: it emits no instructions and costs only a little LDS
capacity. It's a pure addressing change.

## v2.5 — instruction scheduling (setprio + register prefetch)

After padding, the dominant stall **moved** to `waitcnt` (16.7M, 57%) — the wave now
issues the cheap LDS read but then stalls at the `s_waitcnt` that gates the MFMA on the
read's result. The fix is scheduling, not algorithm: give the matrix core independent
work to chew on while a read is outstanding.

Two levers, and an instructive tradeoff between them:

- **`s_setprio(1)` around the MFMA block** tells the scheduler to prefer this wave's
  matrix instructions over round-robining to a stalled wave. Nearly free.
- **Register prefetch of the A fragment** (`read_a(mi+1)` issued before the MFMAs that
  consume `a[mi]`) overlaps the LDS-read latency with useful compute.

The tradeoff: **hoisting *all* A fragments up front** hid the wait best on small shapes
(+10%) but *regressed the big shape* (663 → 589) — because the extra live fragments
raise VGPR pressure, and the large tile-count shape depends on occupancy (many resident
waves) for its own latency hiding. ATT made this legible: per-wave stall actually *fell*,
but wall-clock rose because fewer waves were resident. The resolution was a **2-deep
prefetch** (hold only current + next A fragment): enough overlap to help, cheap enough in
registers to leave the big shape's occupancy intact.

Asymmetry worth remembering: only the *streamed* operand (A here) can be held current+next.
The other operand (B) is reused across the whole inner loop, so it must stay fully
resident — otherwise you'd re-read it `M_STEPS×` and add LDS traffic. Rule of thumb: keep
`min(M_STEPS, N_STEPS)` worth of the reused operand resident, stream the other.

**Result.** ~657 TF/s on the big shape (held, no regression) and +2–3% on the smaller
shapes — landing at **~0.84–0.90× Triton** across the sweep.

## v2.6 — sched_barrier: sink the mask/store below the MFMA

ATT on v2.5 (`att_v2_fresh`) showed the *next* idle resource: an `s_waitcnt vmcnt`
sitting **in front of** the MFMA block (5.7M + 536K cyc), so the matrix cores stalled on
the next tile's global load before doing any math. The MFMA operands come from LDS
(`ds_read_tr`, gated by `lgkmcnt`) and don't depend on that load at all — the wait was
there only because the compiler scheduled the fill's mask (`valid.select`) + `ds_write`
*before* the matrix ops, and the mask is what consumes the loaded regs (forcing `vmcnt`).

`s_setprio` can't fix this: it sets runtime wave priority, not the static instruction
order. The right tool is `rocdl.sched_barrier(0)` — a compile-time schedule fence
(the idiom the production GEMMs use: `splitk_hgemm`, `moe_gemm_2stage`, ...).

Two-part change, register-neutral:

- **Split the fill.** `gather_*` now only *issues* the global loads (returns raw regs +
  validity); the mask/scale + `ds_write` moved into `store_*`.
- **Fence after compute.** A `sched_barrier(0)` between the MFMA and `store_*` stops the
  compiler hoisting the mask/store (and its `vmcnt`) in front of the matrix ops.

ATT after (`att_v2_interleave`): the MFMA block is now preceded only by `lgkmcnt`
(LDS-read) waits; the global-load `vmcnt` (131K/206K/210K) moved *below* the MFMAs, onto
the `ds_write`. The matrix pipe fires while the next tile's loads are still in flight.

Unlike the v2.5 prefetch (which traded VGPRs for overlap and so was capped by occupancy),
this adds no live registers — so it helps every shape without an occupancy penalty.

**Result.** Consistent **+8–12%** across the sweep; the best config (`128x128 w4x2`) lands
at **~0.99–1.13× Triton** (matches/beats it on all three bench shapes). Note this
contradicts the earlier "bandwidth-bound, reordering can't help" read — there *was*
latency headroom the front-loaded `vmcnt` was hiding.

## v2.7 — slot-id prefetch (carry the indirect index two steps ahead)

With the data `vmcnt` moved below the MFMA (v2.6), ATT (`att_v2_interleave`) showed the new
dominant stall was a **`vmcnt` of 7.66M** — the *indirect* `sorted` index load. The gather is
two-hop: `load slot_id` → `vmcnt` → form data address → `load data`. Because the index and
the data address math live in the *same* pipeline stage, the index latency can't hide behind
the MFMA (moving its wait after the MFMA would just drag the dependent data load after it too).

Fix: give the index stream its **own extra pipeline stage**. `load_slot_ids` is split out of
the gather; the resolved ids are prefetched **two steps ahead** and carried across the loop as
`scf.ForOp` iter_args (data stays one step ahead). Each iteration issues the ids for step `s+2`
while consuming the ids for `s+1` that were loaded last iteration — so the ~500cyc index
latency overlaps a full MFMA step. Cost is a few `i32` per fill (cheap; not the VGPR-heavy
data prefetch of v2.5), so no occupancy hit.

ATT after (`att_v2_idxpf`): the 7.66M index `vmcnt` is gone (index loads now ~30–50K); the
remaining `vmcnt` are the small post-MFMA data-store waits. The new top stall is the
**`s_barrier` (2.14M)** — i.e. the kernel crossed from global-load-latency-bound to
workgroup-sync/occupancy-bound.

**Result.** A further **+3–20%** on top of v2.6 (best config shifted to `256x128 w4x2`),
landing at **~1.10–1.25× Triton** across the three bench shapes.

---

## The throughline

Every step attacked one specific idle resource — and once the obvious ones were gone,
**ATT profiling** pointed at the next one instead of guesswork:

| Version | Key idea | Idle resource removed | vs Triton |
|---|---|---|---:|
| v0 | direct gather | (baseline; memory-bound) | — |
| v1 | register blocking | some operand reuse | ~0.35× |
| v2 | LDS fill + hw transpose read | uncoalesced loads; VGPR transpose | — |
| v2.1 | multi-warp cooperative fill | per-warp redundant gather | ~0.57× |
| v2.2 | async register prefetch | **global-load latency** | ~0.77× |
| v2.3 | ping-pong LDS | LDS-write serialization | ~0.74–0.83× |
| v2.4 | LDS padding | **LDS-read bank conflicts** | (+31% big shape) |
| v2.5 | setprio + 2-deep A prefetch | `waitcnt` (read→MFMA) stall | ~0.84–0.90× |
| v2.6 | sched_barrier (mask/store after MFMA) | **`vmcnt` in front of MFMA** | ~0.99–1.13× |
| v2.7 | slot-id prefetch (index 2 steps ahead) | **indirect-gather index `vmcnt`** | ~1.10–1.25× |

The recurring pattern: **find the resource that's stalling, then restructure so
something useful happens during that stall.** Coalescing fixed raw bandwidth; the
hardware transpose removed a data-layout tax; cooperative fill amortized the gather;
prefetch and double-buffering overlapped the two halves of the loop that were taking
turns; padding un-serialized the LDS reads; scheduling kept the matrix pipe fed while
reads were in flight. And the pivot in the middle — **stop guessing, capture an ATT,
compare against the baseline class-by-class** — is what turned "probably bank conflicts"
into a measured +31%.
