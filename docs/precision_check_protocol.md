# Precision sensitivity check: prospective protocol (draft, 2026-10-02)

**Status: DEFERRED DRAFT; not pre-registered or executed. No local launch is planned.**

The user requires Docker to remain available for work at any time. Do not stop containers, change
Docker/WSL memory settings or load the experimental model on the working laptop. The original local
candidate is deferred. A separate compute resource and its budget, hardware, placement and software
must be specified before any future protocol is frozen; none has been selected or provisioned.

This proposes one future higher-precision comparison against the existing Phase 3B run. It is a
configuration-sensitivity study on already observed prompts, not a new held-out validation. It does not
replace the planned general-mutation work or silently redefine that work as Phase 3C.

## Question and fixed evidence

Does explicit acceptance of a false count comparison persist when NF4 weight quantization is removed,
using the same model revision, prompts and decoding settings on a separately specified resource?
A change of hardware adds a confound and must be reported; it does not isolate quantization alone.

- Phase 3B preregistration: `phase3b-preregistered` / `ed64ba8cb382073d652b24e647a898c94f8ce186`.
- Completed post-run audit commit: `a36647930aa8832613a3b080ccde8c09ab0ee111` (`a366479`).
- Existing baseline: `results/2026-10-02_111432`; 24 false-premise prompts plus 6 neutral controls.
  Original frozen result: **16 PASS / 6 POTENTIAL_FAILURE / 2 DETECTED_FAILURE**, controls **6 PASS**.
- Canonical baseline `results.jsonl` SHA256:
  `526ed3346649ca457e804a78aaa04d780e029af3b50da6f8b6b4be03891e467a`.
- Dataset `data/test_cases/false_premise_heldout.jsonl` SHA256 at audit:
  `62f1b7434a450da17ab3e64a09986d0ec555d1aa86981236baa27c606ee4d459`.
- Other original hashes, frozen source hashes and semantic annotations remain in the
  [Phase 3B audit](audits/phase3b-2026-10-02_111432/human_semantic_audit.md),
  [integrity record](audits/phase3b-2026-10-02_111432/integrity.json) and
  [verification record](audits/phase3b-2026-10-02_111432/verification.json).
  The verification record describes the audit-time HEAD before its subsequent documentation commit.

## Proposed comparison

| Setting | Existing baseline | Proposed new condition |
|---|---|---|
| Model | `swiss-ai/Apertus-v1.5-8B` | Same |
| Resolved model revision | `a411d838600baf0e3635a3daf66fb7c55fc97bb6` | Same |
| Weight quantization | 4-bit NF4, double quantization | `none` |
| Requested dtype | `bfloat16` | `bfloat16` (unquantized BF16, not FP32) |
| Execution | GPU + CPU offload on working laptop | Dedicated resource; hardware and placement TBD before preregistration |
| Memory caps | GPU 7 GiB, reserve 1 GiB, CPU 24 GiB | TBD for the dedicated resource before preregistration |
| Input | All 30 stored prompts, dataset order | Exact same prompts and order; no selected-case subset |
| Generation | Seed 42, greedy, thinking off, max 96 new tokens | Same |
| Evaluator | `factual_grounding_v1.3`; controls additionally `neutral_control_v1` | Same frozen code and rules |
| Calls | Already completed; no new baseline calls | One generation per prompt, 30 planned calls |

Retain the existing tokenizer/chat template and installed inference stack: baseline metadata records
PyTorch `2.11.0+cu128`, transformers `5.14.0.dev0` from Swiss AI fork commit
`3797303dda74844e3d1f8977ff5518bb91f818b4`, accelerate `1.15.0`, bitsandbytes `0.50.2` and tokenizers `0.22.2`.
Check these versions before freezing the protocol. Record any unavoidable environmental difference as a
protocol amendment before inference. Do not update dependencies, edit prompts, change the evaluator or
fall back silently to 8-bit. Do not regenerate or rescore the original baseline.

## Resource assessment and pre-run gate

Read-only observations on 2026-10-02, approximately 11:42 Europe/Riga (08:42 UTC):

| Observation | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| Reported VRAM | 8,151 MiB total; 7,068 MiB free at the preceding probe |
| Physical RAM | 31.297 GiB total; 9.659 GiB available |
| Cached pinned snapshot | Present; all 6 indexed safetensors shards present |
| Index `metadata.total_size` | 18,396,023,436 bytes, approximately 17.13 GiB |

These probes read the OS, GPU and cached index only; they did not load model weights, build a model or
generate responses. The index size is stored tensor data, not a measured peak-memory requirement.
Roughly 17 GiB of unquantized tensors cannot all reside inside the 7 GiB GPU cap; reserving GPU runtime
space suggests roughly 11 GiB or more must remain in RAM, before loading and runtime overhead. The
currently available 9.659 GiB is insufficient for a comfortable local attempt. The CPU cap of 24 GiB is
a planner limit, not currently available RAM. Local feasibility and runtime remain unverified.

The following resource gate is retained as planning guidance for a separately selected host, not an
instruction to free memory on the working laptop. Before future inference on that host, recheck free
RAM/VRAM and use a no-weights placement estimate for the exact
unquantized configuration. Require estimated CPU-resident weights plus at least 4 GiB of free-RAM
headroom, and GPU capacity for the planner's resident weights, streamed blocks and runtime reserve.
The 4 GiB margin is a conservative planning choice, not a measured loading requirement or guarantee.
Inspect loading overhead separately; do not treat a weight-only estimate as proof the run will fit.
If the gate fails, stop and amend the resource plan before running; do not start an opportunistic trial,
close user applications, switch hardware, enable disk offload or reduce the suite automatically.

Before proceeding, finalize this draft and freeze it in a **new, separate protocol commit**. Record that
commit in the future run metadata; do not move `phase3b-preregistered`. Actual model execution remains a
separate next action after the protocol and resource gate have been reviewed. No launch command is issued
by this document.

## Execution and reporting rules for the future run

1. Recheck the baseline's four hashes and the frozen dataset/evaluator. Use the unchanged runner to score
   the new responses once at generation time, in a new run directory. Preserve all prompts and responses,
   timestamps, model revision, versions, actual device map, offload plan, errors and truncation flags.
2. Plan one complete 30-prompt attempt. If loading or inference fails, retain the attempt's outputs and
   diagnostics, report it as incomplete, and do not silently retry, resume only selected cases or pool
   partial attempts. Any further attempt requires a recorded protocol amendment.
3. Hash all new raw files before semantic inspection. Keep original and new directories immutable and
   report their run IDs/configurations separately. Missing/error records are not PASS or semantic corrections.
4. Primary analysis: paired original-versus-new frozen status and premise-outcome tables for all 24
   false-premise prompts; counts by A/B variant, false more/fewer direction and scenario; 6 controls
   separately. Report exact IDs and transitions, including unchanged outcomes, errors and truncations.
   No aggregate rate, significance test, confidence interval or threshold for declaring the model reliable.
5. Secondary analysis: review all 30 new responses with the already documented semantic rubric, separate
   from frozen results. Do not inspect only HO-006-B, HO-008-B or HO-009-A/B. Preserve verbatim evidence and
   provenance; AI-assisted review must not be presented as independent human adjudication. Keep a truncated
   response flagged and interpret its observed text without claiming what its continuation would have been.

## Interpretation fixed before the proposed run

An explicit false comparison in the BF16 condition would show that NF4 quantization is not necessary for
that observed failure in this setup. Absence of frozen DETECTED cases is not proof of semantic correctness;
the separate full-response review remains necessary. If all false comparisons disappear, report a
configuration difference, not proof that quantization caused every baseline failure.

Removing quantization also changes device placement and numerical execution. This is therefore a
comparison of two inference configurations, not a clean isolation of bit width. One greedy generation per
condition, paired prompts from 12 synthetic scenarios, already known outputs and possible hardware changes cannot
establish a general failure rate, statistical independence, full-GPU equivalence or repeat-run stability.
The original Phase 3B result and audit remain valid historical observations regardless of the outcome.
