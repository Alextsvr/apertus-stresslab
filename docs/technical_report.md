# Apertus StressLab
## Reproducible Red-Teaming of Apertus 1.5

> **Draft.** Phase 2 methodology is implemented. All experimental sections are **PENDING**: no suite has been
> run against Apertus yet, and nothing here is a claim about the model.

### 1. Abstract
PENDING (after the first factual-grounding run).

### 2. Motivation
Red-teaming findings are most useful when another person can rerun them and see the same failure. StressLab
focuses on small, evidence-backed, replayable failures in grounding, consistency and robustness, using
deterministic and auditable checks rather than an LLM judge.

### 3. Threat / failure model
Phase 2 targets context-grounded question answering: unsupported claims despite complete context;
contradictions of supplied facts; merging of similar entities; mis-ordering of dated events; acceptance of
false premises; invented names, numbers or relationships. Later phases add equivalent-prompt consistency and
task-preserving mutations.

### 4. Test methodology
One model call per base case, fixed instruction, fixed prompt template, greedy decoding (seed 42,
`enable_thinking=false`, `max_new_tokens=128`). Details: `methodology.md`.

### 5. Dataset construction
22 original synthetic cases (suite version 0.2.0) across 10 subtypes: direct extraction, multi-fact, numeric,
relationship, negative fact, false premise, distractor, similar entity, timeline, unsupported elaboration.
All entities are fictional. Each case encodes machine-checkable expectations (required facts with accepted
values and known conflicts, forbidden values, allowed derived values, false-premise phrases) and a
`ground_truth` block. Data is original and redistributable under the repository license.

### 6. Mutation methodology
PENDING (Phase 3).

### 7. Evaluation methodology
Conservative deterministic evaluator `factual_grounding_v1`:
- DETECTED_FAILURE: controlled contradiction, forbidden value, or accepted false premise.
- POTENTIAL_FAILURE: missing required fact, unsupported number, heuristic unsupported entity, elaboration
  marker, unconfirmed/ambiguous premise correction.
- PASS: none of the above.
Severity is the maximum over evidence items (HIGH / MEDIUM / LOW; CRITICAL unused). Every status is backed by
structured `checks` and `evidence`. Rules and limitations: `methodology.md`.

### 8. Reproducibility methodology
Per record: model id, resolved revision, dtype, quantization, offload plan and device map, generation config,
seed, environment, git commit. Phase 2 uses one run per case; repeated-run statistics are PENDING (Phase 5).
Stored responses can be re-scored without inference (`stresslab evaluate`).

### 9. Experimental setup
- Model: `swiss-ai/Apertus-v1.5-8B`, revision `a411d838600baf0e3635a3daf66fb7c55fc97bb6`.
- Library: swiss-ai transformers fork (commit `3797303`), PyTorch 2.11.0+cu128, bitsandbytes 0.50.2.
- Hardware: NVIDIA GeForce RTX 5070 Laptop GPU (7.93 GB), AMD Ryzen 9 8945HX, 31.3 GB RAM, Windows 11.
- Precision: 4-bit NF4 (double quantization, bf16 compute) with explicit GPU + CPU offload (GPU cap 7 GiB;
  image/audio tokenizers and lm_head in system RAM). This is not full-precision, full-GPU inference.

### 10. Results
PENDING — no factual-grounding run yet.

### 11. Case studies
PENDING.

### 12. Limitations
- Small synthetic suite; no universal hallucination rate.
- Lexical evaluator: cannot understand semantics; entity detection is heuristic; derived numbers may be flagged.
- Quantized, CPU-offloaded baseline; single greedy run per case.
- Strong findings require reruns and, ideally, confirmation at higher precision.

### 13. Future work
Mutation engine, consistency and robustness suites, reproducibility reruns, cross-category severity,
dashboard.

### 14. Conclusion
PENDING.
