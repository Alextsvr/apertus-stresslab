# Apertus StressLab
## Reproducible Red-Teaming of Apertus 1.5

> **Draft skeleton.** All experimental sections are PENDING. No results have been produced yet; nothing here is a claim about Apertus.

### 1. Abstract
PENDING.

### 2. Motivation
Red-teaming findings are most useful when another person can rerun them and see the same failure. StressLab focuses on small, evidence-backed, replayable failures in grounding, consistency and robustness.

### 3. Threat / failure model
Unsupported claims given supplied context; contradictory answers to equivalent prompts; answer changes under task-preserving prompt mutations; acceptance of false premises. (Details: `methodology.md`.)

### 4. Test methodology
PENDING (Phase 2+).

### 5. Dataset construction
Original synthetic cases only (fictional people, businesses, projects) to avoid licensing issues and memorized facts. Current: 4 smoke cases. Target: ~50 base cases.

### 6. Mutation methodology
PENDING (Phase 3). Template-based, deterministic, with stored metadata.

### 7. Evaluation methodology
PENDING (Phase 6). Deterministic checks; DETECTED vs POTENTIAL failure.

### 8. Reproducibility methodology
Per record: model id, resolved revision, dtype, quantization, generation config, seed, environment, git commit. Repeated-run statistics PENDING (Phase 5).

### 9. Experimental setup
Model: `swiss-ai/Apertus-v1.5-8B` via the swiss-ai transformers fork (commit `3797303`). Hardware: PENDING (local GPU or Colab T4, 8-bit).

### 10. Results
PENDING — no runs yet.

### 11. Case studies
PENDING.

### 12. Limitations
PENDING. Known so far: quantized inference on free hardware; automated checks cannot catch every hallucination.

### 13. Future work
PENDING.

### 14. Conclusion
PENDING.
