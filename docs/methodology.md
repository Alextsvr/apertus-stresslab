# Methodology (draft)

Status: Phase 1. Only the recording pipeline exists; evaluation methods below are planned, not implemented.

## Principles

1. **Evidence over scores.** A finding is an observable response, the exact prompt that produced it, and the reason it is wrong.
2. **Reproducible.** Every record stores model id, resolved Hub revision, dtype/quantization, generation config, seed, library versions, hardware and git commit.
3. **Replayable.** Prompts are built from test cases by a fixed template (`config.GROUNDED_PROMPT_TEMPLATE`); the stored `base_prompt` is the exact text sent to the chat template.
4. **Auditable evaluation.** Deterministic checks first (value presence, unsupported numbers/entities, polarity, premise correction). Any LLM judge would be optional, local and open-source.
5. **Honest reporting.** Negative results are reported. Automated detection is labelled DETECTED vs POTENTIAL; nothing is presented as a perfect hallucination detector.

## Failure categories

- **Factual grounding**: claims not supported by, or contradicting, the supplied context.
- **Consistency**: materially different conclusions for semantically equivalent prompts (style differences are not failures).
- **Robustness**: answers that change under task-preserving mutations (irrelevant context, reordering, typos, distractors), or that accept a false premise.

## Current pipeline (Phase 1)

```
test case (JSONL) -> build_prompt -> adapter.generate(prompt, gen_config, seed)
                  -> TestResult (status UNSCORED) -> results/<run_id>/results.jsonl
run-level info    -> results/<run_id>/metadata.json
```

## Planned

- Phase 2: grounding suite (~20 cases) + deterministic checks.
- Phase 3: deterministic mutation engine with stored mutation metadata.
- Phase 4–5: consistency/robustness suites, repeated runs with recorded seeds and failure rates.
- Phase 6: explicit, config-defined severity rules (see `docs/severity.md`, to be written).
