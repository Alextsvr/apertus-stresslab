"""Central defaults. Plain constants on purpose: easy to audit, easy to override from the CLI."""

from pathlib import Path

# Official Apertus 1.5 8B checkpoint (verified on Hugging Face, 2026-10-01).
# It is a gated repository: accept the license on the model page and authenticate
# with a free Hugging Face token (HF_TOKEN env var or `hf auth login`).
DEFAULT_MODEL_ID = "swiss-ai/Apertus-v1.5-8B"

# The model card requires a specific transformers fork commit (not mainline transformers).
TRANSFORMERS_FORK_COMMIT = "3797303dda74844e3d1f8977ff5518bb91f818b4"

# Bump whenever the test case files change in a way that affects comparability.
TEST_SUITE_VERSION = "0.1.0-smoke"  # kept for backwards compatibility (smoke suite)
SUITE_VERSIONS = {
    "smoke": TEST_SUITE_VERSION,
    "fg013_reproduction": "0.1.1",  # Phase 3A variants + controls; 0.1.1: structured `comparison` premise (prompts unchanged)
    "factual_grounding": "0.2.2",  # 0.2.1: assertion_patterns; 0.2.2: FG-013 structured `comparison` (prompts unchanged)
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_CASES_DIR = PROJECT_ROOT / "data" / "test_cases"

DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 256

# Default instruction for Phase 2 factual-grounding cases (each case stores it explicitly).
GROUNDING_INSTRUCTION = (
    "Answer using only the supplied context. If the question contains a false premise, correct it. "
    "Do not add information that is not in the context. Keep the answer brief."
)

# Template used when a case carries its own instruction (Phase 2+).
INSTRUCTED_PROMPT_TEMPLATE = "{instruction}\n\nContext:\n{context}\n\nQuestion:\n{question}"

# Simple, fixed prompt template for context-grounded questions (Phase 1 smoke cases).
# Kept explicit so that every stored prompt can be reproduced exactly.
GROUNDED_PROMPT_TEMPLATE = (
    "Answer the question using only the information in the context.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}"
)
