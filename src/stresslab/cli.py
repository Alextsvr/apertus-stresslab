"""Command-line interface.

    python -m stresslab env                      # print environment metadata (no model load)
    python -m stresslab infer --prompt "..."     # Phase 0: one real prompt -> one stored record
    python -m stresslab run --suite smoke        # run a JSONL suite from data/test_cases/ (unscored)
    python -m stresslab run --suite factual_grounding   # Phase 2: scored automatically
    python -m stresslab show results/<run_id> [--failures-only]
    python -m stresslab evaluate results/<run_id>       # re-score stored responses, no model call

Add `--adapter echo` to `infer`/`run` to test the pipeline without loading a model.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

from stresslab.cases import load_cases, suite_path
from stresslab.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SEED,
    SUITE_VERSIONS,
    TEST_SUITE_VERSION,
)
from stresslab.evaluators import EVALUATORS, evaluator_for
from stresslab.environment import collect_environment
from stresslab.models import build_adapter
from stresslab.offload import DEFAULT_CPU_MAX_MEMORY_GIB, DEFAULT_GPU_MAX_MEMORY_GIB, DEFAULT_GPU_RESERVE_GIB
from stresslab.runner import RunOutcome, rescore_run, run_cases, single_prompt_case
from stresslab.schemas import GenerationConfig, Status
from stresslab.storage import read_metadata, read_results


def _str2bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "on"}:
        return True
    if v in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--adapter", choices=["apertus", "echo"], default="apertus",
                   help="'apertus' = real model; 'echo' = fake model for pipeline checks only.")
    g.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    g.add_argument("--revision", default=None, help="Hub revision (branch, tag or commit sha). Default: main.")
    g.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    g.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none",
                   help="bitsandbytes quantization. Needed on GPUs with < ~20 GB VRAM (e.g. Colab T4).")
    g.add_argument("--cpu-offload", action="store_true",
                   help="Plan an explicit GPU + CPU-RAM device map (no disk). Modules that do not fit on the GPU "
                        "stay unquantized in system RAM; unused image/audio tokenizers are always parked on CPU.")
    g.add_argument("--gpu-max-memory-gib", type=float, default=DEFAULT_GPU_MAX_MEMORY_GIB,
                   help="GPU memory cap used by --cpu-offload (default: %(default)s).")
    g.add_argument("--gpu-reserve-gib", type=float, default=DEFAULT_GPU_RESERVE_GIB,
                   help="Part of the GPU cap kept free for activations/KV cache (default: %(default)s).")
    g.add_argument("--cpu-max-memory-gib", type=float, default=DEFAULT_CPU_MAX_MEMORY_GIB,
                   help="System RAM budget for offloaded weights (default: %(default)s).")

    g = p.add_argument_group("generation")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    g.add_argument("--do-sample", action="store_true", help="Sample instead of greedy decoding.")
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--top-p", type=float, default=None)
    g.add_argument("--enable-thinking", type=_str2bool, default=None,
                   help="Pass enable_thinking=true/false to the chat template. Default: template default.")

    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)


def _command_line() -> str:
    return shlex.join(["python", "-m", "stresslab", *sys.argv[1:]])


def _gen_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )


def _adapter(args: argparse.Namespace):
    return build_adapter(args.adapter, model_id=args.model_id, revision=args.revision,
                         dtype=args.dtype, quantization=args.quantization, cpu_offload=args.cpu_offload,
                         gpu_max_memory_gib=args.gpu_max_memory_gib, gpu_reserve_gib=args.gpu_reserve_gib,
                         cpu_max_memory_gib=args.cpu_max_memory_gib)


def _label(r) -> str:  # noqa: ANN001
    sev = f" {r.severity.value}" if r.severity else ""
    sub = f" ({r.subtype})" if r.subtype else ""
    return f"--- {r.test_id}{sub} [{r.status.value}{sev}] ---"


def _print_evidence(r) -> None:  # noqa: ANN001
    for e in r.evidence:
        print(f"  * {e['type']} [{e['confidence']}]: {e['detail']}"
              + (f" | expected {e['expected']}" if e.get("expected") is not None else "")
              + (f" | observed {e['observed']}" if e.get("observed") is not None else ""))


def _print_counts(results) -> None:  # noqa: ANN001
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status.value] = counts.get(r.status.value, 0) + 1
    print("Status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def _print_outcome(outcome: RunOutcome) -> None:
    for r in outcome.results:
        print(f"\n{_label(r)}")
        if r.error:
            print(f"ERROR: {r.error}")
        else:
            print(r.response)
            _print_evidence(r)
    print()
    _print_counts(outcome.results)
    print(f"Stored {len(outcome.results)} record(s) ({outcome.errors} error(s)) in: {outcome.run_dir}")


def cmd_env(_: argparse.Namespace) -> int:
    print(json.dumps(collect_environment(), indent=2))
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    case = single_prompt_case(args.prompt)
    outcome = run_cases([case], _adapter(args), _gen_config(args), args.seed, args.results_dir,
                        command=_command_line(), suite="adhoc")
    _print_outcome(outcome)
    return 1 if outcome.errors else 0


def _select_evaluator(choice: str, suite: str):  # noqa: ANN202
    if choice == "none":
        return None
    return evaluator_for(suite if choice == "auto" else choice)


def _load_validated(path: Path, evaluator) -> list:  # noqa: ANN001
    cases = load_cases(path)
    if evaluator is not None:
        for case in cases:
            evaluator.validate(case)
    return cases


def cmd_run(args: argparse.Namespace) -> int:
    path = Path(args.cases) if args.cases else suite_path(args.suite)
    evaluator = _select_evaluator(args.evaluator, path.stem)
    cases = _load_validated(path, evaluator)
    if args.only:
        wanted = set(args.only)
        cases = [c for c in cases if c.id in wanted]
        missing = wanted - {c.id for c in cases}
        if missing:
            raise ValueError(f"Unknown test id(s): {sorted(missing)}")
    if args.limit:
        cases = cases[: args.limit]
    outcome = run_cases(cases, _adapter(args), _gen_config(args), args.seed, args.results_dir,
                        command=_command_line(), suite=path.stem,
                        test_suite_version=SUITE_VERSIONS.get(path.stem, TEST_SUITE_VERSION),
                        evaluator=evaluator)
    _print_outcome(outcome)
    return 1 if outcome.errors else 0


def cmd_show(args: argparse.Namespace) -> int:
    meta = read_metadata(args.run_dir)
    model = meta.model.model_id if meta.model else "(model not loaded)"
    print(f"run {meta.run_id} | suite {meta.suite} | model {model} | seed {meta.seed} | cases {meta.num_cases}")
    if meta.notes:
        print(f"notes: {meta.notes}")
    results = read_results(args.run_dir)
    for r in results:
        if args.failures_only and r.status not in (Status.POTENTIAL_FAILURE, Status.DETECTED_FAILURE):
            continue
        print(f"\n{_label(r)}\nPROMPT:\n{r.base_prompt}\nRESPONSE:\n{r.error or r.response}")
        _print_evidence(r)
    print()
    _print_counts(results)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    meta = read_metadata(args.run_dir)
    suite = args.suite or meta.suite
    evaluator = _select_evaluator(args.evaluator, suite or "")
    if evaluator is None:
        raise ValueError(f"No evaluator for suite {suite!r}. Available: {sorted(EVALUATORS)}")
    path = Path(args.cases) if args.cases else suite_path(suite)
    out_dir, rescored = rescore_run(args.run_dir, _load_validated(path, evaluator), evaluator)
    for r in rescored:
        if r.status in (Status.POTENTIAL_FAILURE, Status.DETECTED_FAILURE):
            print(_label(r))
            _print_evidence(r)
    _print_counts(rescored)
    print(f"Rescored records written to: {out_dir} (original results.jsonl unchanged)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stresslab", description="Apertus StressLab")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("env", help="Print environment metadata (does not load the model).")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("infer", help="Send one prompt to the model and store the result.")
    p.add_argument("--prompt", required=True)
    _add_model_args(p)
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("run", help="Run a test suite and store the results.")
    p.add_argument("--suite", default="smoke", help="Suite name = file stem in data/test_cases/.")
    p.add_argument("--cases", default=None, help="Explicit path to a JSONL file (overrides --suite).")
    p.add_argument("--limit", type=int, default=None, help="Only run the first N cases.")
    p.add_argument("--only", nargs="+", default=None, metavar="TEST_ID", help="Only run these test ids.")
    p.add_argument("--evaluator", choices=["auto", "none", *sorted(EVALUATORS)], default="auto",
                   help="'auto' picks the evaluator from the suite name (smoke -> none).")
    _add_model_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("show", help="Print a stored run.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--failures-only", action="store_true", help="Only POTENTIAL_FAILURE / DETECTED_FAILURE.")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("evaluate", help="Re-score a stored run with the current evaluator (no model call).")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--suite", default=None, help="Default: the suite recorded in metadata.json.")
    p.add_argument("--cases", default=None, help="Explicit path to the case file used for the run.")
    p.add_argument("--evaluator", choices=["auto", *sorted(EVALUATORS)], default="auto")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Windows consoles may not be UTF-8; never crash on printing model output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
