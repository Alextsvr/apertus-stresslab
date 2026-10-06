# Prospective next-session preparation — 2026-10-06

This is a preparation plan, not a registered semantic experiment or evidence of model functionality. It is written before the new Colab session is executed. The completed diagnostics are published separately in `docs/colab_diagnostics/README.md`.

## Authorized first step

Use the public `notebooks/colab_evidence_and_setup.ipynb` in a fresh **free T4** session. Set `PREPARE_ENV=True` and leave `CACHE_MODEL=False`. Run its cells once, in order. No paid upgrade, local model loading, Docker changes, inference, evaluator invocation or historical experiment replay is part of preparation.

The notebook pins the published project checkpoint and the transformers fork, checks original Phase 3B and all eight diagnostic archives, creates a separate venv without `ensurepip`, and installs the minimal recorded package versions. Preparation checks Python 3.13.15, PyTorch 2.11.0+cu130, T4 compute capability 7.5, the six primary inference versions, the fork commit and frozen source/dataset hashes. It requires at least 13 GiB free CUDA memory, 8 GiB available system RAM and 22 GiB free disk before any future loading. The remaining inventory and GPU UUID/driver are recorded, not assumed equal to a previous session.

The bootstrap has local syntax/guard validation only. Its successful execution in a fresh Colab runtime is still pending. A successful preflight does not demonstrate GPU quantization functionality or semantic behavior: **no model is loaded and no forward is executed**.

## Preserve and inspect

On success, preserve `results/colab-preparation-preflight.json` using the notebook's final export cell. The exported ZIP contains the untouched JSON and an internal SHA256 manifest; its external SHA256 sidecar anchors the ZIP. Credentials, model weights and cache contents are excluded. The export prints `PREPARATION_EXPORT_OK` and downloads both files.

On a preparation/import/version/resource failure, stop and retain the cell output. Do not replace package pins, delete the environment or automatically repeat setup. An existing environment and existing preflight file are never silently overwritten. A change can be proposed after reviewing the failure.

Review the actual package inventory, Python/CUDA/driver, GPU class, source/dataset fingerprints and free resources before registering an inference protocol. A new physical T4 UUID is provenance, not by itself a rejection; other environment differences must be described in that new protocol.

## Boundary before inference

The intended next research question is whether the already tested NF4-weight / FP32-activation intervention produces assessable false-premise and neutral-control responses. Prompts, number/order of calls, decoding, numeric and output-quality stop gates, review criteria and exclusive output location must be committed in a separate prospective protocol before executing it. This document does not freeze those choices or authorize a generation cell.

Previously observed study prompts cannot be presented as new held-out validation. Any reused prompts must be labelled exploratory replication. New prompts need their own separate frozen artifact; the original dataset remains unchanged. Automated labels and any later AI-assisted or human annotations must remain separate, and AI-authored annotations must not be attributed to independent human review.

The original Phase 3B evaluator, dataset, raw results and `phase3b-preregistered` / `ed64ba8` remain unchanged. Preparation performs no rerun or rescore. Laptop work remains limited to documentation, integrity checks and CPU tests.
