# FP32 activation probe attempt 2: GPU identity gate stop

Date: 2026-10-06. Prepared by Codex, AI-assisted technical audit.

The restored preceding replay archive passed its SHA256 and all internal manifest checks. The shared preflight check passed the pinned environment, T4/compute-capability and resource requirements. The attempt then stopped at equality of the raw `gpu_identity` field (GPU UUID plus driver version) against the preceding replay. No model was loaded or forward attempted. Core versions and full package inventory passed equality before this gate; later equality checks were not reached.

The saved failure does not contain the current UUID/driver, so it cannot distinguish UUID reassignment, driver change, or another raw-identity difference. It is not numerical evidence against the FP32 intervention. The assistant-authored cross-session identity requirement was unnecessarily restrictive for this exploratory free-Colab probe.

The archive SHA256 is `a0b21d6664df7bf4f225cac504517a7dd2e6489ac0dfe70e14b2db3f6c13157c`; sender checksum and all 11 internal hashes match. Protocol and launcher match `3ed0ae3b580e7c8e4e75f88b7337f43b154bf3a8`. This attempt and attempt 1 remain preserved unchanged.

A prospective third attempt will require the same T4 class and compute capability, retain core software/source/dataset checks, explicitly record prior/current UUID and driver differences, and save current preflight before cross-session comparisons. This adjustment applies only to the new exploratory short probe; it does not change the historical studies or their within-pair environment checks.
