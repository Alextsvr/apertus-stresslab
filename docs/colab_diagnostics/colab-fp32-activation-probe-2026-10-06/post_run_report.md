# FP32 activation probe: preflight stop

Date: 2026-10-06. Prepared by Codex, AI-assisted technical audit.

The dependency ZIP hash gate rejected the archive at the expected preceding-replay path in Colab. No model was loaded, no forward call or token selection occurred, and no activation dtype or memory-promotion check was reached. This is an unmet dependency-integrity prerequisite, not evidence that the FP32 intervention failed numerically.

The failed attempt ZIP has SHA256 `1fc0b20ba6df520028e8b0e3898a7087baf5122d850937b8b33a84f3f88136e1`; its supplied checksum and all 11 internal file hashes match. Protocol and launcher match commit `956b6d57c8a90787d9850f239dd5541eae38b53c`. All evidence is preserved unchanged.

The verified preceding replay ZIP on the laptop still has the registered SHA256 `ef98d1fe60a71adf4abe2082a9a4974d8ec8ba4eb97e04e596a1d6cd83b2ae23`. The launcher constant is correct. The failed log did not record the actual hash or explain why the Colab-side dependency differed, so that origin cannot be inferred.

A prospectively amended attempt will use a separately uploaded, exact-hash-verified copy of that preceding archive at a new dependency path, and an exclusive attempt-2 output directory. The old dependency path, failed attempt, scripts/protocol frozen in its ZIP, and research results will not be overwritten. No numerical result from the stopped attempt is discarded or treated as successful.
