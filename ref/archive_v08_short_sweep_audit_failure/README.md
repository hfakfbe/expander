# v08 short sweep audit failure archive

This archive preserves the earlier v08 smoke/main outputs that were audited as insufficient for the v08 manual. The archived main runs used 80 steps for copy/selective_copy/niah/ruler/listops and 120 steps for induction_associative_recall, so they are kept for provenance only and must not enter the corrected Phase 6 main comparison. The archived smoke tree also contains one local CPU preflight rerun for copy/random_regular after the random budget fix; it is diagnostic only and is not part of the corrected Phase 5 gate.

The corrected run regenerates Phase 4 with full-train sweep budgets and per-query non-causal random_regular budget alignment.
