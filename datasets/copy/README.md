# Corrected Copy dataset v01

Canonical local data for copy_corrected_v01. train.jsonl comes from the old train split; test.jsonl comes from the old validation split; validation.jsonl is intentionally absent. Inputs are 1024 source tokens plus 1024 marker tokens (63). Labels are read from marker positions 1024..2047. Token IDs use identity integer encoding with vocab_size=64.

Raw JSONL files are ignored by Git; recreate them with: python scripts/materialize_copy_corrected.py
