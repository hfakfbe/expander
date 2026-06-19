# selective_copy corrected valid-as-test data

Version: probes_corrected_valid_as_test_l8_log5

This directory intentionally contains only train.jsonl and test.jsonl.
test.jsonl is byte-identical to the source validation.jsonl; the source test.jsonl is discarded.
The runtime contract forbids concatenating target labels to the input.
