# v08 experiment/report/data archive, 2026-06-22

This directory archives the v08-era experiment artifacts and the corrected valid-as-test rerun artifacts before/while integrating the latest code into `main`.

## Snapshot files

- `snapshots/main_v08_premerge_local_snapshot.tar.gz`
  - Source: `/Users/sxye/Documents/expander` before the latest fast-forward merge.
  - Includes v08 configs, logs, outputs, reports, reference docs, existing v08 archive folders, local report documents/figures, and local demo/mask figure artifacts that were present in the main worktree.
- `snapshots/corrected_valid_as_test_l8_log5_snapshot.tar.gz`
  - Source: `/Users/sxye/Documents/expander-probes-corrected-valid-as-test-l8-log5`.
  - Includes corrected copy/probe configs, logs, outputs, summary CSVs, reports/figures, dataset metadata/materialized data, and corrected experiment scripts.

## Index and verification files

- `MANIFEST.sha256`: SHA256 checksums for both tarballs and index files.
- `ARCHIVE_SIZES.txt`: Size summary generated at archive time.
- `contents_main_v08_premerge_local_snapshot.txt`: File list inside the main v08 snapshot.
- `contents_corrected_valid_as_test_l8_log5_snapshot.txt`: File list inside the corrected valid-as-test snapshot.
- `main_v08_premerge_inputs.txt`: Top-level paths used to build the main v08 snapshot.
- `corrected_valid_as_test_l8_log5_inputs.txt`: Top-level paths used to build the corrected valid-as-test snapshot.

## Exclusions

The archive intentionally excludes model checkpoint/weight blobs and transient cache files:

- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`
- `*/checkpoints/*`
- `*/__pycache__/*`
- macOS sidecar files such as `.DS_Store` and `._*`

Checkpoint manifests, resolved configs, metrics, logs, figures, reports, summaries, graph artifacts, and dataset metadata/data files are retained.

## Verification

From this directory, run:

```bash
shasum -a 256 -c MANIFEST.sha256
```

To inspect a tarball without extracting it:

```bash
tar -tzf snapshots/main_v08_premerge_local_snapshot.tar.gz | less
tar -tzf snapshots/corrected_valid_as_test_l8_log5_snapshot.tar.gz | less
```
