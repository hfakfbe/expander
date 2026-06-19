# Corrected Copy v01 l8/log5 report

Variant: model layers=8, train log_every=5. Same corrected Copy data/graph seed/training budget as v01 except depth/log cadence.

## Results

| method | gate token acc | gate seq acc | gate loss | final token acc | final seq acc | final loss |
|---|---:|---:|---:|---:|---:|---:|
| dense | 1.000000000 | 1.000000000 | 0.004514607 | 1.000000000 | 1.000000000 | 0.004508784 |
| local | 0.015747070 | 0.000000000 | 4.127604276 | 0.016234375 | 0.000000000 | 4.127442790 |
| random_regular | 0.051147461 | 0.000000000 | 3.993629411 | 0.049651367 | 0.000000000 | 3.993598820 |
| zigzag_certified | 0.043884277 | 0.000000000 | 4.018539041 | 0.043608398 | 0.000000000 | 4.018767064 |

## Reachability (8 layers)

| method | target_in_Lhop_rate | unreachable_rate | histogram |
|---|---:|---:|---|
| dense | 1.0 | 0.0 | `{"1": 1024, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "unreachable": 0}` |
| local | 0.0 | 1.0 | `{"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "unreachable": 1024}` |
| random_regular | 1.0 | 0.0 | `{"1": 34, "2": 985, "3": 5, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "unreachable": 0}` |
| zigzag_certified | 1.0 | 0.0 | `{"1": 28, "2": 821, "3": 175, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "unreachable": 0}` |

Curve: `/Users/sxye/Documents/expander-copy-corrected-v01-l8-log5/outputs/copy_corrected_v01_l8_log5/figures/copy_l8_log5_curves.png`

Interpretation: dense solved; local remained structural negative control; random_regular and zigzag_certified improved slightly over marginal baseline but sequence accuracy remained 0 under the frozen 1-epoch budget.
