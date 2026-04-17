# Parallel pipeline master summary — 20260415_102111

Datasets: cares cwlc radgraph mimic_s
Fractions: 5pct 25pct 50pct 100pct
Text sources: generated original
Stages: eval downstream
N_RUNS (seeds): 3
N_GEN: 1
GPUs used: 0 1 2
Wall clock: 15h12m

| Dataset | Fraction | TextSrc | Generate | Real tok/doc | Synth tok/doc | Downstream F1 |
|---------|----------|---------|----------|--------------|---------------|---------------|
| cares | 100pct | generated | ♻ reused | 224.8 | 228.4 ± 0.0 (0m07s) | f1_sample=0.5370 ± 0.0096 (43m34s) |
| cares | 100pct | original | ♻ reused | 224.8 | 228.4 ± 0.0 (0m07s) | f1_sample=0.8713 ± 0.0031 (66m03s) |
| cares | 50pct | generated | ♻ reused | 224.8 | 236.9 ± 0.0 (0m07s) | f1_sample=0.4884 ± 0.0244 (55m25s) |
| cares | 50pct | original | ♻ reused | 224.8 | 236.9 ± 0.0 (0m07s) | f1_sample=0.8667 ± 0.0015 (72m31s) |
| cares | 25pct | generated | ♻ reused | 224.8 | 230.9 ± 0.0 (0m06s) | f1_sample=0.5022 ± 0.0031 (47m04s) |
| cares | 25pct | original | ♻ reused | 224.8 | 230.9 ± 0.0 (0m07s) | f1_sample=0.8663 ± 0.0063 (45m39s) |
| cares | 5pct | generated | ♻ reused | 224.8 | 231.9 ± 0.0 (0m06s) | f1_sample=0.4264 ± 0.0113 (75m42s) |
| cares | 5pct | original | ♻ reused | 224.8 | 231.9 ± 0.0 (0m06s) | f1_sample=0.8697 ± 0.0008 (66m21s) |
| cwlc | 100pct | generated | ♻ reused | 34.0 | 38.4 ± 0.3 (0m18s) | f1=0.2464 ± 0.0195 (6m45s) |
| cwlc | 100pct | original | ♻ reused | 34.0 | 38.4 ± 0.3 (0m18s) | f1=0.5927 ± 0.0070 (6m26s) |
| cwlc | 50pct | generated | ♻ reused | 34.0 | 32.8 ± 0.0 (0m04s) | f1=0.5334 ± 0.0077 (5m32s) |
| cwlc | 50pct | original | ♻ reused | 34.0 | 32.8 ± 0.0 (0m04s) | f1=0.5928 ± 0.0075 (6m26s) |
| cwlc | 25pct | generated | ♻ reused | 34.0 | 35.4 ± 0.0 (0m05s) | f1=0.5213 ± 0.0061 (7m14s) |
| cwlc | 25pct | original | ♻ reused | 34.0 | 35.4 ± 0.0 (0m04s) | f1=0.5855 ± 0.0043 (5m26s) |
| cwlc | 5pct | generated | ♻ reused | 34.0 | 42.2 ± 0.0 (0m04s) | f1=0.4970 ± 0.0062 (6m30s) |
| cwlc | 5pct | original | ♻ reused | 34.0 | 42.2 ± 0.0 (0m05s) | f1=0.5855 ± 0.0043 (5m24s) |
| mimic_s | 100pct | generated | ♻ reused | 1349.2 | 1215.4 ± 0.0 (2m31s) | f1_micro=0.1768 ± 0.0065 (170m07s) |
| mimic_s | 100pct | original | ♻ reused | 1349.2 | 1215.4 ± 0.0 (2m28s) | f1_micro=0.2306 ± 0.0035 (231m12s) |
| mimic_s | 50pct | generated | ♻ reused | 1349.2 | 1255.2 ± 0.0 (2m34s) | f1_micro=0.1856 ± 0.0007 (242m32s) |
| mimic_s | 50pct | original | ♻ reused | 1349.2 | 1255.2 ± 0.0 (2m26s) | f1_micro=0.2386 ± 0.0297 (249m50s) |
| mimic_s | 25pct | generated | ♻ reused | 1349.2 | 1333.5 ± 0.0 (2m33s) | f1_micro=0.1837 ± 0.0020 (243m08s) |
| mimic_s | 25pct | original | ♻ reused | 1349.2 | 1333.5 ± 0.0 (2m32s) | f1_micro=0.2248 ± 0.0054 (164m54s) |
| mimic_s | 5pct | generated | ♻ reused | 1349.2 | 1328.3 ± 0.0 (2m30s) | f1_micro=0.1761 ± 0.0056 (169m51s) |
| mimic_s | 5pct | original | ♻ reused | 1349.2 | 1328.3 ± 0.0 (2m31s) | f1_micro=0.2325 ± 0.0463 (250m07s) |
| radgraph | 100pct | generated | ♻ reused | 411.1 | 887.8 ± 6.1 (1m05s) | f1=0.7514 ± 0.0013 (3m47s) |
| radgraph | 100pct | original | ♻ reused | 411.1 | 887.8 ± 6.1 (1m06s) | f1=0.7996 ± 0.0025 (4m15s) |
| radgraph | 50pct | generated | ♻ reused | 411.1 | 943.9 ± 0.0 (0m16s) | f1=0.7488 ± 0.0006 (3m49s) |
| radgraph | 50pct | original | ♻ reused | 411.1 | 943.9 ± 0.0 (0m15s) | f1=0.7996 ± 0.0025 (4m05s) |
| radgraph | 25pct | generated | ♻ reused | 411.1 | 928.5 ± 0.0 (0m15s) | f1=0.7165 ± 0.0085 (3m46s) |
| radgraph | 25pct | original | ♻ reused | 411.1 | 928.5 ± 0.0 (0m16s) | f1=0.7996 ± 0.0024 (4m05s) |
| radgraph | 5pct | generated | ♻ reused | 411.1 | 838.6 ± 0.0 (0m14s) | f1=0.7199 ± 0.0028 (3m41s) |
| radgraph | 5pct | original | ♻ reused | 411.1 | 838.6 ± 0.0 (0m14s) | f1=0.7999 ± 0.0021 (4m04s) |
