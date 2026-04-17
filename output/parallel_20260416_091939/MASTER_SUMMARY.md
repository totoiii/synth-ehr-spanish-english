# Parallel pipeline master summary — 20260416_091939

Datasets: cares cwlc
Fractions: 5pct 25pct 50pct 100pct
Text sources: generated original
Stages: eval downstream
N_RUNS (seeds): 3
N_GEN: 1
GPUs used: 0 1 2
Wall clock: 2h25m

| Dataset | Fraction | TextSrc | Generate | tok/doc | Downstream F1 |
|---------|----------|---------|----------|---------|---------------|
| cares | 100pct | generated | ♻ reused | 228.4 (0m07s) | f1_sample=0.5714 ± 0.0105 (44m37s) |
| cares | 100pct | original | ♻ reused | 224.8 (0m06s) | f1_sample=0.8780 ± 0.0032 (65m58s) |
| cares | 25pct | generated | ♻ reused | 230.9 (0m06s) | f1_sample=0.5385 ± 0.0051 (69m01s) |
| cares | 50pct | generated | ♻ reused | 236.9 (0m06s) | f1_sample=0.5314 ± 0.0115 (44m28s) |
| cares | 5pct | generated | ♻ reused | 231.9 (0m06s) | f1_sample=0.5232 ± 0.0018 (72m05s) |
| cwlc | 100pct | generated | ♻ reused | 38.4 (0m18s) | — (0m51s) |
| cwlc | 100pct | original | ♻ reused | 34.0 (0m18s) | — (0m51s) |
| cwlc | 25pct | generated | ♻ reused | 35.4 (0m04s) | f1=0.6372 ± 0.0131 (22m04s) |
| cwlc | 50pct | generated | ♻ reused | 32.8 (0m04s) | f1=0.6516 ± 0.0053 (36m11s) |
| cwlc | 5pct | generated | ♻ reused | 42.2 (0m05s) | f1=0.6209 ± 0.0036 (36m45s) |
