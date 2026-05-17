# SpAtten Experiment Summary Report

## Key Finding 1: Perplexity Impact

- **Sp-Quant**: avg PPL degradation = +0.02% (range: +0.02% to +0.02%)
- **Sp-V**: avg PPL degradation = +-0.00% (range: +-0.34% to +0.51%)
- **Sp-Full**: avg PPL degradation = +0.39% (range: +-0.19% to +1.22%)

## Key Finding 2: Throughput Cross-over Point

- seq_len=1024: Best = **Sp-Quant** (0.92x vs Dense) [SLOWER]
- seq_len=2048: Best = **Sp-Quant** (1.17x vs Dense) [FASTER]
- seq_len=4096: Best = **Sp-Quant** (0.91x vs Dense) [SLOWER]
- seq_len=8192: Best = **Sp-V** (0.92x vs Dense) [SLOWER]

## Key Finding 3: Recommended Configuration

Based on the PPL-throughput Pareto frontier:
- **Short sequences (< 2048)**: Use Dense-SDPA (SpAtten overhead dominates)
- **Medium sequences (2048-4096)**: Sp-Quant or Sp-V (good speedup, minimal PPL impact)
- **Long sequences (> 4096)**: Sp-V (best throughput at acceptable PPL cost)
- **Quality-sensitive scenarios**: Sp-Quant (near-zero PPL degradation)

## Perplexity Analysis

### Sequence Length = 512

| Variant | Mode | Head Prune | Quant Th | V Th | PPL | vs Dense |
|---------|------|------------|----------|------|-----|----------|
| Dense | dense | hpn=0 | None | None | 4.5804 | (baseline) |
| Sp-Quant | quant_only | hpn=0 | 0.01 | 0.05 | 4.5813 | +0.02% |
| Sp-V | v_prune_only | hpn=0 | 0.01 | 0.05 | 4.6038 | +0.51% |
| Sp-Full | full | hpn=0 | 0.01 | 0.05 | 4.6363 | +1.22% |

**Best non-dense**: Sp-Quant (PPL=4.5813, +0.02% vs Dense)
**Worst variant**: Sp-Full (PPL=4.6363, +1.22% vs Dense)

### Sequence Length = 1024

| Variant | Mode | Head Prune | Quant Th | V Th | PPL | vs Dense |
|---------|------|------------|----------|------|-----|----------|
| Dense | dense | hpn=0 | None | None | 3.9086 | (baseline) |
| Sp-Quant | quant_only | hpn=0 | 0.01 | 0.05 | 3.9094 | +0.02% |
| Sp-V | v_prune_only | hpn=0 | 0.01 | 0.05 | 3.9107 | +0.05% |
| Sp-Full | full | hpn=0 | 0.01 | 0.05 | 3.9281 | +0.50% |

**Best non-dense**: Sp-Quant (PPL=3.9094, +0.02% vs Dense)
**Worst variant**: Sp-Full (PPL=3.9281, +0.50% vs Dense)

### Sequence Length = 2048

| Variant | Mode | Head Prune | Quant Th | V Th | PPL | vs Dense |
|---------|------|------------|----------|------|-----|----------|
| Sp-V | v_prune_only | hpn=0 | 0.01 | 0.05 | 3.5894 | +-0.23% |
| Dense | dense | hpn=0 | None | None | 3.5976 | (baseline) |
| Sp-Quant | quant_only | hpn=0 | 0.01 | 0.05 | 3.5983 | +0.02% |
| Sp-Full | full | hpn=0 | 0.01 | 0.05 | 3.5990 | +0.04% |

**Best non-dense**: Sp-V (PPL=3.5894, +-0.23% vs Dense)
**Worst variant**: Sp-Full (PPL=3.5990, +0.04% vs Dense)

### Sequence Length = 4096

| Variant | Mode | Head Prune | Quant Th | V Th | PPL | vs Dense |
|---------|------|------------|----------|------|-----|----------|
| Sp-V | v_prune_only | hpn=0 | 0.01 | 0.05 | 3.4116 | +-0.34% |
| Sp-Full | full | hpn=0 | 0.01 | 0.05 | 3.4169 | +-0.19% |
| Dense | dense | hpn=0 | None | None | 3.4234 | (baseline) |
| Sp-Quant | quant_only | hpn=0 | 0.01 | 0.05 | 3.4240 | +0.02% |

**Best non-dense**: Sp-V (PPL=3.4116, +-0.34% vs Dense)
**Worst variant**: Sp-Quant (PPL=3.4240, +0.02% vs Dense)


## Throughput Analysis

### Sequence Length = 1024

| Variant | Prefill tok/s | Decode tok/s | TTFT ms | Total tok/s | vs Dense |
|---------|---------------|--------------|---------|-------------|----------|
| Dense | 22003 | 29.1 | 34.5 | 1743 | (baseline) |
| Sp-Quant | 19608 | 26.8 | 40.7 | 1602 | 0.92x |
| Sp-Full | 17253 | 24.5 | 40.4 | 1460 | 0.84x |
| Sp-V | 17800 | 21.7 | 38.5 | 1311 | 0.75x |

### Sequence Length = 2048

| Variant | Prefill tok/s | Decode tok/s | TTFT ms | Total tok/s | vs Dense |
|---------|---------------|--------------|---------|-------------|----------|
| Sp-Quant | 22720 | 31.5 | 84.0 | 3448 | 1.17x |
| Sp-Full | 22972 | 27.0 | 82.7 | 3025 | 1.03x |
| Dense | 28324 | 25.4 | 66.0 | 2934 | (baseline) |
| Sp-V | 25822 | 23.4 | 72.7 | 2704 | 0.92x |

### Sequence Length = 4096

| Variant | Prefill tok/s | Decode tok/s | TTFT ms | Total tok/s | vs Dense |
|---------|---------------|--------------|---------|-------------|----------|
| Dense | 26590 | 32.0 | 139.7 | 6293 | (baseline) |
| Sp-Quant | 19168 | 31.6 | 198.4 | 5707 | 0.91x |
| Sp-V | 22933 | 26.8 | 163.7 | 5302 | 0.84x |
| Sp-Full | 19643 | 27.9 | 189.8 | 5255 | 0.84x |

### Sequence Length = 8192

| Variant | Prefill tok/s | Decode tok/s | TTFT ms | Total tok/s | vs Dense |
|---------|---------------|--------------|---------|-------------|----------|
| Dense | 22596 | 26.9 | 330.8 | 8564 | (baseline) |
| Sp-V | 18142 | 27.1 | 418.4 | 7878 | 0.92x |
| Sp-Quant | 14157 | 30.7 | 546.7 | 7464 | 0.87x |
| Sp-Full | 14790 | 27.9 | 531.1 | 7286 | 0.85x |


## PPL vs Throughput Trade-off

### Seq Len = 1024

| Variant | PPL | Total tok/s | PPL degradation | Speedup |
|---------|-----|-------------|-----------------|---------|
| Sp-Quant | 3.9094 | 1602 | +0.02% | 0.92x |
| Sp-V | 3.9107 | 1311 | +0.05% | 0.75x |
| Sp-Full | 3.9281 | 1460 | +0.50% | 0.84x |

### Seq Len = 2048

| Variant | PPL | Total tok/s | PPL degradation | Speedup |
|---------|-----|-------------|-----------------|---------|
| Sp-Quant | 3.5983 | 3448 | +0.02% | 1.17x |
| Sp-V | 3.5894 | 2704 | +-0.23% | 0.92x |
| Sp-Full | 3.5990 | 3025 | +0.04% | 1.03x |

### Seq Len = 4096

| Variant | PPL | Total tok/s | PPL degradation | Speedup |
|---------|-----|-------------|-----------------|---------|
| Sp-Quant | 3.4240 | 5707 | +0.02% | 0.91x |
| Sp-V | 3.4116 | 5302 | +-0.34% | 0.84x |
| Sp-Full | 3.4169 | 5255 | +-0.19% | 0.84x |
