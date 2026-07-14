# Asynchronous Self-Calibrating Sensor Array Fusion — Reproducibility Code

Code and data accompanying the manuscript *"Asynchronous Self-Calibrating
Sensor Array Fusion with Online Time-Delay Estimation"* (submitted to MDPI
Sensors).

## Repository layout

```
synthetic/          Synthetic experiment suite (Section 4 of the paper)
    run_experiments.py
    requirements.txt
hardware/           Hardware-experiment analysis (Section 5)
    analyse_real_experiment.py
    data/           16-sensor MPU9250 array recordings (CSV)
```

## Synthetic experiments

The suite reproduces every synthetic table and figure in the paper:
the 3-regime × 5-method RMSE comparison for three excitation signals,
the parameter-identification scatter plots, and the robustness sweeps
(packet loss, rate heterogeneity, time delay, and bursty correlated loss).

### Setup

```bash
pip install -r requirements.txt        # numpy, scipy, matplotlib, numba
```

Numba is an *optional* accelerator: the suite detects it automatically and
falls back to an identical pure-Python implementation if it is missing
(same results, roughly 20x slower).

### Run

```bash
# full paper configuration (50 Monte Carlo runs; ~1-2 min with numba,
# ~20-25 min in the pure-Python fallback)
python run_experiments.py --runs 50 --sweep-runs 10 --param-runs 12 --seed 1 --outdir results

# fast smoke test (~1 min)
python run_experiments.py --quick --outdir results_quick
```

Options: `--runs`, `--sweep-runs`, `--signals cos fluct ramp`, `--seed`,
`--no-sweeps`. All console tables and per-run RMSE values are also written
to `results/results.json`.

### Outputs → paper mapping

| File | Paper item |
|---|---|
| console tables / `results.json` | Table 1 (mean ± std, median, Wilcoxon p) |
| `fig_summary.png` | Figure 1 |
| `fig_sweep_drop.png` | Figure 2 |
| `fig_sweep_rate.png` | Figure 3 |
| `fig_sweep_tau.png` | Figure 4 |
| `fig_sweep_burst.png` | bursty-loss robustness figure (Fig.~\ref{fig:burst}) |
| `fig_param_<signal>.png` | Figure 5 |

### Notes on reproducibility

* Each experiment component draws from an independent, deterministic
  random stream derived from `--seed`, so the suite produces identical
  numbers whether components run sequentially, individually, or in
  parallel.
* The `Nemec (interp)` baseline feeds the synchronous method with an
  offline (non-causal) linear-interpolation reconstruction — the strongest
  practical resampling front-end — alongside the causal zero-order hold.

## License

Code and data are released under the MIT License (see `LICENSE`).

## Citation

Citation information will be added upon publication of the paper.
