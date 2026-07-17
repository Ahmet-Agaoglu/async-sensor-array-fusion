# Reference-Free Fusion and Self-Calibration of Asynchronous Sensor Arrays with Online Time-Delay Estimation — Reproducibility Code

Code and data accompanying the manuscript *"Reference-Free Fusion and Self-Calibration of Asynchronous Sensor Arrays with Online Time-Delay Estimation"*.

## Repository layout

```
synthetic/          Synthetic experiment suite
    run_experiments.py
    requirements.txt
hardware/           Hardware-experiment analysis and array recordings
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

### Outputs

Together these regenerate every synthetic table and figure of the paper:

| File | Content |
|---|---|
| console tables / `results.json` | 3-regime × 5-method RMSE comparison (mean ± std, median, Wilcoxon p) |
| `fig_summary.png` | per-regime RMSE summary bars |
| `fig_sweep_drop.png` | packet-loss robustness sweep |
| `fig_sweep_rate.png` | rate-heterogeneity robustness sweep |
| `fig_sweep_tau.png` | time-delay robustness sweep |
| `fig_sweep_burst.png` | bursty (Gilbert–Elliott) loss robustness sweep |
| `fig_param_<signal>.png` | parameter-identification accuracy scatter |

### Notes on reproducibility

* Each experiment component draws from an independent, deterministic
  random stream derived from `--seed`, so the suite produces identical
  numbers whether components run sequentially, individually, or in
  parallel.
* The `Nemec (interp)` baseline feeds the synchronous method with an
  offline (non-causal) linear-interpolation reconstruction — the strongest
  practical resampling front-end — alongside the causal zero-order hold.

## Hardware experiments

`hardware/analyse_real_experiment.py` reproduces every analysis of the Real
Experiments section from the three CSV recordings of the 16-sensor MPU9250
array: the two-reference RMSE comparison, the encoder-reference resolution
study, the per-sensor parameter and convergence figures, the runtime
statistics, and the asynchrony/packet-loss replay sweeps on the recorded
event streams.

### Run

```bash
cd hardware
python analyse_real_experiment.py data/real_cos.csv data/real_fluct.csv \
       data/real_ramp.csv --outdir results          # ~2 min
```

Options: `--gt-win`, `--gt-win-corrected`, `--windows`, `--seeds`,
`--sweep-signal`, `--no-sweeps`. All numerical results are also written to
`results/real_results.json`.

### Outputs

Together these regenerate every table and figure of the paper's hardware
section:

| File | Content |
|---|---|
| `real_results.json` | RMSE against both encoder references, reference-window study, calibration statistics, timing |
| `real_<profile>_1.png` | fused velocity and error against the encoder reference |
| `real_<profile>_2.png` | estimated per-sensor gain, bias, and timing offset |
| `real_<profile>_3.png` | filter convergence diagnostics |
| `fig_real_sweep_drop_fluct.png` | replayed i.i.d. packet-loss sweep |
| `fig_real_sweep_burst_fluct.png` | replayed bursty packet-loss sweep |

### Data format

Each CSV holds one row per control loop:
`t_enc_us, enc_deg, t1_us, y1, ..., t16_us, y16` — encoder angle in degrees,
per-sensor angular rates in deg/s, and all time stamps in microseconds from
the shared microcontroller clock.

## License

Code and data are released under the MIT License (see `LICENSE`).

## Citation

Citation information will be added upon publication of the paper.
