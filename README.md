# Reference-Free Fusion and Self-Calibration of Asynchronous Sensor Arrays

Reproducibility code and data for the manuscript *"Reference-Free Fusion and
Self-Calibration of Asynchronous Sensor Arrays with Online Time-Delay
Estimation"*.

The method treats a homogeneous sensor array as an ordered stream of
measurement events consumed at their true arrival times, with no resampling
onto a common grid. A fast event-driven recursive layer estimates the latent
signal and its rate; a slow bounded least-squares layer jointly identifies
each sensor's gain, bias and sampling-time offset from the array data alone,
without any external reference.

Everything in the paper is reproducible from this repository with two
commands.

## Repository layout

```
synthetic/
    run_experiments.py      synthetic experiment suite (Section 4)
    requirements.txt
hardware/
    analyse_real_experiment.py   hardware analysis (Section 5)
    data/                        thirty recordings of the 16-sensor array
        real_{cos,fluct,ramp}_rep{1..10}.csv
LICENSE
```

The estimator core (`_kf_pass`, `fuse_proposed`, `fuse_nemec`, ...) lives in
`synthetic/run_experiments.py` and is imported by the hardware script, so
exactly one implementation exists in the repository.

## Setup

```bash
pip install -r synthetic/requirements.txt
```

Numba is an *optional* accelerator. The suite detects it automatically and
falls back to an identical pure-Python implementation without it: the same
numbers, roughly twenty times slower.

## Synthetic experiments

```bash
cd synthetic
python run_experiments.py --runs 50 --sweep-runs 10 --param-runs 12 --seed 1 --outdir results
```

That is the exact configuration behind every synthetic number in the paper
(a few minutes with Numba). A fast smoke test:

```bash
python run_experiments.py --quick --outdir results_quick
```

Useful options: `--signals cos fluct ramp`, `--seed`, `--no-sweeps`,
`--no-convergence`, `--no-sensitivity`. All console tables and the per-run
RMSE values are also written to `results/results.json`.

### Outputs and the paper

| Output | Paper item |
|---|---|
| `results.json`, console tables | Table 2 — RMSE of all six methods across three regimes |
| `results.json` → `ablation` | Table 3 — component ablation |
| `results.json` → `sensitivity` | Table 4 — sensitivity to the hand-set settings |
| `fig_summary.pdf` | Figure 1 |
| `fig_qconv.pdf` | Figure 2 — process noise from five starting values |
| `fig_sweep_drop.pdf` | Figure 3 |
| `fig_sweep_burst.pdf` | Figure 4 |
| `fig_sweep_rate.pdf` | Figure 5 |
| `fig_sweep_tau.pdf` | Figure 6 |
| `fig_param_{cos,fluct,ramp}.pdf` | Figure 7 |

## Hardware experiments

```bash
cd hardware
python analyse_real_experiment.py data/real_*_rep*.csv --outdir results --no-sweeps
python analyse_real_experiment.py data/real_fluct_rep1.csv --outdir results --sweep-signal fluct_rep1
```

The first command analyses all thirty recordings; the second adds the
packet-loss replay sweeps, which are reported for one fluctuating-frequency
recording.

### Outputs and the paper

| Output | Paper item |
|---|---|
| `results.json`, console tables | Tables 5 and 6 — encoder uncertainty budget, fused RMSE |
| `real_{cos,fluct,ramp}_1.pdf` | Figure 9 |
| `real_cos_2.pdf` | Figure 10 |
| `real_cos_3.pdf` | Figure 11 |
| `fig_real_sweep_drop_fluct.pdf` | Figure 12 |
| `fig_real_sweep_burst_fluct.pdf` | Figure 13 |

Figure 8 is a photograph of the test rig and is not produced by the code.

### The encoder reference

The reference angular velocity is a windowed finite difference of the encoder
angle, which is exactly a moving average of the true velocity and therefore
attenuates the signal it measures. Two details matter and both are options of
the script:

* `--attribution midpoint` (default) places the difference at the midpoint of
  the interval it spans. The earlier index-based placement,
  `--attribution index`, offsets the reference by half a sample whenever the
  window length is odd.
* `--warmup 0.5` (default) discards the first half second of each recording,
  which covers the start-up transient.

The script sweeps the window length and reports the resulting
reference-induced error, so the choice of a 0.25-s window can be checked
rather than taken on trust.

## Data format

Each CSV holds one recording: one row per control cycle, 34 columns, roughly
3500 rows over 30 s. The encoder pair comes first, then each sensor's
timestamp and reading interleaved:

```
t_enc_us, enc_deg, t1_us, y1, t2_us, y2, ... , t16_us, y16
```

| Column | Meaning |
|---|---|
| `t_enc_us` | encoder timestamp, microseconds |
| `enc_deg` | encoder angle, degrees |
| `tj_us` | timestamp of sensor *j*, microseconds |
| `yj` | angular rate of sensor *j*, deg/s |

All timestamps come from the same microcontroller clock. The sensors are read
sequentially over a shared SPI bus, so the timestamps within one cycle differ
by the read-out span — this is the asynchrony the method consumes directly.

## Reproducibility

* Each experiment component draws from an independent deterministic random
  stream derived from `--seed`, so results are identical whether components
  run sequentially, individually, or in parallel.
* Variance computations use the unbiased (n-1) normalisation.
* Timing figures are the fastest of twenty-five timed repetitions after a
  discarded warm-up run. They are offline batch measurements on a desktop
  CPU, not an embedded implementation.

## Baselines

| Name | What it is |
|---|---|
| naive mean | unweighted average of the raw outputs |
| calibrated mean | consensus gain/bias removal, then an unweighted average |
| Nemec (ZOH) | the synchronous weighted-fusion-with-calibration method fed by a causal zero-order hold |
| Nemec (interp) | the same method fed by a non-causal linear-interpolation reconstruction |
| async KF (raw) | an event-driven Kalman filter on the raw stream, without calibration or delay compensation, at an oracle-selected process noise |
| proposed | the full two-layer method |

## License

MIT — see `LICENSE`.

## Citation

Citation information will be added once the paper is published.
