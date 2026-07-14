#!/usr/bin/env python3
# =========================================================================
# SYNTHETIC EXPERIMENT SUITE
#
# Reproduces every synthetic table and figure of the paper:
#
#   * 3-regime x 5-method RMSE comparison for three excitation signals
#     (mean +/- std and median over the Monte Carlo runs, with a paired
#     Wilcoxon signed-rank test against the strongest baseline per regime)
#   * parameter-identification accuracy (gain, bias, delay)
#   * robustness sweeps: packet loss, rate heterogeneity, time delay, and
#     bursty (Gilbert-Elliott) correlated packet loss
#
# Signals:  'cos'    pure cosine
#           'fluct'  fluctuating-frequency sine
#           'ramp'   triangular wave (piecewise-linear)
#
# Outputs (written to --outdir, default ./results):
#   comparison tables (console + results.json)
#   fig_summary.png                       summary bars per regime
#   fig_param_<sig>.png                   parameter-accuracy scatter
#   fig_sweep_{drop,rate,tau,burst}.png   robustness sweeps ('fluct' signal)
#
# Dependencies: numpy, scipy, matplotlib; numba is optional (the suite
# falls back to an identical pure-Python implementation without it).
#
# Reproducibility notes
# ---------------------
# * Each experiment component draws from an independent, deterministic
#   random stream derived from --seed, so components produce identical
#   results whether run sequentially, individually, or in parallel.
# * Variance computations use the unbiased (n-1) normalisation (ddof=1).
# =========================================================================

import argparse
import json
import os
import time

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = np.finfo(float).eps

COMPONENT_SEED = {"table:cos": 101, "table:fluct": 102, "table:ramp": 103,
                  "param:cos": 201, "param:fluct": 202, "param:ramp": 203,
                  "sweep:drop": 301, "sweep:rate": 302,
                  "sweep:tau": 303, "sweep:burst": 304}

METHODS = ["naive", "calib (ZOH)", "Nemec (ZOH)", "Nemec (interp)", "proposed"]
REGIMES = ["SYNC", "ASYNC", "ASYNC+SKEW"]


# =========================================================================
# CONFIG
# =========================================================================
def default_cfg():
    return dict(
        J=16, Fs=100.0, N=6000, A=200.0, fsig=1.0,
        gain_sigma=0.04, bias_sigma=30.0, gain_max=0.04, bias_max=30.0,
        decim_set=[1, 1, 2, 2, 3, 4], p_drop=0.10, tau_max=0.005,
        sig="cos",
    )


# =========================================================================
# DATA GENERATION  (signal-type aware; delay via interpolation)
# =========================================================================
def gen_signal(cfg, t, rng):
    sig = cfg["sig"]
    A, fsig, Fs = cfg["A"], cfg["fsig"], cfg["Fs"]
    if sig == "cos":
        return A * np.cos(2 * np.pi * fsig * t)
    if sig == "fluct":  # fluctuating instantaneous frequency
        f = fsig + 0.3 * rng.standard_normal(t.size)
        phase = 2 * np.pi / Fs * np.cumsum(f)
        return A * np.sin(phase)
    if sig == "ramp":   # triangular wave (piecewise-linear)
        Tp = 4.0
        ph = np.mod(t / Tp, 1.0)
        return A * (2 * np.abs(2 * ph - 1) - 1)
    return A * np.cos(2 * np.pi * fsig * t)


def make_data(cfg, with_delay, rng):
    J, N, Fs = cfg["J"], cfg["N"], cfg["Fs"]
    G = 1 + cfg["gain_sigma"] * rng.standard_normal(J)
    B = cfg["bias_sigma"] * rng.standard_normal(J)
    G = G - G.mean() + 1
    B = B - B.mean()
    G = np.clip(G, 1 - cfg["gain_max"], 1 + cfg["gain_max"])
    B = np.clip(B, -cfg["bias_max"], cfg["bias_max"])
    # noise std ~ 0.02*Gamma(5,1)  (mean 0.1 deg/s)
    RMSn = -0.02 * np.sum(np.log(rng.random((J, 5))), axis=1)
    tau = cfg["tau_max"] * (2 * rng.random(J) - 1)
    tau = tau - tau.mean()

    t = np.arange(N) / Fs
    wtrue = gen_signal(cfg, t, rng)
    Y = wtrue[:, None] * G[None, :] + B[None, :] \
        + rng.standard_normal((N, J)) * RMSn[None, :]
    Yd = np.zeros((N, J))
    if with_delay:
        f_ext = interp1d(t, wtrue, kind="linear", fill_value="extrapolate",
                         assume_sorted=True)
        for j in range(J):
            wd = f_ext(t - tau[j])
            Yd[:, j] = wd * G[j] + B[j] + RMSn[j] * rng.standard_normal(N)
    return t, wtrue, Y, Yd, G, B, RMSn, tau


def build_async_mask(N, J, decim_set, p_drop, rng, burst_len=1):
    """Per-sensor decimation + packet loss.

    burst_len = 1   : i.i.d. Bernoulli loss.
    burst_len = L>1 : two-state Gilbert-Elliott loss with stationary loss
                      probability p_drop and mean burst length L samples,
                      emulating the correlated dropouts of wireless links.
    """
    mask = np.zeros((N, J), dtype=bool)
    dset = np.asarray(decim_set)
    for j in range(J):
        d = int(dset[rng.integers(dset.size)])
        keep = np.zeros(N, dtype=bool)
        keep[0:N:d] = True
        if burst_len <= 1 or p_drop <= 0:
            keep &= rng.random(N) > p_drop
        else:
            p_bg = 1.0 / burst_len                        # bad -> good
            p_gb = p_drop / ((1 - p_drop) * burst_len)    # good -> bad
            bad = rng.random() < p_drop
            u = rng.random(N)
            lost = np.zeros(N, dtype=bool)
            for k in range(N):
                lost[k] = bad
                bad = (u[k] < p_gb) if not bad else (u[k] >= p_bg)
            keep &= ~lost
        mask[:, j] = keep
    mask[0, :] = True
    return mask


# =========================================================================
# GRID RECONSTRUCTIONS for synchronous baselines
# =========================================================================
def zoh_fill(Y, mask):
    """Zero-order hold: the last received value is held."""
    N, J = Y.shape
    idx = np.where(mask, np.arange(N)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return Y[idx, np.arange(J)[None, :]]


def interp_fill(Y, mask, t):
    """Linear interpolation between received samples.

    This is the strongest practical reconstruction a synchronous method can
    be given: it is non-causal (uses future samples), so it upper-bounds
    what any real-time resampling front-end could achieve.  Edges are held
    (row 0 is always sampled; a missing tail holds the last sample).
    """
    N, J = Y.shape
    Yg = np.empty_like(Y)
    for j in range(J):
        k = np.nonzero(mask[:, j])[0]
        Yg[:, j] = np.interp(t, t[k], Y[k, j])
    return Yg


def masked_mean(Y, mask):
    cnt = mask.sum(axis=1)
    s = np.where(mask, Y, 0.0).sum(axis=1)
    m = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    # forward-fill NaNs (row 0 is always complete)
    isn = np.isnan(m)
    if isn.any():
        idx = np.where(~isn, np.arange(m.size), 0)
        np.maximum.accumulate(idx, out=idx)
        m = m[idx]
    return m


# =========================================================================
# PROPOSED METHOD
# =========================================================================
# ---------------------------------------------------------------------
# Fast-layer inner recursion, extracted so it can be JIT-compiled.
# Numba is OPTIONAL: if unavailable, the identical pure-Python function
# runs (same expressions, same results, ~50x slower).
# ---------------------------------------------------------------------
def _kf_pass_py(te, ye, je, Ghat, Bhat, tauhat, Rest, q, warm):
    M = te.shape[0]
    w_ev = np.zeros(M)
    wd_ev = np.zeros(M)
    x0 = 0.0
    x1 = 0.0
    P00 = 1e4
    P01 = 0.0
    P11 = 1e4
    tprev = te[0]
    innov2 = 0.0
    cnt = 0
    for i in range(M):
        dt = te[i] - tprev
        if dt < 0.0:
            dt = 0.0
        tprev = te[i]
        # predict (F=[1 dt;0 1], Q=q*[dt^3/3 dt^2/2; dt^2/2 dt])
        x0 += dt * x1
        P00 += dt * (2 * P01 + dt * P11) + q * dt ** 3 / 3
        P01 += dt * P11 + q * dt ** 2 / 2
        P11 += q * dt
        # calibrated pseudo-measurement
        j = je[i]
        z = (ye[i] - Bhat[j]) / Ghat[j] + tauhat[j] * x1
        Rj = Rest[j] / Ghat[j] ** 2
        S = P00 + Rj
        nu = z - x0
        if i + 1 > warm:
            innov2 += nu * nu / S
            cnt += 1
        K0 = P00 / S
        K1 = P01 / S
        x0 += K0 * nu
        x1 += K1 * nu
        # P = (I-KH)P
        P11 -= K1 * P01
        P01 *= (1 - K0)
        P00 *= (1 - K0)
        w_ev[i] = x0
        wd_ev[i] = x1
    return w_ev, wd_ev, innov2, cnt


try:  # optional accelerator
    from numba import njit
    _kf_pass = njit(cache=True)(_kf_pass_py)
except Exception:  # pragma: no cover
    _kf_pass = _kf_pass_py


def fuse_proposed(t, Y, mask, Fs, gmax, bmax, tmax, est_tau, mu=3.0):
    N, J = Y.shape
    q = (200.0 * (2 * np.pi) ** 2) ** 2 * 1e-3   # start; adapted from data

    kk, jj = np.nonzero(mask)                    # row-major
    te = t[kk]
    ye = Y[kk, jj]
    je = jj
    order = np.argsort(te, kind="stable")  # keeps sensor order at equal times
    te, ye = te[order], ye[order]
    je = je[order].astype(np.int64)
    M = te.size

    Ghat = np.ones(J)
    Bhat = np.zeros(J)
    tauhat = np.zeros(J)
    Rest = np.full(J, 0.1 ** 2)
    warm = min(20, M // 10)                      # skip init transient

    for _ in range(5):
        w_ev, wd_ev, innov2, cnt = _kf_pass(te, ye, je, Ghat, Bhat,
                                            tauhat, Rest, q, warm)
        # --- data-driven q: drive normalized-innovation variance toward 1 ---
        if cnt > 0:
            ratio = innov2 / cnt
            q *= min(max(ratio, 0.3), 3.0) ** 0.7
            q = min(max(q, 1e1), 1e9)
        # --- slow layer: bounded least-squares self-calibration ---
        for j in range(J):
            idx = je == j
            n = int(idx.sum())
            if n < 8:
                continue
            if est_tau:
                A = np.column_stack([w_ev[idx], wd_ev[idx], np.ones(n)])
                s, *_ = np.linalg.lstsq(A, ye[idx], rcond=None)
                Ghat[j], tauhat[j], Bhat[j] = s[0], -s[1] / s[0], s[2]
            else:
                A = np.column_stack([w_ev[idx], np.ones(n)])
                s, *_ = np.linalg.lstsq(A, ye[idx], rcond=None)
                Ghat[j], Bhat[j] = s[0], s[1]
            Rest[j] = np.var(ye[idx] - (Ghat[j] * w_ev[idx] + Bhat[j]), ddof=1)
        Ghat = Ghat - Ghat.mean() + 1
        Bhat = Bhat - Bhat.mean()
        if est_tau:
            tauhat = tauhat - tauhat.mean()
        Ghat = np.clip(Ghat, 1 - gmax, 1 + gmax)
        Bhat = np.clip(Bhat, -bmax, bmax)
        tauhat = np.clip(tauhat, -tmax, tmax)
        wts = 1.0 / np.maximum(Rest, EPS)
        wts = np.minimum(wts, mu * wts.mean())
        Rest = 1.0 / wts

    # last state estimate per unique event time, interpolated to the grid
    te_rev_u, idx_rev = np.unique(te[::-1], return_index=True)
    last_idx = M - 1 - idx_rev
    te_u, w_u = te_rev_u, w_ev[last_idx]
    if te_u.size < 2:
        w_grid = np.full(t.size, w_u[0])
    else:
        w_grid = interp1d(te_u, w_u, kind="linear",
                          fill_value="extrapolate", assume_sorted=True)(t)
    return w_grid, Ghat, Bhat, tauhat, q


# =========================================================================
# BASELINE METHODS
# =========================================================================
def fuse_nemec(Yg, gmax, bmax, mu=3.0):
    N, J = Yg.shape
    Ghat = np.ones(J)
    Bhat = np.zeros(J)
    Cal = Yg.copy()
    Omega = Yg.mean(axis=1)
    for _ in range(3):
        A = np.column_stack([Omega, np.ones(N)])
        for j in range(J):
            s, *_ = np.linalg.lstsq(A, Yg[:, j], rcond=None)
            Ghat[j], Bhat[j] = s[0], s[1]
        Ghat = Ghat - Ghat.mean() + 1
        Bhat = Bhat - Bhat.mean()
        Ghat = np.clip(Ghat, 1 - gmax, 1 + gmax)
        Bhat = np.clip(Bhat, -bmax, bmax)
        Cal = (Yg - Bhat[None, :]) / Ghat[None, :]
        MSE = np.var(Cal - Omega[:, None], axis=0, ddof=1)
        wts = 1.0 / np.maximum(MSE, EPS)
        wts = np.minimum(wts, mu * wts.mean())
        wts = wts / wts.sum()
        Omega = Cal @ wts
    return Omega


def calibrated_mean(Yg, gmax, bmax):
    N, J = Yg.shape
    Gh = np.ones(J)
    Bh = np.zeros(J)
    Om = Yg.mean(axis=1)
    for _ in range(3):
        A = np.column_stack([Om, np.ones(N)])
        for j in range(J):
            s, *_ = np.linalg.lstsq(A, Yg[:, j], rcond=None)
            Gh[j], Bh[j] = s[0], s[1]
        Gh = Gh - Gh.mean() + 1
        Bh = Bh - Bh.mean()
        Gh = np.clip(Gh, 1 - gmax, 1 + gmax)
        Bh = np.clip(Bh, -bmax, bmax)
        Om = ((Yg - Bh[None, :]) / Gh[None, :]).mean(axis=1)
    return Om


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# =========================================================================
# COMPARISON TABLE (3 regimes x 5 methods) for one signal
# =========================================================================
def comparison_table(cfg, n_runs, label, rng):
    R = {reg: np.zeros((n_runs, len(METHODS))) for reg in REGIMES}
    for r in range(n_runs):
        t, wtrue, Y, Yd, *_ = make_data(cfg, True, rng)
        full = np.ones((cfg["N"], cfg["J"]), dtype=bool)
        # --- SYNC ---
        R["SYNC"][r] = [
            rmse(Y.mean(axis=1), wtrue),
            rmse(calibrated_mean(Y, cfg["gain_max"], cfg["bias_max"]), wtrue),
            rmse(fuse_nemec(Y, cfg["gain_max"], cfg["bias_max"]), wtrue),
            rmse(fuse_nemec(Y, cfg["gain_max"], cfg["bias_max"]), wtrue),
            rmse(fuse_proposed(t, Y, full, cfg["Fs"], cfg["gain_max"],
                               cfg["bias_max"], cfg["tau_max"], False)[0], wtrue),
        ]
        # --- ASYNC ---
        m = build_async_mask(cfg["N"], cfg["J"], cfg["decim_set"],
                             cfg["p_drop"], rng)
        R["ASYNC"][r] = [
            rmse(masked_mean(Y, m), wtrue),
            rmse(calibrated_mean(zoh_fill(Y, m), cfg["gain_max"],
                                 cfg["bias_max"]), wtrue),
            rmse(fuse_nemec(zoh_fill(Y, m), cfg["gain_max"],
                            cfg["bias_max"]), wtrue),
            rmse(fuse_nemec(interp_fill(Y, m, t), cfg["gain_max"],
                            cfg["bias_max"]), wtrue),
            rmse(fuse_proposed(t, Y, m, cfg["Fs"], cfg["gain_max"],
                               cfg["bias_max"], cfg["tau_max"], False)[0], wtrue),
        ]
        # --- ASYNC + SKEW ---
        m2 = build_async_mask(cfg["N"], cfg["J"], cfg["decim_set"],
                              cfg["p_drop"], rng)
        R["ASYNC+SKEW"][r] = [
            rmse(masked_mean(Yd, m2), wtrue),
            rmse(calibrated_mean(zoh_fill(Yd, m2), cfg["gain_max"],
                                 cfg["bias_max"]), wtrue),
            rmse(fuse_nemec(zoh_fill(Yd, m2), cfg["gain_max"],
                            cfg["bias_max"]), wtrue),
            rmse(fuse_nemec(interp_fill(Yd, m2, t), cfg["gain_max"],
                            cfg["bias_max"]), wtrue),
            rmse(fuse_proposed(t, Yd, m2, cfg["Fs"], cfg["gain_max"],
                               cfg["bias_max"], cfg["tau_max"], True)[0], wtrue),
        ]

    print(f"\n========= SIGNAL: {label}  (RMSE deg/s over {n_runs} runs) =========")
    hdr = f"{'method':<15}" + "".join(f"{reg:<28}" for reg in REGIMES)
    print(hdr)
    for k, name in enumerate(METHODS):
        row = f"{name:<15}"
        for reg in REGIMES:
            v = R[reg][:, k]
            row += f"{v.mean():6.3f} +/-{v.std(ddof=1):5.3f} (med {np.median(v):5.3f})  "
        print(row)

    # paired Wilcoxon: proposed vs strongest baseline, per regime
    print(f"{'-'*15} paired Wilcoxon signed-rank: proposed vs strongest baseline")
    wilcox = {}
    for reg in REGIMES:
        base_means = R[reg][:, :-1].mean(axis=0)
        b = int(np.argmin(base_means))
        prop = R[reg][:, -1]
        base = R[reg][:, b]
        try:
            stat_res = stats.wilcoxon(prop, base)
            p = float(stat_res.pvalue)
        except ValueError:
            p = float("nan")
        wilcox[reg] = dict(strongest_baseline=METHODS[b], p_value=p)
        print(f"  {reg:<12} vs {METHODS[b]:<15} p = {p:.2e}"
              f"   (proposed mean {prop.mean():.3f} vs {base.mean():.3f})")
    return R, wilcox


# =========================================================================
# SUMMARY BAR: Nemec(ZOH) / Nemec(interp) / proposed, per regime, per signal
# =========================================================================
def summary_bar(SUM, labels, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    x = np.arange(len(labels))
    w = 0.26
    for rg, ax in enumerate(axes):
        ax.bar(x - w, SUM[:, rg, 0], w, label="Nemec (ZOH)")
        ax.bar(x,     SUM[:, rg, 1], w, label="Nemec (interp)")
        ax.bar(x + w, SUM[:, rg, 2], w, label="Proposed")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("RMSE (deg/s)")
        ax.set_title(REGIMES[rg])
        ax.grid(True, axis="y", alpha=0.4)
        if rg == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(outdir, "fig_summary.png"), dpi=200)
    plt.close(fig)


# =========================================================================
# PARAMETER ACCURACY for one signal
# =========================================================================
def param_accuracy(cfg, n_runs, sig, rng, outdir):
    Ge, Gt, Be, Bt, Te, Tt = ([] for _ in range(6))
    for _ in range(n_runs):
        t, _, _, Yd, G, B, _, tau = make_data(cfg, True, rng)
        m = build_async_mask(cfg["N"], cfg["J"], cfg["decim_set"],
                             cfg["p_drop"], rng)
        _, Gh, Bh, th, _ = fuse_proposed(t, Yd, m, cfg["Fs"], cfg["gain_max"],
                                         cfg["bias_max"], cfg["tau_max"], True)
        Ge.append(Gh); Gt.append(G)
        Be.append(Bh); Bt.append(B)
        Te.append(th * 1000); Tt.append(tau * 1000)
    Ge, Gt = np.concatenate(Ge), np.concatenate(Gt)
    Be, Bt = np.concatenate(Be), np.concatenate(Bt)
    Te, Tt = np.concatenate(Te), np.concatenate(Tt)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    panels = [
        (Gt, Ge, "true gain", f"Gain (MAE={np.mean(np.abs(Ge-Gt)):.4f})"),
        (Bt, Be, "true bias (deg/s)", f"Bias (MAE={np.mean(np.abs(Be-Bt)):.3f})"),
        (Tt, Te, "true delay (ms)", f"Delay (MAE={np.mean(np.abs(Te-Tt)):.3f} ms)"),
    ]
    for ax, (xt, xe, xl, ttl) in zip(axes, panels):
        ax.plot(xt, xe, ".", ms=4)
        lim = [xt.min(), xt.max()]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel(xl); ax.set_ylabel("estimated")
        ax.set_title(ttl); ax.grid(True, alpha=0.4)
    fig.suptitle(f"Parameter identification -- {sig}")
    fig.savefig(os.path.join(outdir, f"fig_param_{sig}.png"), dpi=200)
    plt.close(fig)
    return dict(gain_mae=float(np.mean(np.abs(Ge - Gt))),
                bias_mae=float(np.mean(np.abs(Be - Bt))),
                delay_mae_ms=float(np.mean(np.abs(Te - Tt))))


# =========================================================================
# SWEEPS (run on whatever cfg['sig'] is set; 'fluct' in the paper)
# =========================================================================
def exp_drop_sweep(cfg, n_runs, rng, outdir):
    drops = np.arange(0, 0.51, 0.1)
    Rn, Ri, Rp = (np.zeros(drops.size) for _ in range(3))
    for i, d in enumerate(drops):
        c = dict(cfg, p_drop=float(d))
        for _ in range(n_runs):
            t, wtrue, Y, *_ = make_data(c, False, rng)
            m = build_async_mask(c["N"], c["J"], c["decim_set"], c["p_drop"], rng)
            Rn[i] += rmse(fuse_nemec(zoh_fill(Y, m), c["gain_max"], c["bias_max"]), wtrue)
            Ri[i] += rmse(fuse_nemec(interp_fill(Y, m, t), c["gain_max"], c["bias_max"]), wtrue)
            Rp[i] += rmse(fuse_proposed(t, Y, m, c["Fs"], c["gain_max"],
                                        c["bias_max"], c["tau_max"], False)[0], wtrue)
    Rn /= n_runs; Ri /= n_runs; Rp /= n_runs
    _sweep_fig(drops * 100, [(Rn, "-o", "Nemec (ZOH)"),
                             (Ri, "-d", "Nemec (interp)"),
                             (Rp, "-s", "Proposed")],
               "packet loss (%)", "Robustness to packet loss",
               os.path.join(outdir, "fig_sweep_drop.png"))
    return dict(x=drops.tolist(), nemec_zoh=Rn.tolist(),
                nemec_interp=Ri.tolist(), proposed=Rp.tolist())


def exp_rate_sweep(cfg, n_runs, rng, outdir):
    sets = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 6], [1, 2, 3, 4, 6, 8]]
    xh = np.array([max(s) for s in sets])
    Rn, Ri, Rp = (np.zeros(len(sets)) for _ in range(3))
    for i, ds in enumerate(sets):
        c = dict(cfg, decim_set=ds)
        for _ in range(n_runs):
            t, wtrue, Y, *_ = make_data(c, False, rng)
            m = build_async_mask(c["N"], c["J"], c["decim_set"], c["p_drop"], rng)
            Rn[i] += rmse(fuse_nemec(zoh_fill(Y, m), c["gain_max"], c["bias_max"]), wtrue)
            Ri[i] += rmse(fuse_nemec(interp_fill(Y, m, t), c["gain_max"], c["bias_max"]), wtrue)
            Rp[i] += rmse(fuse_proposed(t, Y, m, c["Fs"], c["gain_max"],
                                        c["bias_max"], c["tau_max"], False)[0], wtrue)
    Rn /= n_runs; Ri /= n_runs; Rp /= n_runs
    _sweep_fig(xh, [(Rn, "-o", "Nemec (ZOH)"),
                    (Ri, "-d", "Nemec (interp)"),
                    (Rp, "-s", "Proposed")],
               "rate heterogeneity (max decimation factor)",
               "Robustness to heterogeneous sampling rates",
               os.path.join(outdir, "fig_sweep_rate.png"))
    return dict(x=xh.tolist(), nemec_zoh=Rn.tolist(),
                nemec_interp=Ri.tolist(), proposed=Rp.tolist())


def exp_tau_sweep(cfg, n_runs, rng, outdir):
    taus = np.array([0, 0.002, 0.004, 0.006, 0.008, 0.010])
    Rn, Ri, Rp0, Rp1 = (np.zeros(taus.size) for _ in range(4))
    for i, tm in enumerate(taus):
        c = dict(cfg, tau_max=float(tm))
        for _ in range(n_runs):
            t, wtrue, _, Yd, *_ = make_data(c, True, rng)
            m = build_async_mask(c["N"], c["J"], c["decim_set"], c["p_drop"], rng)
            tb = max(c["tau_max"], 1e-6)
            Rn[i] += rmse(fuse_nemec(zoh_fill(Yd, m), c["gain_max"], c["bias_max"]), wtrue)
            Ri[i] += rmse(fuse_nemec(interp_fill(Yd, m, t), c["gain_max"], c["bias_max"]), wtrue)
            Rp0[i] += rmse(fuse_proposed(t, Yd, m, c["Fs"], c["gain_max"],
                                         c["bias_max"], tb, False)[0], wtrue)
            Rp1[i] += rmse(fuse_proposed(t, Yd, m, c["Fs"], c["gain_max"],
                                         c["bias_max"], tb, True)[0], wtrue)
    Rn /= n_runs; Ri /= n_runs; Rp0 /= n_runs; Rp1 /= n_runs
    _sweep_fig(taus * 1000, [(Rn, "-o", "Nemec (ZOH)"),
                             (Ri, "-d", "Nemec (interp)"),
                             (Rp0, "-^", r"Proposed (no $\tau$)"),
                             (Rp1, "-s", r"Proposed ($\tau$)")],
               r"per-sensor time delay limit $\tau_{max}$ (ms)",
               "Robustness to clock skew / time delay",
               os.path.join(outdir, "fig_sweep_tau.png"))
    return dict(x=(taus * 1000).tolist(), nemec_zoh=Rn.tolist(),
                nemec_interp=Ri.tolist(), proposed_no_tau=Rp0.tolist(),
                proposed_tau=Rp1.tolist())


def exp_burst_sweep(cfg, n_runs, rng, outdir):
    """Correlated (bursty) packet loss at a fixed mean loss rate.

    Sweeps the mean burst length at a constant 20% stationary loss rate.
    Grid reconstructions must bridge ever longer per-sensor gaps, whereas
    the event-driven estimator simply keeps consuming whichever sensors
    are alive at each instant.
    """
    Ls = np.array([1, 10, 25, 50, 100])           # samples (10 ms .. 1 s)
    p_loss = 0.20
    Rn, Ri, Rp = (np.zeros(Ls.size) for _ in range(3))
    for i, L in enumerate(Ls):
        for _ in range(n_runs):
            t, wtrue, Y, *_ = make_data(cfg, False, rng)
            m = build_async_mask(cfg["N"], cfg["J"], cfg["decim_set"],
                                 p_loss, rng, burst_len=int(L))
            Rn[i] += rmse(fuse_nemec(zoh_fill(Y, m), cfg["gain_max"], cfg["bias_max"]), wtrue)
            Ri[i] += rmse(fuse_nemec(interp_fill(Y, m, t), cfg["gain_max"], cfg["bias_max"]), wtrue)
            Rp[i] += rmse(fuse_proposed(t, Y, m, cfg["Fs"], cfg["gain_max"],
                                        cfg["bias_max"], cfg["tau_max"], False)[0], wtrue)
    Rn /= n_runs; Ri /= n_runs; Rp /= n_runs
    x_ms = Ls / cfg["Fs"] * 1000
    _sweep_fig(x_ms, [(Rn, "-o", "Nemec (ZOH)"),
                      (Ri, "-d", "Nemec (interp)"),
                      (Rp, "-s", "Proposed")],
               "mean burst length (ms) at 20% stationary loss",
               "Robustness to bursty (correlated) packet loss",
               os.path.join(outdir, "fig_sweep_burst.png"))
    return dict(x_ms=x_ms.tolist(), loss=p_loss, nemec_zoh=Rn.tolist(),
                nemec_interp=Ri.tolist(), proposed=Rp.tolist())


def _sweep_fig(x, series, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(6.0, 3.9), constrained_layout=True)
    for y, style, lab in series:
        ax.plot(x, y, style, lw=1.6, label=lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE (deg/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description="Synthetic experiment suite")
    ap.add_argument("--runs", type=int, default=50,
                    help="Monte Carlo runs for the comparison tables (paper: 50)")
    ap.add_argument("--sweep-runs", type=int, default=10,
                    help="Monte Carlo runs per sweep point (paper: 10)")
    ap.add_argument("--param-runs", type=int, default=12,
                    help="runs for the parameter-accuracy scatter")
    ap.add_argument("--signals", nargs="+",
                    default=["cos", "fluct", "ramp"],
                    choices=["cos", "fluct", "ramp"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke test (runs=4, sweep-runs=2, param-runs=3)")
    ap.add_argument("--no-sweeps", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.runs, args.sweep_runs, args.param_runs = 4, 2, 3

    os.makedirs(args.outdir, exist_ok=True)
    labels = {"cos": "pure cosine", "fluct": "fluctuating-frequency",
              "ramp": "triangular ramp"}

    # Independent deterministic stream per experiment component: the suite
    # produces identical numbers whether components are run sequentially,
    # individually, or in parallel.
    def rng_for(tag):
        return np.random.default_rng([args.seed, COMPONENT_SEED[tag]])

    t0 = time.time()
    print(f"Running suite (runs={args.runs}, N={default_cfg()['N']}) "
          f"over {len(args.signals)} signal(s)...")

    results = dict(config=default_cfg(), runs=args.runs, seed=args.seed,
                   signals={}, sweeps={})
    SUM = np.zeros((len(args.signals), 3, 3))  # signal x regime x [nZOH nINT prop]
    for si, sig in enumerate(args.signals):
        cfg = dict(default_cfg(), sig=sig)
        R, wilcox = comparison_table(cfg, args.runs, labels[sig], rng_for(f"table:{sig}"))
        for rg, reg in enumerate(REGIMES):
            SUM[si, rg, 0] = R[reg][:, 2].mean()   # Nemec ZOH
            SUM[si, rg, 1] = R[reg][:, 3].mean()   # Nemec interp
            SUM[si, rg, 2] = R[reg][:, 4].mean()   # proposed
        pa = param_accuracy(cfg, args.param_runs, sig, rng_for(f"param:{sig}"), args.outdir)
        results["signals"][sig] = dict(
            rmse={reg: R[reg].tolist() for reg in REGIMES},
            methods=METHODS, wilcoxon=wilcox, param_accuracy=pa)
    summary_bar(SUM, [labels[s] for s in args.signals], args.outdir)

    if not args.no_sweeps:
        cfg = dict(default_cfg(), sig="fluct")
        results["sweeps"]["packet_loss"] = exp_drop_sweep(cfg, args.sweep_runs, rng_for("sweep:drop"), args.outdir)
        results["sweeps"]["rate_heterogeneity"] = exp_rate_sweep(cfg, args.sweep_runs, rng_for("sweep:rate"), args.outdir)
        results["sweeps"]["time_delay"] = exp_tau_sweep(cfg, args.sweep_runs, rng_for("sweep:tau"), args.outdir)
        results["sweeps"]["bursty_loss"] = exp_burst_sweep(cfg, args.sweep_runs, rng_for("sweep:burst"), args.outdir)

    with open(os.path.join(args.outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)

    print(f"\nAll done in {time.time()-t0:.1f} s. "
          f"Figures + results.json written to '{args.outdir}/'.")


if __name__ == "__main__":
    main()
