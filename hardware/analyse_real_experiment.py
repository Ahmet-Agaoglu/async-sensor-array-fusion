#!/usr/bin/env python3
# =========================================================================
# REAL-EXPERIMENT ANALYSIS
#
# Reproduces every analysis of the Real Experiments section from the three
# CSV recordings of the 16-sensor MPU9250 array:
#
#   A. BASELINE ANALYSIS: fused RMSE against the encoder reference,
#      per-sensor parameter estimates, convergence diagnostics, timing.
#   B. ENCODER-REFERENCE STUDY: the windowed finite difference acts as a
#      moving average of the true velocity, attenuating a sinusoidal
#      component of frequency f by sinc(pi f T_w). The study sweeps the
#      window length, quantifies the reference-induced error, and adopts a
#      0.25-s window (fundamental attenuation < 0.1 deg/s RMS, encoder
#      quantisation noise ~0.03 deg/s) as the corrected reference.
#   C. EMULATED ASYNCHRONY ON THE RECORDED DATA: the recorded event
#      streams are replayed with per-sensor i.i.d. and bursty
#      (Gilbert-Elliott) packet loss, comparing Nemec (ZOH),
#      Nemec (interp), and the proposed method on real measurements.
#
# Usage:
#   python analyse_real_experiment.py data/real_cos.csv data/real_fluct.csv \
#          data/real_ramp.csv --outdir results
#
# The estimator core (_kf_pass, fuse_nemec, ...) is imported from the
# synthetic suite so that exactly one implementation exists in the repo.
# =========================================================================

import argparse
import importlib.util
import json
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "synth", os.path.join(_HERE, "..", "synthetic", "run_experiments.py"))
synth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synth)
try:  # numba's on-disk cache cannot serve dynamically loaded modules
    from numba import njit
    synth._kf_pass = njit(cache=False)(synth._kf_pass_py)
except Exception:  # pragma: no cover
    synth._kf_pass = synth._kf_pass_py

EPS = np.finfo(float).eps


# =========================================================================
# CSV PARSING
# =========================================================================
def parse_csv(path):
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    J = (raw.shape[1] - 2) // 2
    t_enc = raw[:, 0] * 1e-6
    enc_deg = raw[:, 1]
    T_imu = raw[:, 2 + 2 * np.arange(J)] * 1e-6
    Y = raw[:, 3 + 2 * np.arange(J)]
    dt = np.diff(t_enc)
    dt = dt[(dt > 0) & (dt < 0.5)]
    Fs = 1.0 / np.median(dt)
    return t_enc, enc_deg, T_imu, Y, J, Fs


# =========================================================================
# ENCODER GROUND-TRUTH VELOCITY (window-centre attributed)
# =========================================================================
def encoder_gt(t_enc, enc_deg, Fs, win_s):
    N = t_enc.size
    w_win = int(round(win_s * Fs))
    w_half = w_win // 2
    gt = np.full(N, np.nan)
    k = np.arange(w_win, N)
    gt[k - w_half] = (enc_deg[k] - enc_deg[k - w_win]) / (t_enc[k] - t_enc[k - w_win])
    return gt


# =========================================================================
# PROPOSED FUSION on the recorded event stream (paper settings: 8 outer
# iterations, warm = min(50, M/10), q0 from the median inter-event gap)
# =========================================================================
def fuse_proposed_hw(t_enc, Y, T_imu, mask, gmax, bmax, tmax, est_tau,
                     n_iter=8, mu=3.0):
    N, J = Y.shape
    kk, jj = np.nonzero(mask)
    te = T_imu[kk, jj]
    ye = Y[kk, jj]
    je = jj.astype(np.int64)
    order = np.argsort(te, kind="stable")
    te, ye, je = te[order], ye[order], je[order]
    M = te.size

    q = (1.0 / np.median(np.diff(te))) ** 2 * 1e-3
    Ghat = np.ones(J)
    Bhat = np.zeros(J)
    tauhat = np.zeros(J)
    Rest = np.full(J, 0.1 ** 2)
    warm = min(50, M // 10)

    dbg = dict(gamma=[], q=[], G_hist=[], B_hist=[], tau_hist=[], M=M)
    for _ in range(n_iter):
        w_ev, wd_ev, innov2, cnt = synth._kf_pass(te, ye, je, Ghat, Bhat,
                                                  tauhat, Rest, q, warm)
        ratio = innov2 / cnt if cnt > 0 else np.nan
        if cnt > 0:
            q *= min(max(ratio, 0.3), 3.0) ** 0.7
            q = min(max(q, 1e1), 1e9)
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
            Rest[j] = max(np.var(ye[idx] - (Ghat[j] * w_ev[idx] + Bhat[j]),
                                 ddof=1), EPS)
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
        dbg["gamma"].append(float(ratio))
        dbg["q"].append(float(q))
        dbg["G_hist"].append(Ghat.copy())
        dbg["B_hist"].append(Bhat.copy())
        dbg["tau_hist"].append(tauhat.copy() * 1e3)

    te_rev_u, idx_rev = np.unique(te[::-1], return_index=True)
    last_idx = M - 1 - idx_rev
    from scipy.interpolate import interp1d
    w_grid = interp1d(te_rev_u, w_ev[last_idx], kind="linear",
                      fill_value="extrapolate", assume_sorted=True)(t_enc)
    return w_grid, Ghat, Bhat, tauhat, dbg


# =========================================================================
# GRID FRONT-ENDS on recorded data (row grid = control loops)
# =========================================================================
def interp_fill_hw(t_enc, Y, T_imu, mask):
    """Non-causal linear interpolation of each sensor's surviving samples
    (at their true time stamps) onto the loop grid."""
    N, J = Y.shape
    Yg = np.empty_like(Y)
    for j in range(J):
        k = np.nonzero(mask[:, j])[0]
        Yg[:, j] = np.interp(t_enc, T_imu[k, j], Y[k, j])
    return Yg


def replay_mask(N, J, p_drop, rng, burst_len=1, decim_set=None):
    """Per-sensor loss (and optional decimation) applied to the recorded
    loop indices; row 0 always kept."""
    mask = np.zeros((N, J), dtype=bool)
    for j in range(J):
        keep = np.ones(N, dtype=bool)
        if decim_set is not None:
            d = int(np.asarray(decim_set)[rng.integers(len(decim_set))])
            keep[:] = False
            keep[0:N:d] = True
        if p_drop > 0:
            if burst_len <= 1:
                keep &= rng.random(N) > p_drop
            else:
                p_bg = 1.0 / burst_len
                p_gb = p_drop / ((1 - p_drop) * burst_len)
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


def rmse_valid(w, gt):
    v = ~np.isnan(gt)
    return float(np.sqrt(np.mean((w[v] - gt[v]) ** 2)))


# =========================================================================
# A. BASELINE ANALYSIS (paper Table 2 + diagnostics figures)
# =========================================================================
def baseline_analysis(tag, t_enc, enc_deg, T_imu, Y, J, Fs, gmax, bmax, tmax,
                      gt_win, outdir):
    gt = encoder_gt(t_enc, enc_deg, Fs, gt_win)
    mask = T_imu > 0
    t0 = time.time()
    w_prop, G, B, tau, dbg = fuse_proposed_hw(t_enc, Y, T_imu, mask,
                                              gmax, bmax, tmax, True)
    t_fuse = time.time() - t0
    w_nem = synth.fuse_nemec(synth.zoh_fill(Y, mask), gmax, bmax)
    rp, rn = rmse_valid(w_prop, gt), rmse_valid(w_nem, gt)

    out = dict(rmse_proposed=rp, rmse_nemec=rn,
               gain=dict(mean=float(G.mean()), std=float(G.std(ddof=1)),
                         min=float(G.min()), max=float(G.max())),
               bias=dict(mean=float(B.mean()), std=float(B.std(ddof=1)),
                         min=float(B.min()), max=float(B.max())),
               tau_ms=dict(mean=float(tau.mean() * 1e3),
                           std=float(tau.std(ddof=1) * 1e3),
                           min=float(tau.min() * 1e3),
                           max=float(tau.max() * 1e3)),
               tau_vs_read_order_corr=float(np.corrcoef(
                   np.arange(J), tau)[0, 1]),
               spec_ok_gain=int(np.sum(np.abs(G - 1) <= gmax)),
               spec_ok_bias=int(np.sum(np.abs(B) <= bmax)),
               gamma_final=dbg["gamma"][-1], q_final=dbg["q"][-1],
               M=dbg["M"], fuse_time_s=t_fuse,
               us_per_event_total=t_fuse / dbg["M"] / 8 * 1e6,
               realtime_factor=float((t_enc[-1] - t_enc[0]) / t_fuse))

    # ---- figures (fusion overlay, parameter bars, convergence) ----
    v = ~np.isnan(gt)
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True,
                           constrained_layout=True)
    ax[0].plot(t_enc[v], gt[v], "k", lw=1.2, label="Encoder GT")
    ax[0].plot(t_enc, w_prop, "b-", lw=1.1, label="Proposed")
    ax[0].plot(t_enc, w_nem, "r--", lw=1.0, label="Nemec")
    ax[0].set_ylabel("angular velocity (deg/s)")
    ax[0].legend(loc="best", fontsize=8)
    ax[0].set_title(f"Fused velocity vs ground truth   "
                    f"[Proposed RMSE={rp:.3f}  Nemec RMSE={rn:.3f} deg/s]")
    ax[0].grid(alpha=0.4)
    ax[1].plot(t_enc[v], w_prop[v] - gt[v], "b-", lw=0.8, label="Proposed")
    ax[1].plot(t_enc[v], w_nem[v] - gt[v], "r--", lw=0.8, label="Nemec")
    ax[1].axhline(0, color="k", ls=":", lw=0.8)
    ax[1].set_ylabel("error (deg/s)")
    ax[1].set_xlabel("time (s)")
    ax[1].legend(loc="best", fontsize=8)
    ax[1].grid(alpha=0.4)
    fig.savefig(os.path.join(outdir, f"real_{tag}_1.png"), dpi=200)
    plt.close(fig)

    idx = np.arange(1, J + 1)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    axs[0].bar(idx, G - 1, color="#3366cc")
    axs[0].axhline(gmax, color="r", ls="--")
    axs[0].axhline(-gmax, color="r", ls="--")
    axs[0].set_ylim(-1.5 * gmax, 1.5 * gmax)
    axs[0].set_title(f"Gain deviation   std={G.std(ddof=1):.4f}")
    axs[0].set_xlabel("sensor index")
    axs[0].set_ylabel(r"$\alpha_j$ (gain deviation)")
    axs[1].bar(idx, B, color="#339966")
    axs[1].axhline(bmax, color="r", ls="--")
    axs[1].axhline(-bmax, color="r", ls="--")
    axs[1].set_ylim(-1.5 * bmax, 1.5 * bmax)
    axs[1].set_title(f"Bias   std={B.std(ddof=1):.3f} deg/s")
    axs[1].set_xlabel("sensor index")
    axs[1].set_ylabel(r"$\beta_j$ (deg/s)")
    axs[2].bar(idx, tau * 1e3, color="#b36b1a")
    axs[2].axhline(0, color="k", ls=":")
    axs[2].set_title(f"Time delay   std={tau.std(ddof=1)*1e3:.3f} ms")
    axs[2].set_xlabel("sensor index")
    axs[2].set_ylabel(r"$\tau_j$ (ms)")
    for a in axs:
        a.grid(alpha=0.4)
    fig.suptitle("Estimated per-sensor parameters")
    fig.savefig(os.path.join(outdir, f"real_{tag}_2.png"), dpi=200)
    plt.close(fig)

    it = np.arange(1, len(dbg["gamma"]) + 1)
    Gh = np.array(dbg["G_hist"])
    Bh = np.array(dbg["B_hist"])
    Th = np.array(dbg["tau_hist"])
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    axs[0].semilogy(it, dbg["gamma"], "o-")
    axs[0].axhline(1, color="k", ls="--")
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel(r"$\gamma$ (innovation ratio)")
    axs[0].set_title("Innovation consistency")
    ax2 = axs[1].twinx()
    axs[1].plot(it, np.sqrt(np.mean((Gh - 1) ** 2, axis=1)) * 1e3, "b-o",
                label="G RMS")
    ax2.plot(it, np.sqrt(np.mean(Bh ** 2, axis=1)), "r-s", label="B RMS")
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel(r"gain RMS deviation ($\times10^{-3}$)", color="b")
    ax2.set_ylabel("bias RMS (deg/s)", color="r")
    axs[1].set_title("Parameter convergence (G and B)")
    axs[2].plot(it, np.sqrt(np.mean(Th ** 2, axis=1)), "k-o")
    axs[2].set_xlabel("iteration")
    axs[2].set_ylabel(r"$\tau$ RMS (ms)")
    axs[2].set_title(r"Parameter convergence ($\tau$)")
    for a in axs:
        a.grid(alpha=0.4)
    fig.suptitle("Filter convergence diagnostics")
    fig.savefig(os.path.join(outdir, f"real_{tag}_3.png"), dpi=200)
    plt.close(fig)
    return out


# =========================================================================
# B. ENCODER-REFERENCE STUDY
# =========================================================================
def reference_study(tag, t_enc, enc_deg, T_imu, Y, Fs, gmax, bmax, tmax,
                    windows, outdir):
    mask = T_imu > 0
    w_prop, *_ = fuse_proposed_hw(t_enc, Y, T_imu, mask, gmax, bmax, tmax, True)
    w_nem = synth.fuse_nemec(synth.zoh_fill(Y, mask), gmax, bmax)
    rows = []
    for win in windows:
        gt = encoder_gt(t_enc, enc_deg, Fs, win)
        rows.append(dict(window_s=win,
                         rmse_proposed=rmse_valid(w_prop, gt),
                         rmse_nemec=rmse_valid(w_nem, gt)))
    # empirical reference-smoothing error: long-window GT vs short-window GT
    gt_long = encoder_gt(t_enc, enc_deg, Fs, windows[0])
    gt_short = encoder_gt(t_enc, enc_deg, Fs, windows[-1])
    v = ~np.isnan(gt_long) & ~np.isnan(gt_short)
    smoothing_rms = float(np.sqrt(np.mean((gt_long[v] - gt_short[v]) ** 2)))
    return dict(rows=rows, smoothing_rms_long_vs_short=smoothing_rms)


# =========================================================================
# C. EMULATED ASYNCHRONY ON THE RECORDED DATA
# =========================================================================
def replay_sweeps(tag, t_enc, enc_deg, T_imu, Y, Fs, gmax, bmax, tmax,
                  gt_win, n_seeds, outdir):
    N, J = Y.shape
    gt = encoder_gt(t_enc, enc_deg, Fs, gt_win)

    def run_point(p_drop, burst_len, seed):
        rng = np.random.default_rng([seed, int(p_drop * 1000), burst_len])
        m = replay_mask(N, J, p_drop, rng, burst_len)
        rn = rmse_valid(synth.fuse_nemec(synth.zoh_fill(Y, m), gmax, bmax), gt)
        ri = rmse_valid(synth.fuse_nemec(
            interp_fill_hw(t_enc, Y, T_imu, m), gmax, bmax), gt)
        rp = rmse_valid(fuse_proposed_hw(t_enc, Y, T_imu, m, gmax, bmax,
                                         tmax, True)[0], gt)
        return rn, ri, rp

    out = {}
    drops = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    res = np.array([[run_point(p, 1, s) for s in range(n_seeds)]
                    for p in drops])            # [pts, seeds, 3]
    out["packet_loss"] = dict(x=drops, nemec_zoh=res[:, :, 0].mean(1).tolist(),
                              nemec_interp=res[:, :, 1].mean(1).tolist(),
                              proposed=res[:, :, 2].mean(1).tolist())
    _plot_sweep([d * 100 for d in drops], res, "packet loss (%)",
                "Recorded data: robustness to packet loss",
                os.path.join(outdir, f"fig_real_sweep_drop_{tag}.png"))

    bursts = [1, 12, 30, 60, 120]               # loops (~8.5 ms each)
    resb = np.array([[run_point(0.2, L, s) for s in range(n_seeds)]
                     for L in bursts])
    x_ms = [L / Fs * 1000 for L in bursts]
    out["bursty_loss"] = dict(x_ms=x_ms, loss=0.2,
                              nemec_zoh=resb[:, :, 0].mean(1).tolist(),
                              nemec_interp=resb[:, :, 1].mean(1).tolist(),
                              proposed=resb[:, :, 2].mean(1).tolist())
    _plot_sweep(x_ms, resb, "mean burst length (ms) at 20% loss",
                "Recorded data: robustness to bursty packet loss",
                os.path.join(outdir, f"fig_real_sweep_burst_{tag}.png"))
    return out


def _plot_sweep(x, res, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(6.0, 3.9), constrained_layout=True)
    labels = ["Nemec (ZOH)", "Nemec (interp)", "Proposed"]
    styles = ["-o", "-d", "-s"]
    for k in range(3):
        m = res[:, :, k].mean(1)
        s = res[:, :, k].std(1, ddof=1)
        ax.errorbar(x, m, yerr=s, fmt=styles[k], lw=1.6, capsize=3,
                    label=labels[k])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE (deg/s)")
    ax.set_title(title)
    ax.grid(alpha=0.4)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvfiles", nargs="+")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--gt-win", type=float, default=1.0,
                    help="paper's original encoder window [s]")
    ap.add_argument("--gt-win-corrected", type=float, default=0.25,
                    help="corrected encoder window [s] (reference study)")
    ap.add_argument("--windows", type=float, nargs="+",
                    default=[1.0, 0.5, 0.25, 0.1])
    ap.add_argument("--seeds", type=int, default=5,
                    help="mask realizations per replay point")
    ap.add_argument("--sweep-signal", default="fluct",
                    help="tag of the recording used for the replay sweeps")
    ap.add_argument("--no-sweeps", action="store_true")
    args = ap.parse_args()

    gmax, bmax, tmax = 0.03, 5.0, 10e-3        # spec limits at +25 C
    os.makedirs(args.outdir, exist_ok=True)
    results = dict(spec=dict(gmax=gmax, bmax=bmax, tmax_ms=tmax * 1e3),
                   gt_win_original=args.gt_win,
                   gt_win_corrected=args.gt_win_corrected, profiles={})

    for path in args.csvfiles:
        tag = os.path.splitext(os.path.basename(path))[0].replace("real_", "")
        t_enc, enc_deg, T_imu, Y, J, Fs = parse_csv(path)
        print(f"\n=== {tag}: N={t_enc.size}  J={J}  Fs={Fs:.3f} Hz ===")
        prof = {}
        prof["original_reference"] = baseline_analysis(
            tag + "_orig", t_enc, enc_deg, T_imu, Y, J, Fs,
            gmax, bmax, tmax, args.gt_win, args.outdir)
        prof["corrected_reference"] = baseline_analysis(
            tag, t_enc, enc_deg, T_imu, Y, J, Fs,
            gmax, bmax, tmax, args.gt_win_corrected, args.outdir)
        prof["reference_study"] = reference_study(
            tag, t_enc, enc_deg, T_imu, Y, Fs, gmax, bmax, tmax,
            args.windows, args.outdir)
        for key in ("original_reference", "corrected_reference"):
            b = prof[key]
            print(f"  [{key:20s}] RMSE prop={b['rmse_proposed']:.3f} "
                  f"nemec={b['rmse_nemec']:.3f}  gamma={b['gamma_final']:.3f} "
                  f"tau_std={b['tau_ms']['std']:.3f} ms")
        if (not args.no_sweeps) and tag == args.sweep_signal:
            prof["replay"] = replay_sweeps(
                tag, t_enc, enc_deg, T_imu, Y, Fs, gmax, bmax, tmax,
                args.gt_win_corrected, args.seeds, args.outdir)
        results["profiles"][tag] = prof

    with open(os.path.join(args.outdir, "real_results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nAll done. Figures + real_results.json in '{args.outdir}/'.")


if __name__ == "__main__":
    main()
