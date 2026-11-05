import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .plots import LIGHT, PAL

ORDER = ["nominal", "disturbed", "dropout30", "dropout60", "no_replan", "odom_only", "pid", "ukf"]
NICE = {"nominal": "nominal", "disturbed": "slip + push", "dropout30": "30 % marker\ndropout",
        "dropout60": "60 % marker\ndropout", "no_replan": "no LiDAR\nreplanning", "odom_only": "odometry\nonly",
        "pid": "PID instead\nof LQR", "ukf": "UKF instead\nof EKF"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0), min(c + h, 1)


def flat(name):
    return NICE[name].replace("\n", " ")


def boxplot(ax, data, labels, face, **kw):
    ax.boxplot(data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor=face),
               medianprops=dict(color=PAL["est"]), **kw)


def plot_conditions(main, conds, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    for i, c in enumerate(conds):
        d = main[main.condition == c]
        k, n = int(d.success.sum()), len(d)
        lo, hi = wilson(k, n)
        col = PAL["truth"] if k / n > 0.9 else (PAL["obstacle"] if k / n > 0.5 else PAL["surprise"])
        ax[0].bar(i, k / n * 100, color=col, alpha=.9)
        pct = k / n * 100
        ax[0].errorbar(i, pct, yerr=[[pct - lo * 100], [hi * 100 - pct]], color="#333", capsize=3)
    ax[0].set_xticks(range(len(conds)), [flat(c) for c in conds], fontsize=8, rotation=35, ha="right")
    ax[0].set(ylabel="docking success [%]  (95 % Wilson CI)", ylim=(0, 105), title="Success rate over repeated trials")

    data = [main[(main.condition == c) & main.reached].dock_pos_err_cm.clip(upper=60) for c in conds]
    boxplot(ax[1], data, [NICE[c] for c in conds], "#cfe8df", showfliers=True)
    ax[1].axhline(5, color=PAL["surprise"], ls="--", lw=1)
    ax[1].text(0.6, 5.8, "5 cm tolerance", color=PAL["surprise"], fontsize=8)
    ax[1].set(ylabel="final position error [cm]", title="Docking accuracy (clipped at 60 cm)")
    ax[1].set_xticklabels([flat(c) for c in conds], rotation=35, ha="right", fontsize=8)

    data = [main[main.condition == c].track_rmse_ey_cm.dropna().clip(upper=30) for c in conds]
    boxplot(ax[2], data, [NICE[c] for c in conds], "#e6dcf0")
    ax[2].set(ylabel="cross-track RMSE [cm]", title="Tracking error (estimated pose vs reference)")
    ax[2].set_xticklabels([flat(c) for c in conds], rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_consistency(main, conds, path):
    cons = [c for c in conds if c != "odom_only"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, key, ideal, lab in ((ax[0], "nees_mean", 3, "mean NEES  (ideal 3)"),
                               (ax[1], "nis_mean", 2, "mean NIS  (ideal 2)")):
        groups = [main[main.condition == c][key] for c in cons]
        vals = [g.mean() for g in groups]
        err = [g.std() / np.sqrt(max(len(g), 1)) * 1.96 for g in groups]
        a.bar(range(len(cons)), vals, yerr=err, color=PAL["marker"], alpha=.85, capsize=3)
        a.axhline(ideal, color=PAL["surprise"], ls="--")
        a.set_xticks(range(len(cons)), [flat(c) for c in cons], fontsize=8, rotation=35, ha="right")
        a.set(ylabel=lab, title=lab.split("(")[0].strip())
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_noise_sweep(sweep, path):
    g = sweep.groupby(["q", "r"]).agg(nees=("nees_mean", "mean"), rmse=("est_rmse_pos_cm", "mean"),
                                      succ=("success", "mean"), nis=("nis_mean", "mean")).reset_index()
    qs, rs = sorted(g.q.unique()), sorted(g.r.unique())
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for a, key, title, cmap in ((ax[0], "nees", "mean NEES (ideal 3)", "RdYlGn_r"),
                                (ax[1], "rmse", "position RMSE [cm]", "viridis_r"),
                                (ax[2], "succ", "docking success", "viridis")):
        M = np.array([[g[(g.q == q) & (g.r == r)][key].iloc[0] for q in qs] for r in rs])
        norm = matplotlib.colors.LogNorm(vmin=max(M.min(), 0.3), vmax=M.max()) if key == "nees" else None
        im = a.imshow(M, origin="lower", cmap=cmap, aspect="auto", norm=norm)
        for i in range(len(rs)):
            for j in range(len(qs)):
                a.text(j, i, f"{M[i, j]:.2f}" if key != "rmse" else f"{M[i, j]:.1f}", ha="center", va="center",
                       fontsize=8, color="#111", bbox=dict(fc="white", ec="none", alpha=.55, pad=1))
        a.set_xticks(range(len(qs)), [str(q) for q in qs])
        a.set_yticks(range(len(rs)), [str(r) for r in rs])
        a.set(xlabel="process-noise scale $q$ (x config Q)", ylabel="measurement-noise scale $r$", title=title)
        a.grid(False)
        fig.colorbar(im, ax=a, fraction=.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_alternatives(main, path):
    have = set(main.condition)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    est_set = [(c, n) for c, n in (("nominal", "EKF"), ("ukf", "UKF"), ("odom_only", "odometry\nonly")) if c in have]
    data = [main[main.condition == c].est_rmse_pos_cm.clip(upper=150) for c, _ in est_set]
    boxplot(ax[0], data, [n for _, n in est_set], "#fde0c5")
    ax[0].set(ylabel="position RMSE [cm]", title="Estimator comparison", yscale="log")

    trk_set = [(c, n) for c, n in (("nominal", "LQR"), ("pid", "PID")) if c in have]
    for k, (key, lab) in enumerate((("track_rmse_ey_cm", "cross-track RMSE [cm]"),
                                    ("effort", "control effort $\\int(v^2+w^2)dt$"))):
        data = [main[main.condition == c][key].dropna() for c, _ in trk_set]
        boxplot(ax[1 + k], data, [n for _, n in trk_set], "#cfe8df" if k == 0 else "#e6dcf0")
        ax[1 + k].set(ylabel=lab, title="Controller comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_all(df, out):
    main = df[~df.condition.str.startswith("sweep")]
    conds = [c for c in ORDER if c in set(main.condition)]
    with plt.rc_context(LIGHT):
        plot_conditions(main, conds, out / "fig_conditions.png")
        plot_consistency(main, conds, out / "fig_consistency.png")
        sweep = df[df.condition.str.startswith("sweep")]
        if len(sweep):
            plot_noise_sweep(sweep, out / "fig_noise_sweep.png")
        plot_alternatives(main, out / "fig_alternatives.png")
