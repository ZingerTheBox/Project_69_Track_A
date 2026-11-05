from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle
from scipy.stats import chi2

from .. import config as C
from ..utils import wrap
from ..world import MARKER_BY_ID, MARKERS, dock_pose
from .metrics import nees_series
from .runlog import log_arrays

PAL = dict(truth="#1b9e77", est="#d95f02", ref="#7570b3", obstacle="#f28e2b", surprise="#d62728",
           wall="#444444", marker="#4e79a7", dock="#2ca02c", lidar="#59a14f", grid="#dddddd")
LIGHT = {"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#444",
         "text.color": "#222", "axes.labelcolor": "#222", "xtick.color": "#222", "ytick.color": "#222",
         "grid.color": "#e3e3e3", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True}
DARK = {"figure.facecolor": "#0e1117", "axes.facecolor": "#151a23", "axes.edgecolor": "#8b93a7",
        "text.color": "#e6e9f0", "axes.labelcolor": "#e6e9f0", "xtick.color": "#c3c8d6", "ytick.color": "#c3c8d6",
        "grid.color": "#2a3140", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True}


# --- helpers
def ellipse_xy(P2, n_sigma=2.0):
    w, V = np.linalg.eigh(P2[:2, :2])
    w = np.maximum(w, 1e-12)
    return 2 * n_sigma * np.sqrt(w[1]), 2 * n_sigma * np.sqrt(w[0]), np.degrees(np.arctan2(V[1, 1], V[0, 1]))


def wrap_error(xh, gt):
    e = xh - gt
    e[:, 2] = wrap(e[:, 2])
    return e


def save_fig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_world(ax, obstacles, surprise=None, dark=False, label_markers=True):
    wall = "#8b93a7" if dark else PAL["wall"]
    ax.add_patch(Rectangle((0, 0), C.ROOM_W, C.ROOM_H, fill=False, lw=2.5, ec=wall))
    for x, y, r in obstacles:
        ax.add_patch(Circle((x, y), r, fc=PAL["obstacle"], ec="none", alpha=.9, zorder=2))
    if surprise:
        ax.add_patch(Circle(surprise[:2], surprise[2], fc=PAL["surprise"], ec="none", alpha=.95, zorder=2))
        ax.annotate("unknown\nobstacle", surprise[:2], xytext=(surprise[0] - .1, surprise[1] + .45), ha="center",
                    fontsize=7.5, color=PAL["surprise"], arrowprops=dict(arrowstyle="-", color=PAL["surprise"], lw=.8))
    for m in MARKERS:
        n = np.array([np.cos(m.normal), np.sin(m.normal)])
        tang = np.array([-n[1], n[0]])
        a, b = m.pos - tang * .09, m.pos + tang * .09
        col = PAL["dock"] if m.id == C.DOCK_ID else PAL["marker"]
        ax.plot([a[0], b[0]], [a[1], b[1]], lw=5, color=col, solid_capstyle="butt", zorder=3)
        if label_markers:
            q = m.pos + n * .22
            ax.text(q[0], q[1], str(m.id), ha="center", va="center", fontsize=7, color=col, fontweight="bold")
    g = dock_pose()
    ax.add_patch(Circle(g[:2], .13, fc=PAL["dock"], ec="none", alpha=.28, zorder=1))
    ax.set_xlim(-.25, C.ROOM_W + .25)
    ax.set_ylim(-.25, C.ROOM_H + .25)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_robot(ax, pose, color, lw=2, zorder=6, fill=False):
    face = color if fill else "none"
    ax.add_patch(Circle(pose[:2], C.BODY_RADIUS, fc=face, ec=color, lw=lw, zorder=zorder, alpha=.95))
    d = np.array([np.cos(pose[2]), np.sin(pose[2])]) * C.BODY_RADIUS
    ax.plot([pose[0], pose[0] + d[0]], [pose[1], pose[1] + d[1]], color=color, lw=lw, zorder=zorder)


# --- static figures
def plot_run_overview(log, path, sc, title=None):
    a = log_arrays(log)
    with plt.rc_context(LIGHT):
        fig, ax = plt.subplots(figsize=(8.6, 7.2))
        draw_world(ax, sc.obstacles, sc.surprise)
        for i, (_, wp) in enumerate(log["paths"]):
            label = "planned path" if i == 0 else ("replanned path" if i == 1 else None)
            ax.plot(wp[:, 0], wp[:, 1], "--", color=PAL["ref"], lw=1.4, alpha=.8 if i else .5, label=label, zorder=3)
        ax.plot(a["gt"][:, 0], a["gt"][:, 1], color=PAL["truth"], lw=2.4, label="true path", zorder=4)
        ax.plot(a["xh"][:, 0], a["xh"][:, 1], color=PAL["est"], lw=1.6, ls=(0, (4, 2)), label="EKF estimate", zorder=5)
        for i in range(0, len(a["t"]), int(5 / C.DT_CTRL)):             # ellipse every 5 s
            w_, h_, ang = ellipse_xy(a["P"][i], 2.0)
            ax.add_patch(Ellipse(a["xh"][i, :2], max(w_, .02), max(h_, .02), angle=ang,
                                 fc=PAL["est"], alpha=.25, ec="none", zorder=5))
        draw_robot(ax, a["gt"][0], PAL["truth"])
        draw_robot(ax, a["gt"][-1], PAL["truth"], fill=True)
        ax.plot(*a["gt"][0, :2], "o", color=PAL["truth"], ms=5)
        ax.text(a["gt"][0, 0], a["gt"][0, 1] - .22, "start", ha="center", fontsize=8, color=PAL["truth"])
        ax.text(5.2, 2.9, "DOCK", fontsize=9, color=PAL["dock"], fontweight="bold", ha="center")
        meta = log["meta"]
        title = title or f"Track A - visual docking (seed {meta.get('seed', '?')}, {meta.get('backend', '')})"
        ax.set_title(title, fontsize=12, loc="left")
        ax.plot([], [], "s", color=PAL["est"], alpha=.3, label=r"2$\sigma$ position ellipse")
        ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=.92)
        fig.tight_layout()
        save_fig(fig, path)


def plot_estimation(log, path):
    a = log_arrays(log)
    t = a["t"]
    e = wrap_error(a["xh"].copy(), a["gt"])
    sig = np.sqrt(np.array([np.diag(P) for P in a["P"]]))
    nees = nees_series(a["xh"].copy(), a["P"], a["gt"])
    up = np.array(log["updates"]) if log["updates"] else np.zeros((0, 6))
    with plt.rc_context(LIGHT):
        fig, ax = plt.subplots(3, 2, figsize=(11, 7.2), sharex="col")
        for k, (lab, scale, unit) in enumerate([("x error", 100, "cm"), ("y error", 100, "cm"),
                                                ("heading error", 180 / np.pi, "deg")]):
            ax[k, 0].fill_between(t, -3 * sig[:, k] * scale, 3 * sig[:, k] * scale, color=PAL["est"], alpha=.2,
                                  label=r"$\pm3\sigma$ (EKF)")
            ax[k, 0].plot(t, e[:, k] * scale, color=PAL["truth"], lw=1.2, label="actual error")
            ax[k, 0].set_ylabel(f"{lab} [{unit}]")
        ax[0, 0].legend(loc="upper right", fontsize=8)
        ax[0, 0].set_title("Estimate error stays inside its own uncertainty", loc="left", fontsize=10)
        ax[2, 0].set_xlabel("time [s]")

        lo, hi = chi2.ppf([0.005, 0.995], 3)
        ax[0, 1].semilogy(t, np.maximum(nees, 1e-3), color=PAL["truth"], lw=1)
        ax[0, 1].axhspan(lo, hi, color="#999", alpha=.2, label="99 % chi$^2$(3) band")
        ax[0, 1].set(ylabel="NEES", title=f"NEES  (mean {nees.mean():.2f}, ideal 3)")
        ax[0, 1].legend(fontsize=8)

        if len(up):
            lo2, hi2 = chi2.ppf([0.005, 0.995], 2)
            ok = up[:, 5] > 0
            ax[1, 1].semilogy(up[ok, 0], np.maximum(up[ok, 4], 1e-4), ".", ms=2.5, color=PAL["marker"],
                              label="accepted")
            ax[1, 1].semilogy(up[~ok, 0], up[~ok, 4], "x", ms=4, color=PAL["surprise"], label="gated out")
            ax[1, 1].axhspan(lo2, hi2, color="#999", alpha=.2)
            ax[1, 1].set(ylabel="NIS", title=f"NIS per marker update (mean {up[ok, 4].mean():.2f}, ideal 2)")
            ax[1, 1].legend(fontsize=8)

        ax[2, 1].plot(t, sig[:, 0] * 100, label=r"$\sigma_x$ [cm]", color=PAL["est"])
        ax[2, 1].plot(t, sig[:, 1] * 100, label=r"$\sigma_y$ [cm]", color="#e7ba52")
        ax[2, 1].plot(t, np.degrees(sig[:, 2]), label=r"$\sigma_\theta$ [deg]", color=PAL["ref"])
        ax[2, 1].set(xlabel="time [s]", ylabel="EKF std",
                     title="Uncertainty: grows on odometry, collapses on marker sightings")
        ax[2, 1].legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, path)


def plot_control(log, path):
    a = log_arrays(log)
    t = a["t"]
    tr = a["phase"] == "TRACK"
    with plt.rc_context(LIGHT):
        fig, ax = plt.subplots(2, 2, figsize=(11, 5.6), sharex=True)
        ax[0, 0].plot(t, a["ref"][:, 3], color=PAL["ref"], ls="--", label="reference v")
        ax[0, 0].plot(t, a["cmd"][:, 0], color=PAL["truth"], label="command v")
        ax[0, 0].set(ylabel="v [m/s]", title="Linear speed")
        ax[0, 0].legend(fontsize=8)
        ax[0, 1].plot(t, a["ref"][:, 4], color=PAL["ref"], ls="--", label="reference w")
        ax[0, 1].plot(t, a["cmd"][:, 1], color=PAL["truth"], label="command w")
        ax[0, 1].set(ylabel="w [rad/s]", title="Angular speed")
        ax[0, 1].legend(fontsize=8)
        for k, (lab, col) in enumerate([("along-track $e_x$ [cm]", PAL["est"]),
                                        ("cross-track $e_y$ [cm]", PAL["marker"])]):
            ax[1, 0].plot(t[tr], a["err"][tr, k] * 100, label=lab, color=col)
        ax[1, 0].set(xlabel="time [s]", ylabel="[cm]", title="Tracking error (estimated pose vs reference)")
        ax[1, 0].legend(fontsize=8)
        ax[1, 1].plot(t[tr], np.degrees(a["err"][tr, 2]), color=PAL["ref"])
        ax[1, 1].set(xlabel="time [s]", ylabel="[deg]", title="Heading error")
        for ev_t, _ in log["events"]:
            for x in ax.ravel():
                x.axvline(ev_t, color="#999", lw=.8, ls=":")
        fig.tight_layout()
        save_fig(fig, path)


# --- dashboard video
def make_dashboard(log, sc, path, fps=20, stride=1, dpi=100):
    '''Render the dashboard to MP4 (ffmpeg) or GIF (Pillow), chosen by the file extension.'''
    a = log_arrays(log)
    t = a["t"]
    e = wrap_error(a["xh"].copy(), a["gt"])
    pe = np.hypot(e[:, 0], e[:, 1]) * 100
    sig = np.sqrt(np.array([np.diag(P) for P in a["P"]]))
    s3 = 3 * np.sqrt(np.array([np.linalg.eigvalsh(P[:2, :2]).max() for P in a["P"]])) * 100
    scans = log["scans"]
    up = np.array(log["updates"]) if log["updates"] else np.zeros((0, 6))
    frames = log["meta"].get("frames") or []
    idx = np.arange(0, len(t), stride)
    dist_goal = np.hypot(*(a["gt"][:, :2] - dock_pose()[:2]).T)

    with plt.rc_context(DARK):
        # --- layout
        fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
        gs = fig.add_gridspec(4 if frames else 3, 2, width_ratios=[1.55, 1],
                              height_ratios=[1, 1, 0.5, 0.8] if frames else [1, 1, 0.55],
                              left=.02, right=.98, top=.9, bottom=.07, wspace=.08, hspace=.55)
        axm = fig.add_subplot(gs[:, 0])
        axp = fig.add_subplot(gs[0, 1])
        axh = fig.add_subplot(gs[1, 1])
        axt = fig.add_subplot(gs[2, 1])
        axt.axis("off")

        # --- map panel
        draw_world(axm, sc.obstacles, sc.surprise, dark=True)
        for _, wp in log["paths"][:1]:
            axm.plot(wp[:, 0], wp[:, 1], "--", color=PAL["ref"], lw=1.2, alpha=.7, zorder=3)
        tr_gt, = axm.plot([], [], color=PAL["truth"], lw=2.4, zorder=4)
        tr_est, = axm.plot([], [], color=PAL["est"], lw=1.5, ls=(0, (4, 2)), zorder=5)
        ell = Ellipse((0, 0), .1, .1, fc=PAL["est"], alpha=.32, ec=PAL["est"], lw=1, zorder=5)
        axm.add_patch(ell)
        rob_gt = Circle((0, 0), C.BODY_RADIUS, fc="none", ec=PAL["truth"], lw=2.2, zorder=6)
        rob_est = Circle((0, 0), C.BODY_RADIUS, fc="none", ec=PAL["est"], lw=1.8, ls="--", zorder=6)
        axm.add_patch(rob_gt)
        axm.add_patch(rob_est)
        nose_gt, = axm.plot([], [], color=PAL["truth"], lw=2.2, zorder=6)
        nose_est, = axm.plot([], [], color=PAL["est"], lw=1.8, zorder=6)
        lid = axm.scatter([], [], s=5, c=PAL["lidar"], alpha=.8, zorder=3)
        det_lines = [axm.plot([], [], color="#f0e442", lw=1, alpha=.9, zorder=4)[0] for _ in range(4)]
        axm.plot([], [], color=PAL["truth"], lw=2.4, label="ground truth (evaluation only)")
        axm.plot([], [], color=PAL["est"], lw=1.5, ls=(0, (4, 2)), label=r"EKF estimate  +  2$\sigma$ ellipse")
        axm.plot([], [], color="#f0e442", lw=1, label="camera sighting (range, bearing)")
        axm.scatter([], [], s=8, c=PAL["lidar"], label="LiDAR returns")
        axm.legend(loc="lower left", fontsize=8, facecolor="#151a23", edgecolor="#2a3140", labelcolor="#e6e9f0")

        # --- side panels
        axp.set(xlim=(0, t[-1]), ylim=(0, max(15, min(pe.max() * 1.1, 40))), ylabel="position error [cm]")
        axp.set_title("Estimate error vs. its own 3$\\sigma$ bound", loc="left", fontsize=10)
        axp.fill_between(t, 0, s3, color=PAL["est"], alpha=.18)
        line_pe, = axp.plot([], [], color=PAL["truth"], lw=1.6)
        axh.set(xlim=(0, t[-1]), ylabel="distance to dock [m]", xlabel="time [s]")
        axh.set_title("Progress to the charging dock", loc="left", fontsize=10)
        axh.plot(t, dist_goal, color="#3a4256", lw=1)
        line_dg, = axh.plot([], [], color=PAL["dock"], lw=2)
        title = fig.text(.02, .945, "", fontsize=16, fontweight="bold")
        sub = fig.text(.02, .91, "", fontsize=10, color="#9aa3b5")
        info = axt.text(0, .9, "", fontsize=10, va="top", family="monospace")

        inset_cam = inset_top = None
        if frames:
            inset_gs = gs[3, 1].subgridspec(1, 2, wspace=.06)
            inset_cam = fig.add_subplot(inset_gs[0, 0])
            inset_cam.axis("off")
            inset_cam.set_title("robot camera", fontsize=8, color="#9aa3b5")
            inset_top = fig.add_subplot(inset_gs[0, 1])
            inset_top.axis("off")
            inset_top.set_title("MuJoCo top view", fontsize=8, color="#9aa3b5")
        im_c = im_t = None

        # --- one frame
        def update(n):
            nonlocal im_c, im_t
            i = idx[n]
            tt = t[i]
            tr_gt.set_data(a["gt"][:i + 1, 0], a["gt"][:i + 1, 1])
            tr_est.set_data(a["xh"][:i + 1, 0], a["xh"][:i + 1, 1])
            w_, h_, ang = ellipse_xy(a["P"][i], 2.0)
            ell.set_center(a["xh"][i, :2])
            ell.width, ell.height, ell.angle = max(w_, .03), max(h_, .03), ang
            for circ, nose, pose in ((rob_gt, nose_gt, a["gt"][i]), (rob_est, nose_est, a["xh"][i])):
                circ.center = pose[:2]
                d = np.array([np.cos(pose[2]), np.sin(pose[2])]) * C.BODY_RADIUS
                nose.set_data([pose[0], pose[0] + d[0]], [pose[1], pose[1] + d[1]])

            k = max([j for j, s in enumerate(scans) if s[0] <= tt], default=None)
            if k is not None:
                _, r, p = scans[k]
                ok = np.isfinite(r)
                ang_ = p[2] + np.linspace(0, 2 * np.pi, C.LIDAR_BEAMS, endpoint=False)[ok]
                lid.set_offsets(np.c_[p[0] + r[ok] * np.cos(ang_), p[1] + r[ok] * np.sin(ang_)])
            seen = up[(up[:, 0] <= tt) & (up[:, 0] > tt - .15)] if len(up) else []
            for j, ln in enumerate(det_lines):
                if j < len(seen):
                    mpos = MARKER_BY_ID[int(seen[j, 1])].pos
                    ln.set_data([a["gt"][i, 0], mpos[0]], [a["gt"][i, 1], mpos[1]])
                else:
                    ln.set_data([], [])

            line_pe.set_data(t[:i + 1], pe[:i + 1])
            line_dg.set_data(t[:i + 1], dist_goal[:i + 1])
            title.set_text("Visual docking  -  EKF + LQR on a TurtleBot3")
            sub.set_text(f"t = {tt:5.1f} s      phase: {a['phase'][i]:<6}      markers in view: {len(seen)}")
            info.set_text(f"position error  {pe[i]:5.1f} cm   (3-sigma bound {s3[i]:4.1f} cm)\n"
                          f"heading  error  {np.degrees(abs(e[i, 2])):5.2f} deg  "
                          f"(3-sigma {3 * np.degrees(sig[i, 2]):4.2f} deg)\n"
                          f"command  v = {a['cmd'][i, 0]:.2f} m/s   w = {a['cmd'][i, 1]:+.2f} rad/s\n"
                          "controller input: ESTIMATED pose only")
            if frames:
                f = min(frames, key=lambda q: abs(q["t"] - tt))
                if f.get("cam") is not None:
                    im_c = inset_cam.imshow(f["cam"]) if im_c is None else (im_c.set_data(f["cam"]) or im_c)
                top = f["top"][::4, ::4]
                im_t = inset_top.imshow(top) if im_t is None else (im_t.set_data(top) or im_t)

        # --- write
        ani = animation.FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if str(path).endswith(".gif"):
            ani.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        else:
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=4000, codec="libx264",
                                                extra_args=["-pix_fmt", "yuv420p"])
                ani.save(path, writer=writer, dpi=dpi)
            except Exception as exc:                                    # no ffmpeg: fall back to GIF
                gif = str(Path(path).with_suffix(".gif"))
                print(f"[dashboard] ffmpeg unavailable ({exc}); writing {gif} instead")
                ani.save(gif, writer=animation.PillowWriter(fps=min(fps, 15)), dpi=dpi)
        plt.close(fig)
