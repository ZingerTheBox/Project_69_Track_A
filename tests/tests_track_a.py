import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src.control.lqr import body_error, lqr_track
from src.estimation.ekf import P0, update
from src.eval.metrics import summarise
from src.model import (controllability_matrix, discretise, error_model, lqr_gain, observability_rank,rank_and_sv)
from src.pipeline import run_trial
from src.utils import wrap
from src.world import MARKER_BY_ID, make_scenario

@pytest.mark.parametrize("v,w,rank", [(0.0, 0.0, 2), (0.18, 0.0, 3), (0.0, 0.5, 3), (0.18, 0.4, 3)])
def test_controllability_rank(v, w, rank):
    assert rank_and_sv(controllability_matrix(*error_model(v, w)))[0] == rank

def test_lqr_closed_loop_is_stable():
    Kr = lqr_gain(0.18, 0.0, C.Q_LQR, C.R_LQR)
    Ad, Bd = discretise(*error_model(0.18, 0.0), C.DT_CTRL)
    assert np.abs(np.linalg.eigvals(Ad - Bd @ Kr)).max() < 1.0

def test_observability_ranks():
    x0 = np.array([3.0, 2.5, 0.0])
    m0, m1, m2 = (MARKER_BY_ID[i].pos for i in (0, 4, 6))
    still, arc = [(0.0, 0.0)] * 5, [(0.02, 0.025)] * 20
    assert observability_rank(x0, still, [[]] * 6)[0] == 0
    assert observability_rank(x0, still, [[m0]] * 6)[0] == 2
    assert observability_rank(x0, still, [[m0, m1]] * 6)[0] == 3
    assert observability_rank(x0, arc, [[m0]] * 21)[0] == 2 # rotation
    assert observability_rank(x0, arc, [[m0, m1, m2]] * 21)[0] == 3


@pytest.mark.parametrize("kind", ["ekf", "ukf"])
def test_filter_corrects_a_biased_prior(kind):
    xh, P = np.array([3.0, 2.0, 0.3]), P0.copy()
    truth = np.array([3.1, 2.1, 0.2])
    for mid in (0, 1, 2, 4, 6):
        d = MARKER_BY_ID[mid].pos - truth[:2]
        for _ in range(5):
            xh, P, _, _ = update(xh, P, mid, float(np.hypot(*d)), float(wrap(np.arctan2(d[1], d[0]) - truth[2])), kind)
    assert np.hypot(*(xh[:2] - truth[:2])) < 0.03 and abs(wrap(xh[2] - truth[2])) < np.radians(2)
    assert np.all(np.linalg.eigvalsh(P) > 0)

def test_ekf_rejects_outlier():
    xh, P, nis, ok = update(np.array([3.0, 2.5, 0.0]), P0.copy(), 0, 9.0, 1.5) # absurd range
    assert not ok and nis > C.GATE_CHI2

def test_tracker_is_zero_on_reference():
    v, w, e = lqr_track(np.array([1.0, 1.0, 0.5]), (1.0, 1.0, 0.5, 0.18, 0.1))
    assert np.allclose(e, 0) and (v, w) == pytest.approx((0.18, 0.1))
    assert np.allclose(body_error(np.array([0, 0, np.pi / 2]), (0, 1, np.pi / 2, 0, 0)), [1, 0, 0])

@pytest.fixture(scope="module")
def nominal():
    sc = make_scenario(3)
    return sc, run_trial(sc)

def test_nominal_trial_docks(nominal):
    sc, log = nominal
    m = summarise(log, sc)
    assert m["success"] and not m["collided"] and m["dock_pos_err_cm"] < 5 and m["dock_hdg_err_deg"] < 5
    assert m["n_replans"] >= 1 # hidden obstacle avoide

def test_estimate_is_consistent(nominal):
    sc, log = nominal
    assert 0.5 < summarise(log, sc)["nees_mean"] < 8

def test_odometry_alone_fails_to_dock():
    sc = make_scenario(3)
    assert not summarise(run_trial(sc, estimator="odom"), sc)["success"]
