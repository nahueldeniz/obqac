import numpy as np
import pytest

from obqac import LTVLQR, NMPC, OBQAC, LeggedKinematics, Reference, circle, lemniscate, rectangle, simulate
from obqac.model import F, wrap


@pytest.fixture
def model():
    return LeggedKinematics(0.05)


def test_F_is_orthogonal():
    for th in np.linspace(-np.pi, np.pi, 7):
        assert np.allclose(F(th).T @ F(th), np.eye(4))


def test_one_step_inverse_is_exact(model):
    """With no constraints and delta_e = 1, u = F^T Delta / Ts lands on the target exactly."""
    q = np.array([0.7, 1.0, -2.0, 0.29])
    p = np.array([0.9, 1.02, -1.97, 0.29])
    u = model.unconstrained_control(q, p)
    assert np.allclose(model.step(q, u), p)


def test_lemniscate_matches_the_paper(model):
    """Eq. (21) at nu_r = 0.75 m/s and Ts = 0.05 s is one lap of ~535 samples (Figs. 2-4 run to ~530)."""
    path = lemniscate()
    assert 530 <= Reference(path, 0.75, 0.05).steps_per_lap() <= 540
    assert abs(path.sample()[0].max() - 1.75 * np.sqrt(2)) < 1e-3
    assert abs(path.sample()[1].max() - 2.0) < 1e-3


def test_reference_moves_at_the_requested_speed(model):
    ref = Reference(lemniscate(), 0.75, 0.05)
    for _ in range(50):
        a = ref.pose
        b = ref.advance()
        assert abs(np.hypot(b[1] - a[1], b[2] - a[2]) - 0.0375) < 2e-4


def test_closed_form_equals_qp(model):
    """For a diagonal W the QP of Eq. (7) is a component-wise clip."""
    rng = np.random.default_rng(1)
    qp, cf = OBQAC(model, solver="qp"), OBQAC(model, solver="closed_form")
    for _ in range(30):
        q = rng.normal(size=4)
        p0 = q + rng.normal(scale=2, size=4)
        p1 = p0 + rng.normal(scale=0.05, size=4)
        u1, u2 = qp(q, p0, p1), cf(q, p0, p1)
        assert np.allclose(u1, u2, atol=1e-6)


def test_obqac_respects_both_boxes(model):
    ctrl = OBQAC(model, delta_e=1.0)
    q = np.array([0.0, 0.0, 0.0, 0.29])
    p0 = np.array([3.0, 5.0, -5.0, 0.29])
    for _ in range(20):
        u_prev = ctrl.u_prev.copy()
        u = ctrl(q, p0, p0)
        assert np.all(np.abs(u) <= model.u_max + 1e-9)
        assert np.all(np.abs(u - u_prev) <= model.du_max + 1e-9)


def test_obqac_converges_on_the_lemniscate(model):
    for seed in range(3):
        r = simulate(OBQAC(model), lemniscate(), model, seed=seed)
        assert r.error[-1] < 0.05
        assert r.error[300] < 0.6


def test_error_decays_at_one_minus_delta_e(model):
    """Eq. (10): e_{k+1} = (1 - delta_e) e_k while no constraint is active."""
    ctrl = OBQAC(model, delta_e=0.05, K1=0.0, K2=0.0)
    ref = Reference(lemniscate(), 0.75, 0.05)
    q = ref.pose + np.array([0.0, 0.006, 0.008, 0.0])    # small: no limit active
    ctrl.u_prev = F(q[0]).T @ ref.velocity()        # already moving with the reference
    ratios = []
    for _ in range(20):
        p0, p1 = ref.pose, ref.pose_at(ref.s + 0.0375)
        e0 = np.linalg.norm((p0 - q)[1:3])
        u = ctrl(q, p0, p1)
        q = model.step(q, u)
        ref.advance()
        e1 = np.linalg.norm((ref.pose - q)[1:3])
        ratios.append(e1 / e0)
    assert np.allclose(ratios[5:], 0.95, atol=0.01)


def test_literal_eq7_does_not_converge(model):
    """delta_e = 1 under the paper's acceleration limits: the commanded body-frame velocity
    swings with the heading faster than the robot may accelerate, and the position error
    never closes (the reason the error gain of Section III-D is needed)."""
    r = simulate(OBQAC(model, delta_e=1.0, K1=0, K2=0), lemniscate(), model, seed=0)
    d_s = np.linalg.norm(r.p[:, 1:3] - r.q[1:, 1:3], axis=1)
    assert d_s[-1] > 1.0


def test_nmpc_short_horizon_is_worse_than_long(model):
    path = lemniscate()
    r1 = simulate(NMPC(model, horizon=1), path, model, seed=1, steps=400)
    r20 = simulate(NMPC(model, horizon=20), path, model, seed=1, steps=400)
    assert r20.error[-1] < r1.error[-1]
    assert r20.rtf > r1.rtf


def test_nmpc_respects_acceleration_limits(model):
    r = simulate(NMPC(model, horizon=5), lemniscate(), model, seed=2, steps=150)
    assert np.all(np.abs(np.diff(r.u, axis=0)) <= model.du_max + 1e-5)
    assert np.all(np.abs(r.u) <= model.u_max + 1e-6)


def test_lqr_tracks_but_ignores_acceleration_limits(model):
    r = simulate(LTVLQR(model), lemniscate(), model, seed=0, steps=300)
    assert r.error[-1] < 0.1
    assert np.abs(np.diff(r.u, axis=0)).max() > model.du_max.max()


def test_obqac_is_much_cheaper_than_nmpc(model):
    path = lemniscate()
    a = simulate(OBQAC(model), path, model, seed=0, steps=100)
    b = simulate(NMPC(model, horizon=20), path, model, seed=0, steps=100)
    assert a.rtf < 0.1 * b.rtf


def test_other_paths_are_closed_and_trackable(model):
    lap = Reference(circle(2.0), 0.75, 0.05).steps_per_lap()
    r = simulate(OBQAC(model), circle(2.0), model, seed=0, steps=2 * lap)
    assert r.error[lap:].mean() < 0.1               # second lap: converged
    path = rectangle(5.0, 4.0)                      # near-square corners at 0.75 m/s are cut,
    lap = Reference(path, 0.75, 0.05).steps_per_lap()   # as in the field runs (Fig. 7 middle)
    r = simulate(OBQAC(model), path, model, seed=0, steps=2 * lap)
    assert r.error[lap:].mean() < r.error[:lap].mean()
    assert np.all(r.error[lap:] < 2.0)
